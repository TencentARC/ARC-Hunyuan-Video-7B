import torch
import math
import librosa
import argparse
import time
import numpy as np
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from torchvision.transforms.functional import InterpolationMode
from transformers import WhisperFeatureExtractor
import model_vllm.monkey_patch_mrope
from model_vllm import VideoAudioLLM

from decord import VideoReader, cpu

class VideoProcessorConfig:
    """配置视频处理相关参数"""
    def __init__(self):
        self.image_size = 640
        self.max_num_frame = 150
        self.factor = 2
        self.dtype = torch.bfloat16
        self.device_enc = "cuda:0"
        self.device_llm = "cuda:1"
        self.pretrain_path = 'TencentARC/ARC-Hunyuan-Video-7B'
        self.hunyuan_mean = (0.48145466, 0.4578275, 0.40821073)
        self.hunyuan_std = (0.26862954, 0.26130258, 0.27577711)
        self.generation_config = {
            'max_new_tokens': 1024,
            'do_sample': False,
            'top_p': 0.7,
            'repetition_penalty': 1.1,
        }

class HunyuanVideoProcessor:
    def __init__(self, config: VideoProcessorConfig):
        self.config = config
        self.wav_processor = self._init_audio_processor()
        self.model = self._init_model_and_tokenizer()

    def _init_audio_processor(self) -> WhisperFeatureExtractor:
        return WhisperFeatureExtractor.from_pretrained(self.config.pretrain_path)

    def _init_model_and_tokenizer(self):
        video_audio_llm = VideoAudioLLM(
            model_path=self.config.pretrain_path,
            device_enc=self.config.device_enc,
            device_llm=self.config.device_llm,
            temperature=0.0,
            max_tokens=1024,
        )

        return video_audio_llm

    def build_video_transform(self) -> T.Compose:
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.config.image_size, self.config.image_size),
                    interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.config.hunyuan_mean, std=self.config.hunyuan_std)
        ])

    def sec2hms(self, seconds):
        seconds = int(round(seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def add_timestamp_to_frame(self, frame, start_sec, end_sec, font_size=40):
        draw = ImageDraw.Draw(frame)
        font_size = int(frame.height * 0.05)
        font = ImageFont.truetype("ARIAL.TTF", font_size)
        text = f"{self.sec2hms(start_sec)}-{self.sec2hms(end_sec)}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = frame.width - text_w - 20
        y = 20
        draw.rectangle([x-10, y-10, x+text_w+10, y+text_h+10], fill=(0,0,0,180))
        draw.text((x, y), text, fill=(255,255,255), font=font)
        return frame

    def load_video_frames(self, video_path: str) -> tuple[torch.Tensor, list]:
        video_reader = VideoReader(video_path, ctx=cpu(0), num_threads=4)
        vlen = len(video_reader)
        input_fps = video_reader.get_avg_fps()
        duration = vlen / input_fps

        transform = self.build_video_transform()
        frame_indices, intervals_sec = self._calculate_frame_indices(vlen, input_fps, duration)

        pixel_values = []
        for i, idx in enumerate(frame_indices):
            frame = Image.fromarray(video_reader[idx].asnumpy())
            start_sec, end_sec = intervals_sec[i]
            frame = self.add_timestamp_to_frame(frame, start_sec, end_sec)
            pixel_values.append(transform(frame))

        return torch.stack(pixel_values).to(self.config.dtype), [1]*len(pixel_values)

    def _calculate_frame_indices(self, vlen: int, fps: float, duration: float) -> list:
        frames_per_second = fps

        if duration <= self.config.max_num_frame:
            interval = 1
            intervals = [(int(i * interval * frames_per_second), int((i + 1) * interval * frames_per_second)) for i in range(math.ceil(duration))]
            intervals_sec = [(int(i * interval), int((i + 1) * interval)) for i in range(math.ceil(duration))]
        else:
            num_segments = self.config.max_num_frame
            segment_duration = duration / num_segments
            intervals = [(int(i * segment_duration * frames_per_second), int((i + 1) * segment_duration * frames_per_second)) for i in range(num_segments)]
            intervals_sec = [(round(i * segment_duration), round((i + 1) * segment_duration)) for i in range(num_segments)]

        frame_indices = []
        for start, end in intervals:
            if end > vlen:
                end = vlen
            frame_indices.append((start + end) // 2)

        return frame_indices, intervals_sec

    def cut_audio_with_librosa(self, audio_path, max_num_frame=150, segment_sec=2, max_total_sec=300, sr=16000):
        audio, _ = librosa.load(audio_path, sr=sr)
        total_samples = len(audio)
        total_sec = total_samples / sr

        if total_sec <= max_total_sec:
            return audio, sr

        segment_length = total_samples // max_num_frame
        segment_samples = int(segment_sec * sr)
        segments = []
        for i in range(max_num_frame):
            start = i * segment_length
            end = min(start + segment_samples, total_samples)
            segments.append(audio[start:end])
        new_audio = np.concatenate(segments)
        return new_audio, sr

    def process_audio(self, video_path: str, audio_path: str) -> torch.Tensor:
        video = VideoFileClip(video_path)
        try:
            video.audio.write_audiofile(audio_path, logger=None)
            audio, sr = self.cut_audio_with_librosa(audio_path, max_num_frame=150, segment_sec=2, max_total_sec=300, sr=16000)
        except:
            # when no audios
            duration = min(math.ceil(video.duration), 300)
            silent_audio = AudioSegment.silent(duration=duration * 1000)
            silent_audio.export(audio_path, format="mp3")
            print('no audio', audio_path)
            audio, sr = librosa.load(audio_path, sr=16000)

        audio = self._pad_audio(audio, sr)
        duration = math.ceil(len(audio) / sr)

        return self._extract_spectrogram(audio, sr), duration

    def _pad_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if len(audio.shape) == 2:
            audio = audio[:, 0]
        if len(audio) < sr:
            sil = np.zeros(sr - len(audio), dtype=float)
            audio = np.concatenate((audio, sil), axis=0)
        return audio

    def _extract_spectrogram(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        segment_length = sr * 30
        spectrograms = []

        for i in range(0, len(audio), segment_length):
            segment = audio[i:i+segment_length]
            spectrograms.append(
                self.wav_processor(segment, sampling_rate=sr, return_tensors="pt")["input_features"]
            )
        return torch.cat(spectrograms).to(self.config.dtype)

    def _build_prompt(self, num_frames: int, question: str, task: str) -> str:
        video_prefix = '<image>' * num_frames

        if task == "MCQ":
            return f"{video_prefix}\n{question}\nOutput the thinking process in <think> </think> and final answer (only option index) in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>.<sep>"
        elif task == "Grounding":
            return f"{video_prefix}\n{question}\nOutput the thinking process in <think> </think> and final answer (only time range) in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>.<sep>"
        else:  # QA、summary、segment
            return f"{video_prefix}\n{question}\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>.<sep>"

    def generate_summary(self, video_path: str, audio_path: str, question: str = "", task: str = "summary") -> str:
        pixel_values, num_patches = self.load_video_frames(video_path)
        audio_features, duration = self.process_audio(video_path, audio_path)

        # avoid extreme cases
        if duration < pixel_values.shape[0]:
            pixel_values = pixel_values[:duration]

        if duration <= self.config.max_num_frame:
            duration = pixel_values.shape[0]
        else:
            assert pixel_values.shape[0] == self.config.max_num_frame

        prompt = self._build_prompt(pixel_values.shape[0], question, task)

        batch = [
            {
                "pixel_values": pixel_values,
                "audio_values": audio_features,
                "duration": duration,
                "text_prompt": prompt,
            }
        ]

        outputs = self.model(batch)
        response = outputs[0]['output']

        return response


    def generate_summary_batch(self, video_paths: list[str], audio_paths: list[str], questions: list[str], tasks: list[str]) -> list[str]:
        # Read all data, this should be done with a dataloader for faster offline inference
        batch = []

        for video_path, audio_path, question, task in zip(video_paths, audio_paths, questions, tasks):
            pixel_values, num_patches = self.load_video_frames(video_path)
            audio_features, duration = self.process_audio(video_path, audio_path)

            # avoid extreme cases
            if duration < pixel_values.shape[0]:
                pixel_values = pixel_values[:duration]

            if duration <= self.config.max_num_frame:
                duration = pixel_values.shape[0]
            else:
                assert pixel_values.shape[0] == self.config.max_num_frame

            prompt = self._build_prompt(pixel_values.shape[0], question, task)

            batch.append({
                "pixel_values": pixel_values,
                "audio_values": audio_features,
                "duration": duration,
                "text_prompt": prompt,
            })

        outputs = self.model(batch)

        responses = [output['output'] for output in outputs]

        return responses


def get_sample_data():
    video_paths, audio_paths, questions, tasks = [], [], [], []

    # First video test cases
    video_path = 'examples/demo1.mp4'
    audio_path = 'examples/temp.mp3'

    # Summary task
    video_paths.append(video_path)
    audio_paths.append(audio_path)
    tasks.append('summary')
    title = "白金枪鱼寿司的陷阱"
    questions.append(f"该视频标题为{title}\n描述视频内容.")

    # Grounding task
    video_paths.append(video_path)
    audio_paths.append(audio_path)
    tasks.append('Grounding')
    questions.append("我们何时能看到一个穿制服的男人站在菊花门前?")

    # QA task
    video_paths.append(video_path)
    audio_paths.append(audio_path)
    tasks.append('QA')
    questions.append("这个视频有哪些笑点？")

    # MCQ task
    video_paths.append(video_path)
    audio_paths.append(audio_path)
    tasks.append('MCQ')
    questions.append("视频中最后老板提供了什么给顾客作为赠品？\nA.纸尿裤\nB.寿司\nC.现金\nD.面巾纸")

    # Second video test cases
    video_path = 'examples/demo3.mov'
    audio_path = 'examples/temp.mp3'

    # Multi-granularity caption task
    video_paths.append(video_path)
    audio_paths.append(audio_path)
    tasks.append('segment')
    questions.append("请按时间顺序给出视频的章节摘要和对应时间点")

    # Third video test cases
    video_path = 'examples/demo2.mp4'
    audio_path = 'examples/temp.mp3'

    # Summary task
    video_paths.append(video_path)
    audio_paths.append(audio_path)
    tasks.append('summary')
    title = ""
    questions.append(f"The title of the video is{title}\nDescribe the video content.")

    # Grounding task
    video_paths.append(video_path)
    audio_paths.append(audio_path)
    tasks.append('Grounding')
    questions.append("When will we be able to see the man in the video eat the pork cutlet in the restaurant?")

    # QA task
    video_paths.append(video_path)
    audio_paths.append(audio_path)
    tasks.append('QA')
    questions.append("Why is the man dissatisfied with the pork cutlet he cooked himself at home?")

    # Multi-granularity caption task
    video_paths.append(video_path)
    audio_paths.append(audio_path)
    tasks.append('segment')
    questions.append("Localize video chapters with temporal boundaries and the corresponding sentence description.")

    return video_paths, audio_paths, questions, tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_inference", action="store_true")
    args = parser.parse_args()

    config = VideoProcessorConfig()
    processor = HunyuanVideoProcessor(config)

    video_paths, audio_paths, questions, tasks = get_sample_data()

    # Original: Total inference time: 161.21281695365906 seconds
    start_time = time.time()
    if not args.batch_inference:
        for video_path, audio_path, question, task in zip(video_paths, audio_paths, questions, tasks):
            response = processor.generate_summary(video_path, audio_path, question, task)
            print(question, '\n', response, '\n')
        # Sequential: Total inference time: 87.79057502746582 seconds
        print(f"Total inference time: {time.time() - start_time} seconds")
    else:
        responses = processor.generate_summary_batch(video_paths, audio_paths, questions, tasks)
        for question, response in zip(questions, responses):
            print(question, '\n', response, '\n')
        # Batch: Total inference time: 55.16641354560852 seconds
        print(f"Total inference time: {time.time() - start_time} seconds")


if __name__ == "__main__":
    main()
