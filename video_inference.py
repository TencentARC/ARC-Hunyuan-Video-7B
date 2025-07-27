import torch
import math
import numpy as np
import decord
from decord import VideoReader, cpu
from PIL import Image
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import librosa
import tempfile

from transformers import ARCHunyuanVideoProcessor, ARCHunyuanVideoForConditionalGeneration


def calculate_frame_indices(vlen: int, fps: float, duration: float) -> list:
    frames_per_second = fps

    if duration <= 150:
        interval = 1
        intervals = [
            (int(i * interval * frames_per_second), int((i + 1) * interval * frames_per_second))
            for i in range(math.ceil(duration))
        ]
        sample_fps = 1
    else:
        num_segments = 150
        segment_duration = duration / num_segments
        intervals = [
            (int(i * segment_duration * frames_per_second), int((i + 1) * segment_duration * frames_per_second))
            for i in range(num_segments)
        ]
        sample_fps = 1 / segment_duration

    frame_indices = []
    for start, end in intervals:
        if end > vlen:
            end = vlen
        frame_indices.append((start + end) // 2)

    return frame_indices, sample_fps


def load_video_frames(video_path: str):
    video_reader = VideoReader(video_path, ctx=cpu(0), num_threads=4)
    vlen = len(video_reader)
    input_fps = video_reader.get_avg_fps()
    duration = vlen / input_fps

    frame_indices, sample_fps = calculate_frame_indices(vlen, input_fps, duration)

    return [Image.fromarray(video_reader[idx].asnumpy()) for idx in frame_indices], sample_fps


def cut_audio_with_librosa(audio_path, max_num_frame=150, segment_sec=2, max_total_sec=300, sr=16000):
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


def pad_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    if len(audio.shape) == 2:
        audio = audio[:, 0]
    if len(audio) < sr:
        sil = np.zeros(sr - len(audio), dtype=float)
        audio = np.concatenate((audio, sil), axis=0)
    return audio


def load_audio(video_path, audio_path):
    if audio_path is None:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as temp_audio:
            audio_path = temp_audio.name
            video = VideoFileClip(video_path)
            try:
                video.audio.write_audiofile(audio_path, logger=None)
                audio, sr = cut_audio_with_librosa(
                    audio_path, max_num_frame=150, segment_sec=2, max_total_sec=300, sr=16000
                )
            except:
                duration = min(math.ceil(video.duration), 300)
                silent_audio = AudioSegment.silent(duration=duration * 1000)
                silent_audio.export(audio_path, format="mp3")
                audio, sr = librosa.load(audio_path, sr=16000)
    else:
        audio, sr = cut_audio_with_librosa(audio_path, max_num_frame=150, segment_sec=2, max_total_sec=300, sr=16000)

    audio = pad_audio(audio, sr)
    duration = math.ceil(len(audio) / sr)

    return audio, sr, duration


def build_prompt(question: str, num_frames: int, task: str = "summary"):
    video_prefix = "<image>" * num_frames

    if task == "MCQ":
        return f"<|startoftext|>{video_prefix}\n{question}\nOutput the thinking process in <think> </think> and final answer (only option index) in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>.<sep>"
    elif task == "Grounding":
        return f"<|startoftext|>{video_prefix}\n{question}\nOutput the thinking process in <think> </think> and final answer (only time range) in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>.<sep>"
    else:  # QA、summary、segment
        return f"<|startoftext|>{video_prefix}\n{question}\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>.<sep>"


def prepare_inputs(question: str, video_path: str, audio_path: str = None, task: str = "summary"):
    video_frames, sample_fps = load_video_frames(video_path)
    audio, sr, duration = load_audio(video_path, audio_path)

    # To solve mismatched duration between video and audio
    video_duration = int(len(video_frames) / sample_fps)
    audio_duration = duration

    # Truncate video frames to match audio duration
    # The audio duration will be truncated in the model
    duration = min(video_duration, audio_duration)
    if duration <= 150:
        video_frames = video_frames[: int(duration * sample_fps)]

    prompt = build_prompt(question, len(video_frames), task)

    video_inputs = {
        "video": video_frames,
        "video_metadata": {
            "fps": sample_fps,
        },
    }

    audio_inputs = {
        "audio": audio,
        "sampling_rate": sr,
        "duration": duration,
    }

    return prompt, video_inputs, audio_inputs


def inference(model, processor, question: str, video_path: str, audio_path: str = None, task: str = "summary"):
    prompt, video_inputs, audio_inputs = prepare_inputs(question, video_path, audio_path, task)
    inputs = processor(
        text=prompt,
        **video_inputs,
        **audio_inputs,
        return_tensors="pt",
    )

    inputs = inputs.to("cuda", dtype=torch.bfloat16)
    outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    output_text = processor.decode(outputs[0], skip_special_tokens=True)

    return output_text


def get_sample_data():
    video_paths, audio_paths, questions, tasks = [], [], [], []

    # First video test cases
    video_path = "examples/demo1.mp4"
    audio_path = None

    # Summary task
    video_paths.append(video_path)
    audio_paths.append(audio_path)
    tasks.append("summary")
    title = "白金枪鱼寿司的陷阱"
    questions.append(f"该视频标题为{title}\n描述视频内容.")

    # Grounding task
    video_paths.append(video_path)
    audio_paths.append(audio_path)
    tasks.append("Grounding")
    questions.append("我们何时能看到一个穿制服的男人站在菊花门前?")

    # QA task
    video_paths.append(video_path)
    audio_paths.append(audio_path)
    tasks.append("QA")
    questions.append("这个视频有什么笑点？")

    # MCQ task
    video_paths.append(video_path)
    audio_paths.append(audio_path)
    tasks.append("MCQ")
    questions.append("视频中最后老板提供了什么给顾客作为赠品？\nA.纸尿裤\nB.寿司\nC.现金\nD.面巾纸")

    # Second video test cases
    video_path = "examples/demo3.mov"
    audio_path = None

    # Multi-granularity caption task
    video_paths.append(video_path)
    audio_paths.append(audio_path)
    tasks.append("segment")
    questions.append("请按时间顺序给出视频的章节摘要和对应时间点")

    # Third video test cases
    video_path = "examples/demo2.mp4"
    audio_path = None

    # Summary task
    video_paths.append(video_path)
    audio_paths.append(audio_path)
    tasks.append("summary")
    title = ""
    questions.append(f"The title of the video is{title}\nDescribe the video content.")

    # Grounding task
    video_paths.append(video_path)
    audio_paths.append(audio_path)
    tasks.append("Grounding")
    questions.append("When will we be able to see the man in the video eat the pork cutlet in the restaurant?")

    # QA task
    video_paths.append(video_path)
    audio_paths.append(audio_path)
    tasks.append("QA")
    questions.append("Why is the man dissatisfied with the pork cutlet he cooked himself at home?")

    # Multi-granularity caption task
    video_paths.append(video_path)
    audio_paths.append(audio_path)
    tasks.append("segment")
    questions.append("Localize video chapters with temporal boundaries and the corresponding sentence description.")

    return video_paths, audio_paths, questions, tasks


def main():
    model_path = "TencentARC/ARC-Hunyuan-Video-7B"

    model = (
        ARCHunyuanVideoForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        .eval()
        .to("cuda")
    )

    processor = ARCHunyuanVideoProcessor.from_pretrained(model_path)

    for video_path, audio_path, question, task in zip(*get_sample_data()):
        output_text = inference(model, processor, question, video_path, audio_path, task)
        print(question, '\n', output_text, '\n')


if __name__ == "__main__":
    main()
