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
    return f"<|startoftext|>{video_prefix}\n{question}"


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

    video_path = "examples/无极.mp4"
    audio_path = None

    video_paths.append(video_path)
    audio_paths.append(audio_path)
    tasks.append("summary")
    title = ""
    prompt = f"你是一个视频内容总结助手，你需要按照如下规则根据视频及其标题生成简要描述：\n1. 请以不超过100字的文本简要描述视频的主要内容，请仅准确反映视频中的核心内容，不要引入任何视频中未出现的信息\n2. 请在简要描述中保留视频中的核心人物、事件、场景及可能引人关注的信息\n视频为：<video>\n该视频标题为{title}\n请根据规则为给定视频内容生成该视频的简要描述。"
    questions.append(prompt)

    return video_paths, audio_paths, questions, tasks


def main():
    # Path of your instruction tuned model
    model_path = "work_dirs/brief_summary_sft/checkpoint-500"

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
