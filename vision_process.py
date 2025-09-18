from __future__ import annotations

import logging
import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from functools import lru_cache
from typing import Optional

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


logger = logging.getLogger(__name__)

## for fixed resolution
VIDEO_HEIGHT = 392
VIDEO_WIDTH = 392
MAX_FRAMES = 300


def sec2hms(seconds):
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def add_timestamp_to_frame(frame, start_sec, end_sec, font_size=40):
    draw = ImageDraw.Draw(frame)
    font_size = min(int(frame.width * 0.06), int(frame.height * 0.06))
    font = ImageFont.truetype("ARIAL.TTF", font_size)
    text = f"{sec2hms(start_sec)}-{sec2hms(end_sec)}"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = frame.width - text_w - 20
    y = 20
    draw.rectangle([x-10, y-10, x+text_w+10, y+text_h+10], fill=(0,0,0,180))
    draw.text((x, y), text, fill=(255, 0, 0), font=font)
    return frame

def _calculate_frame_indices(vlen: int, fps: float, duration: float) -> list:
    """计算采样帧索引"""
    frames_per_second = fps

    if duration <= MAX_FRAMES:
        interval = 1
        intervals = [(int(i * interval * frames_per_second), int((i + 1) * interval * frames_per_second)) for i in range(math.ceil(duration))]
        intervals_sec = [(int(i * interval), int((i + 1) * interval)) for i in range(math.ceil(duration))]
        sample_fps = 1
    else:
        num_segments = MAX_FRAMES
        segment_duration = duration / num_segments
        intervals = [(int(i * segment_duration * frames_per_second), int((i + 1) * segment_duration * frames_per_second)) for i in range(num_segments)]
        intervals_sec = [(round(i * segment_duration), round((i + 1) * segment_duration)) for i in range(num_segments)]
        sample_fps = 1 / segment_duration

    frame_indices = []
    for start, end in intervals:
        if end > vlen:
            end = vlen
        frame_indices.append((start + end) // 2)

    return frame_indices, intervals_sec, sample_fps
    
def _read_video_decord_arc(
    ele: dict,
) -> (torch.Tensor, float):
    from decord import VideoReader, cpu
    from pyav_reader import VideoReaderAV
    video_path = ele["video"]
    
    if not FORCE_PYAV:
        video_reader = VideoReader(video_path, ctx=cpu(0), num_threads=4)
    else:
        video_reader = VideoReaderAV(video_path)
    vlen = len(video_reader)
    input_fps = video_reader.get_avg_fps()
    duration = vlen / float(input_fps)

    frame_indices, intervals_sec, sample_fps = _calculate_frame_indices(vlen, input_fps, duration)

    processed_frames = []
    for i, idx in enumerate(frame_indices):
        if not FORCE_PYAV:
            frame = Image.fromarray(video_reader[idx].asnumpy()).convert('RGB')
        else:
            frame = Image.fromarray(video_reader[idx]).convert('RGB')
        start_sec, end_sec = intervals_sec[i]
        frame = add_timestamp_to_frame(frame, start_sec, end_sec)
        processed_np = np.array(frame)
        frame_tensor = torch.from_numpy(processed_np).permute(2, 0, 1)
        processed_frames.append(frame_tensor)
  
    if len(processed_frames) % 2 != 0 and len(processed_frames) != 1:
        processed_frames = processed_frames[:-1]
    elif len(processed_frames) == 1:
        processed_frames.append(frame_tensor)
    # Stack all processed frame tensors into a single video tensor (T, C, H, W)
    video_tensor = torch.stack(processed_frames)

    return video_tensor, sample_fps


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord_arc,
}

FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)
FORCE_PYAV = os.getenv("FORCE_PYAV", "False") == "True"

@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    ## only support decord
    video_reader_backend = "decord"
    return video_reader_backend

def fetch_video_arc(ele: dict, return_video_sample_fps: bool = False) -> torch.Tensor | list[Image.Image]:
    assert isinstance(ele["video"], str)
    video_reader_backend = get_video_reader_backend()
    video, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele)

    nframes, _, height, width = video.shape
    video = transforms.functional.resize(
        video,
        [VIDEO_HEIGHT, VIDEO_WIDTH],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()

    if return_video_sample_fps:
        return video, sample_fps
    return video


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "video" in ele
                        or ele.get("type","") in ("video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:

    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if "video" in vision_info:
            video_input, video_sample_fps = fetch_video_arc(vision_info, return_video_sample_fps=True)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {'fps': video_sample_fps_list}
    return image_inputs, video_inputs
