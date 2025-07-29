# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Modified from InternVL
# Copyright (c) 2025 ARC Lab
# Licensed under LICENSE [see LICENSE for details]
# --------------------------------------------------------

import io

from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
import os
import random
import re
from collections import Counter
from typing import Dict
import math
import cv2
import imageio
import numpy as np
import torch
import warnings
import torch.nn.functional as F
import torchvision.transforms as T
import transformers
from decord import VideoReader
from PIL import Image
from PIL import ImageDraw, ImageFont
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from torchvision.transforms.functional import InterpolationMode

from .constants import (CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD,
                        IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN,
                        VID_END_TOKEN, VID_START_TOKEN, LINE_TOKEN,
                        SIGLIP_MEAN, SIGLIP_STD, HUNYUAN_MEAN, HUNYUAN_STD)

xdrope_section = [
    0.25,
    0.25,
    0.25,
    0.25
  ]

def calculate_ngram_repetition(text, n):
    words = text.split()
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    ngram_counts = Counter(ngrams)
    total_ngrams = len(ngrams)
    repeated_ngrams = sum(1 for count in ngram_counts.values() if count > 1)
    return repeated_ngrams / total_ngrams if total_ngrams > 0 else 0

def check_conversations_repetition(conversations, repeat_threshold=0.4, ngram=10):
    for conversation in conversations:
        if conversation['from'] == 'gpt':
            model_answer = conversation['value']
            repeat_ratio = calculate_ngram_repetition(model_answer, ngram)
            if repeat_ratio > repeat_threshold:
                raise Exception

def get_frame_indices(vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    duration = vlen / input_fps

    frames_per_second = input_fps

    ## current support videos > 300s
    if duration > 300:
        warnings.warn("The video is longer than 5 minutes. Due to sampling, some audio information may be lost!")

    if duration <= max_num_frames:
        interval = 1
        intervals = [(int(i * interval * frames_per_second), int((i + 1) * interval * frames_per_second)) for i in range(math.ceil(duration))]
        intervals_sec = [(int(i * interval), int((i + 1) * interval)) for i in range(math.ceil(duration))]
    else:
        num_segments = max_num_frames
        segment_duration = duration / num_segments
        intervals = [(int(i * segment_duration * frames_per_second), int((i + 1) * segment_duration * frames_per_second)) for i in range(num_segments)]
        intervals_sec = [(round(i * segment_duration), round((i + 1) * segment_duration)) for i in range(num_segments)]

    frame_indices = []

    if sample == 'rand':
        for start, end in intervals:
            if end > vlen:
                end = vlen
            frame_indices.append(random.choice(range(start, end)))
    elif sample == 'middle':
        for start, end in intervals:
            if end > vlen:
                end = vlen
            frame_indices.append((start + end) // 2)
    else:
        raise NotImplementedError

    return frame_indices, intervals_sec

def seconds_to_mmss(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def sec2hms(seconds):
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def add_timestamp_to_frame(frame, start_sec, end_sec, font_size=40):
    draw = ImageDraw.Draw(frame)
    font_size = int(frame.height * 0.05)
    font = ImageFont.truetype("ARIAL.TTF", font_size)
    text = f"{sec2hms(start_sec)}-{sec2hms(end_sec)}"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = frame.width - text_w - 20
    y = 20
    draw.rectangle([x-10, y-10, x+text_w+10, y+text_h+10], fill=(0,0,0,180))
    draw.text((x, y), text, fill=(255,255,255), font=font)
    return frame

def read_frames_decord(
        video_path, num_frames, sample='rand', fix_start=None,
        client=None, clip=None, use_time=False, min_num_frames=4
):
    video_reader = VideoReader(video_path, num_threads=1)

    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)

    frame_indices, intervals_sec = get_frame_indices(
        vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, max_num_frames=num_frames
    )

    frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C), np.uint8
    frames = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]

    if use_time:
        frames_with_ts = []
        for i, frame in enumerate(frames):
            start_sec, end_sec = intervals_sec[i]
            frame_with_ts = add_timestamp_to_frame(frame, start_sec, end_sec)
            frames_with_ts.append(frame_with_ts)
        frames = frames_with_ts

        save_dir = 'output_frames'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

            for idx, frame in enumerate(frames):
                video_name = video_path.split('/')[-1].replace('.mp4', '')
                frame.save(os.path.join(save_dir, f'{video_name}_frame_{idx:03d}.jpg'))

    return frames

def extract_frame_number(filename):
    # Extract the numeric part from the filename using regular expressions
    match = re.search(r'_(\d+).jpg$', filename)
    return int(match.group(1)) if match else -1

def sort_frames(frame_paths):
    # Extract filenames from each path and sort by their numeric part
    return sorted(frame_paths, key=lambda x: extract_frame_number(os.path.basename(x)))

def read_frames_folder(
        video_path, num_frames, sample='rand', fix_start=None,
        client=None, clip=None, min_num_frames=4
):
    image_list = sort_frames(list(os.listdir(video_path)))
    frames = []
    for image in image_list:
        fp = os.path.join(video_path, image)
        frame = Image.open(fp).convert('RGB')
        frames.append(frame)

    return frames

class WeightedConcatDataset(ConcatDataset):
    def __init__(self, datasets, weights):
        super().__init__(datasets)
        self.weights = torch.DoubleTensor(weights)
        self.total_size = sum(len(d) for d in datasets)
        self.sampler = WeightedRandomSampler(weights=self.weights, num_samples=self.total_size, replacement=True)

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        return self.total_size

def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    img = Image.open(buff)
    return img.convert('RGB')

class TCSLoader(object):

    def __init__(self, conf_path, sc_config_key='sensecore'):
        print(f'[TCSLoader] config_path: {conf_path}')
        # print('--> before Client(conf_path)')
        # self.client = Client(conf_path)
        # self.sc_config_key = sc_config_key
        # print('--> after Client(conf_path)')
        self.client = None

    def __call__(self, fn, image_type='image', max_num_frames=-1, min_num_frames=8, sample='rand', use_time=False, clip=None):
        #print(image_type, max_num_frames, min_num_frames, clip)
        if image_type == 'image':
            img_value_str = self.client.get(fn)
            img = pil_loader(img_value_str)
            return img

        elif image_type == 'video':
            if fn.endswith('/'):
                frames = read_frames_folder(fn, num_frames=max_num_frames, min_num_frames=min_num_frames,
                                            client=self.client, sample=sample)
            else:
                frames = read_frames_decord(fn, num_frames=max_num_frames, min_num_frames=min_num_frames,
                                            client=self.client, sample=sample, use_time=use_time, clip=clip)
            return frames


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def simulate_jpeg_degradation(quality):
    def jpeg_degrade(img):
        with io.BytesIO() as output:
            img.convert('RGB').save(output, format='JPEG', quality=quality)
            output.seek(0)  # Move the reading cursor to the start of the stream
            img_jpeg = Image.open(output).copy()  # Use .copy() to make sure the image is loaded in memory
        return img_jpeg
    return jpeg_degrade


# Define the JPEG compression quality range, pre-create all JPEG compression functions
qualities = list(range(75, 101))
jpeg_degrade_functions = {quality: simulate_jpeg_degradation(quality) for quality in qualities}


def build_transform(is_train, input_size, pad2square=False, normalize_type='imagenet'):
    if normalize_type == 'imagenet':
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == 'clip':
        MEAN, STD = CLIP_MEAN, CLIP_STD
    elif normalize_type == 'siglip':
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
    elif normalize_type == 'hunyuan':
        MEAN, STD = HUNYUAN_MEAN, HUNYUAN_STD
    else:
        raise NotImplementedError
    if is_train:  # use data augumentation
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomChoice([T.Lambda(jpeg_degrade_functions[quality]) for quality in qualities]),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        if pad2square is False:  # now we use this transform function by default
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
        else:
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Lambda(lambda img: expand2square(img, tuple(int(x * 255) for x in MEAN))),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])

    return transform

def generate_tokens(w=448, h=448, use_xrope=True):
    total_patch_size = 16 * 2 * 2
    tokens = ''
    tokens += IMG_START_TOKEN
    tokens += IMG_CONTEXT_TOKEN
    for i in range(h // total_patch_size):
        for j in range(w // total_patch_size):
            tokens += IMG_CONTEXT_TOKEN
        if use_xrope:
            tokens += LINE_TOKEN
        else:
            tokens += IMG_CONTEXT_TOKEN
    tokens += IMG_CONTEXT_TOKEN
    tokens += IMG_END_TOKEN
    return tokens

def get_xdrope_position_ids(
        position_ids_t,
        position_ids_x,
        position_ids_y,
        b,
        i,
        prev_index,
        boi_index,
        eoi_index,
        eol_index,
    ):

    position_ids_t[b, (i + 1):] -= (i + 1 - prev_index)
    position_ids_x[b, (i + 1):] -= (i + 1 - prev_index)
    position_ids_y[b, (i + 1):] -= (i + 1 - prev_index)

    idx_cur = 0
    for x in range(boi_index.size()[0]):
        m = boi_index[x]
        n = eoi_index[x]
        assert m < n
        # Reset image token position ids.
        if m >= prev_index and m < i:
            assert n < i
            position_ids_t[b, m+1+1:n-1] = idx_cur
            idx_cur += 1

            eol_idx_list = []
            for y in range(eol_index.size()[0]):
                eol_idx = eol_index[y]
                # Reset image token position ids.
                if eol_idx > m and eol_idx < n:
                    eol_idx_list.append(eol_idx)
            row = len(eol_idx_list)
            assert row > 0, 'the row of an image must be a positive integer'
            # -2 is for learnable img start and img end for each image
            # -1 is for getting rid of endpoint
            column = torch.round((n-m-2-1)/row).long().item()

            assert row * column == n-m-2-1, f"row:\t{row}, column:\t{column}, n:\t{n}, m:\t{m}, {int((n-m-2-1)/row)}"

            idx_xy = 0
            for rr in range(row):
                for cc in range(column):
                    position_ids_x[b, m+1+1+idx_xy] = cc
                    position_ids_y[b, m+1+1+idx_xy] = rr
                    idx_xy += 1

    return position_ids_t, position_ids_x, position_ids_y


def get_attention_masks_and_position_ids(data, eod_id, im_start_id, im_end_id, im_newline_id):
    position_embedding_xdrope = True

    micro_batch_size, seq_length = data.size()
    att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length))).view(
        att_mask_batch, 1, seq_length, seq_length).int()
    position_ids = torch.arange(seq_length, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    if position_embedding_xdrope:
        position_ids_t = position_ids.clone()
        position_ids_x = position_ids.clone()
        position_ids_y = position_ids.clone()
    for b in range(micro_batch_size):
        # Find indecies where EOD token is.
        eod_index = position_ids[b, data[b] == eod_id]

        eod_index = eod_index.clone()
        # Detach indecies from positions if going to modify positions.
        # Loop through EOD indecies:
        prev_index = 0
        if position_embedding_xdrope:
            boi_index = position_ids[b, data[b] == im_start_id]
            eoi_index = position_ids[b, data[b] == im_end_id]
            eol_index = position_ids[b, data[b] == im_newline_id]

        #print(boi_index, eoi_index, eol_index)
        for j in range(eod_index.size()[0]):
            i = eod_index[j]
            attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
            position_ids[b, (i + 1):] -= (i + 1 - prev_index)

            if position_embedding_xdrope:
                position_ids_t, position_ids_x, position_ids_y = get_xdrope_position_ids(position_ids_t, position_ids_x, position_ids_y,
                                                                                            b, i, prev_index, boi_index,
                                                                                            eoi_index, eol_index
                                                                                            )
            prev_index = i + 1
    if position_embedding_xdrope:
        position_ids = torch.cat([position_ids.unsqueeze(1), position_ids_x.unsqueeze(1), position_ids_y.unsqueeze(1),position_ids_t.unsqueeze(1)], dim=1)

    return attention_mask, position_ids


def preprocess_hunyuan(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        text_only: bool = False,
        image_size: int = 448,
        num_image: int = 1
) -> Dict:
    assert len(sources) == 1, 'process only the first conversations'
    conversations = sources[0]

    if not text_only:
        new_conversations = []
        current_image_idx = 0
        img_tokens_per_frame = generate_tokens(w=image_size, h=image_size)
        for conversation in conversations:
            if conversation['from'] == 'human':
                image_cnt = conversation['value'].count('<image>')
                for i in range(image_cnt):
                    if current_image_idx == num_image:
                        break
                    if current_image_idx == 0:
                        if num_image != 1:
                            image_tokens = f'{VID_START_TOKEN}{img_tokens_per_frame}'
                        else:
                            image_tokens = f'{VID_START_TOKEN}{img_tokens_per_frame}{VID_END_TOKEN}'
                    elif current_image_idx == num_image - 1:
                        image_tokens = f'{img_tokens_per_frame}{VID_END_TOKEN}'
                    else:
                        image_tokens = f'{img_tokens_per_frame}'
                    conversation['value'] = conversation['value'].replace('<image>', image_tokens, 1)
                    current_image_idx += 1
            new_conversations.append(conversation)
        conversations = new_conversations
        assert current_image_idx == num_image, f'{current_image_idx} != {num_image}'

    batches, roles = [], []

    for conversation in conversations:
        if conversation['from'] == 'human':
            batches.append(conversation["value"] + '<sep>')
            roles.append('human')
        elif conversation['from'] == 'gpt':
            batches.append(f'{conversation["value"]}<|endoftext|>')
            roles.append('gpt')
        else:
            raise NotImplementedError

    final_input_ids, final_targets = [tokenizer.bos_id], [IGNORE_TOKEN_ID]
    for role, batch in zip(roles, batches):

        input_ids = tokenizer.encode(
            batch,
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        input_ids = np.array(input_ids)

        final_input_ids.extend(input_ids.tolist())

        if role == 'system' or role == 'human':
            final_targets.extend(np.full(input_ids.shape, IGNORE_TOKEN_ID).tolist())  # ignore
        elif role == 'gpt':
            target = input_ids.copy()
            final_targets.extend(target.tolist())
        else:
            raise NotImplementedError

    final_input_ids = np.array(final_input_ids)
    final_targets = np.array(final_targets)
    input_ids = torch.tensor(final_input_ids)[:tokenizer.model_max_length]
    targets = torch.tensor(final_targets)[:tokenizer.model_max_length]

    _, position_ids = get_attention_masks_and_position_ids(input_ids.unsqueeze(0), \
        tokenizer.eod_id, tokenizer.im_start_id, tokenizer.im_end_id, tokenizer.im_newline_id)

    input_ids[input_ids == tokenizer.im_newline_id] = tokenizer.image_token_id

    torch.set_printoptions(threshold=float('inf'))

    padding = False
    if padding:
        current_length = input_ids.size(0)
        padding_length = tokenizer.model_max_length - current_length
        input_ids = F.pad(input_ids, (0, padding_length), value=tokenizer.pad_id)
        targets = F.pad(targets, (0, padding_length), value=IGNORE_TOKEN_ID)

    input_ids = input_ids.unsqueeze(0)
    targets = targets.unsqueeze(0)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_id),
        position_ids=position_ids,
    )
