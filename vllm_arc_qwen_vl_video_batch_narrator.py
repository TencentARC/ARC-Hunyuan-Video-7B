import sys
import math
import os
import json
import torch
import time
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as safetensors_load_file
from transformers import AutoProcessor, WhisperFeatureExtractor, AutoConfig
import vllm
from vllm import LLM, SamplingParams
from vision_process import process_vision_info
from vision_utils import load_audio_from_video
from video_audio_encoder import VideoAudioEncoder


def load_state_dict_from_safetensors(path: str, prefixes: list[str]):
    def filter_dict_with_k_prefix(d, prefixes):
        return {
            k: v
            for k, v in d.items()
            if any(k.startswith(prefix) for prefix in prefixes)
        }

    index_path = os.path.join(path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print(f"Index file {index_path} does not exist, loading all weights")
        pre_trained_dir = Path(path)
        weights_files = sorted(pre_trained_dir.glob("model-*.safetensors"))
    else:
        weight_map = json.load(open(index_path))["weight_map"]
        weights_files = set(
            filter_dict_with_k_prefix(weight_map, prefixes).values()
        )
        weights_files = [os.path.join(path, f) for f in weights_files]

    if len(weights_files) == 0:
        raise ValueError(
            f"No weights files found in {path} with prefixes {prefixes}"
        )

    state_dict = {}
    for file in weights_files:
        part_state_dict = safetensors_load_file(file)
        state_dict.update(part_state_dict)

    state_dict = filter_dict_with_k_prefix(state_dict, prefixes)
    return state_dict

def build_prompt(question, task_type):
    return f"{question}\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>."


class ProcessorConfig:
    """配置模型和推理所需的所有参数"""
    def __init__(self):
        self.model_path = 'TencentARC/ARC-Qwen-Video-7B-Narrator'
        self.whisper_path = 'openai/whisper-large-v3'
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1024,
        )
        self.llm_params = {
            "model": self.model_path,
            "limit_mm_per_prompt": {"video": 10},
        }

        if not os.path.isdir(self.model_path):
            self.model_path = snapshot_download(repo_id=self.model_path)


class VideoAudioProcessor:
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        print("Initializing models...")
        self._init_models()
        print("Models initialized successfully.")

    def _init_models(self):
        # Load AutoProcessor for chat template
        self.processor = AutoProcessor.from_pretrained(self.config.model_path)

        # Load WhisperFeatureExtractor for audio
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(self.config.whisper_path)

        # Load Multi-Modal Encoder
        model_config = AutoConfig.from_pretrained(self.config.model_path)
        self.multi_modal_encoder = VideoAudioEncoder(model_config)
        multi_modal_state_dict = load_state_dict_from_safetensors(
            self.config.model_path, ("visual.", "mlp_speech.", "speech_encoder.")
        )
        missing, unexpected = self.multi_modal_encoder.load_state_dict(
            multi_modal_state_dict, strict=False
        )
        assert len(missing) == 0, f"Missing keys in mm encoder: {missing}"
        assert len(unexpected) == 0, f"Unexpected keys in mm encoder: {unexpected}"
        self.multi_modal_encoder.eval()
        self.multi_modal_encoder.to(self.device)

        # Load LLM with vLLM
        self.llm = LLM(**self.config.llm_params)

    def _prepare_llm_inputs(self, video_path: str, question: str, task: str) -> dict:
        # 1. Build text prompt
        prompt_text = build_prompt(question, task)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        final_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 2. Process video frames
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = self.processor(
            text=[final_prompt], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt", **video_kwargs
        )
        pixel_values_videos = inputs['pixel_values_videos']
        video_grid_thw = inputs['video_grid_thw']
        second_per_grid_ts = inputs['second_per_grid_ts']

        # 3. Process audio
        audios, _ = load_audio_from_video(video_path)
        sr = 16000
        max_num_segments = 10
        segment_length = sr * 30
        num_segments = math.ceil(len(audios) / segment_length)
        assert num_segments <= max_num_segments, f"Video {video_path} has too many audio segments."

        all_spectrograms = []
        for i in range(num_segments):
            segment = audios[i * segment_length : (i + 1) * segment_length]
            if len(segment) > 0:
                spectrogram = self.wav_processor(segment, sampling_rate=sr, return_tensors="pt")["input_features"]
                all_spectrograms.append(spectrogram)
        
        values_audios = torch.cat(all_spectrograms, dim=1).squeeze(0) if all_spectrograms else None

        # 4. Generate multi-modal embeddings
        pixel_values_videos = pixel_values_videos.to(device=self.device, dtype=self.config.dtype)
        video_grid_thw = video_grid_thw.to(device=self.device)
        audio_values = values_audios.to(device=self.device, dtype=self.config.dtype) if values_audios is not None else None

        with torch.no_grad():
            mixed_embeds = self.multi_modal_encoder(
                self.device, pixel_values_videos, video_grid_thw, audio_values
            )

        # 5. Prepare final input dictionary for vLLM
        mm_data = {}
        if video_inputs is not None:
            mm_data["video"] = {
                "video_embeds": mixed_embeds.to(device="cpu").float().share_memory_(),
                "video_grid_thw": video_grid_thw.to(device="cpu").share_memory_(),
                "second_per_grid_ts": second_per_grid_ts
            }
        llm_inputs = {
            "prompt": final_prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }
        return llm_inputs

    def generate_response(self, video_path: str, question: str, task: str) -> str:
        llm_inputs = self._prepare_llm_inputs(video_path, question, task)
        outputs = self.llm.generate([llm_inputs], sampling_params=self.config.sampling_params)
        generated_text = outputs[0].outputs[0].text
        return generated_text

    def generate_response_batch(self, video_paths: list[str], questions: list[str], tasks: list[str]) -> list[str]:
        batch_llm_inputs = []
        print(f"Preparing batch of {len(video_paths)} items...")
        for video_path, question, task in zip(video_paths, questions, tasks):
            try:
                llm_inputs = self._prepare_llm_inputs(video_path, question, task)
                batch_llm_inputs.append(llm_inputs)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")

        if not batch_llm_inputs:
            return []

        print("Starting batch generation...")
        outputs = self.llm.generate(batch_llm_inputs, sampling_params=self.config.sampling_params)
        
        responses = [output.outputs[0].text for output in outputs]
        return responses


def get_sample_data():
    video_paths, questions, tasks = [], [], []

    video_path = 'examples/寿司.mp4'

    # Summary task
    video_paths.append(video_path)
    tasks.append('Summary')
    questions.append("介绍一下视频的主要信息，你的思考过程需要包含ASR的结果。")

    return video_paths, questions, tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_inference", action="store_true", help="Enable batch inference mode.")
    args = parser.parse_args()

    config = ProcessorConfig()
    processor = VideoAudioProcessor(config)

    video_paths, questions, tasks = get_sample_data()

    start_time = time.time()

    if not args.batch_inference:
        print("\n--- Running in Sequential (Single Case) Mode ---\n")
        for i, (video_path, question, task) in enumerate(zip(video_paths, questions, tasks)):
            print(f"Processing item {i+1}/{len(video_paths)}...")
            response = processor.generate_response(video_path, question, task)
            print(f"Q: {question}\nA: {response}\n" + "-"*20)
    else:
        print("\n--- Running in Batch Inference Mode ---\n")
        responses = processor.generate_response_batch(video_paths, questions, tasks)
        for question, response in zip(questions, responses):
            print(f"Q: {question}\nA: {response}\n" + "-"*20)

    end_time = time.time()
    print(f"Total inference time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()
