# This create a class to use the model in vllm with mm encoder + llm
# The inputs should be preprocessed
# The mm encoder will process input sequentially and llm will do batch inference
# This may be less efficient, but much more clear and easy to use

import os
import json
from pathlib import Path
import tempfile
import shutil

import torch

from transformers import AutoTokenizer, AutoConfig, PretrainedConfig
from vllm import LLM, SamplingParams
from safetensors.torch import load_file as safetensors_load_file

from model_vllm import VideoAudioEncoder


def convert_config_to_legacy(config):
    legacy_config = PretrainedConfig()

    legacy_config.update(config.vision_config.to_dict())
    legacy_config.update(config.text_config.to_dict())

    force_image_size = config.vision_config.force_image_size
    num_image_token = int(
        (force_image_size / 64)
        * (force_image_size / 64 + 1)
        + 2
    )

    legacy_config.update({
        # Such that vllm can caculate the max_model_len correctly
        "architectures": ["HunyuanVideoModel"],
        "image_token_id": config.text_config.image_token_id,
        "vision_start_token_id": config.text_config.im_start_id,
        "num_image_token": num_image_token,
        "rope_scaling": {
            "alpha": 1000.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 1000.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "rope_type": "dynamic",
            "mrope_section": [0.25, 0.25, 0.25, 0.25],
        },
    })

    if hasattr(legacy_config, "torch_dtype") and isinstance(legacy_config.torch_dtype, str):
        # Convert string torch_dtype to torch.dtype
        legacy_config.torch_dtype = getattr(torch, legacy_config.torch_dtype)

    return legacy_config


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


class VideoAudioLLM:
    def __init__(
        self,
        model_path,
        device_enc="cuda",
        device_llm="cuda",
        **kwargs,
    ):

        self.config = AutoConfig.from_pretrained(model_path)
        self.device_enc = device_enc
        self.device_llm = device_llm

        self.mm_encoder = self.init_mm_encoder(model_path, self.config, self.device_enc)

        self.llm, self.sampling_params = self.init_llm(model_path, self.config, self.device_llm, **kwargs)


    def init_mm_encoder(self, model_path, config, device):
        multi_modal_state_dict = load_state_dict_from_safetensors(
            model_path, ("vision_model.", "mlp2.", "speech_encoder.")
        )

        multi_modal_encoder = VideoAudioEncoder(
            config,
            max_num_frames=config.max_num_frame,
        )

        missing, unexpected = multi_modal_encoder.load_state_dict(
            multi_modal_state_dict, strict=False
        )
        assert len(missing) == 0, f"Missing keys in mm encoder: {missing}"
        assert (
            len(unexpected) == 0
        ), f"Unexpected keys in mm encoder: {unexpected}"

        multi_modal_encoder.eval()
        multi_modal_encoder.to(device)

        return multi_modal_encoder

    def init_llm(self, model_path, config, device, **kwargs):

        if self.device_enc != self.device_llm:
            gpu_memory_utilization = 0.9
        else:  # Reserve memory for the encoder
            gpu_memory_utilization = 0.6

        llm = LLM(
            model=model_path,
            tokenizer=model_path,
            trust_remote_code=True,
            max_model_len=20480,
            max_seq_len_to_capture=20480,
            dtype="bfloat16",
            hf_overrides=convert_config_to_legacy,
            limit_mm_per_prompt={"image": 150},
            enforce_eager=True,
            disable_mm_preprocessor_cache=True,
            enable_prefix_caching=False,
            device=device,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        sampling_params = SamplingParams(
            **kwargs,
        )

        return llm, sampling_params

    def forward_mm_encoder(self, batch):
        """
        This function will process the batch of data in the mm encoder
        Input:
            - batch: list of dicts, each dict contains the following keys:
                - pixel_values: torch.Tensor
                - audio_values: torch.Tensor
                - duration: float

        Output:
            - list of dicts, each dict contains the following keys:
                - embeddings: torch.Tensor
                - other keys in the original dict
        """
        device = self.device_enc
        ret = []

        for data in batch:
            pixel_values = data["pixel_values"]
            audio_values = data["audio_values"]
            duration = data["duration"]

            with torch.no_grad(), torch.autocast(device, torch.bfloat16):
                pixel_values = pixel_values.to(
                    device=device, dtype=torch.bfloat16, non_blocking=True
                )
                audio_values = audio_values.to(
                    device=device, dtype=torch.bfloat16, non_blocking=True
                )

                mixed_embeds = self.mm_encoder(
                    pixel_values, audio_values, duration
                )

                mixed_embeds = mixed_embeds.to(device="cpu").float().share_memory_()
                ret.append({"embeddings": mixed_embeds, **data})

        
        return ret

    def forward_llm(self, batch):
        num_patches = (
            self.config.vision_config.force_image_size // 32 // 2
        )
        image_grid_thw = torch.tensor([[1, num_patches, num_patches + 1]])
        prompts = [
            {
                "prompt": "<|startoftext|>" + item["text_prompt"],
                "multi_modal_data": {
                    "image": {
                        "image_embeds": item["embeddings"],
                        "image_grid_thw": image_grid_thw.repeat(
                            item["embeddings"].shape[0], 1
                        ),
                    }
                },
            }
            for item in batch
        ]

        print(prompts[0]['prompt'], prompts[0]['multi_modal_data']['image']['image_embeds'].shape)

        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)

        ret = []

        for data, output in zip(batch, outputs):
            if "output" in data:
                raise ValueError("Check the batch, there is a key called output")

            ret.append({"output": output.outputs[0].text, **data})

        return ret

    def __call__(self, batch):
        if isinstance(batch, dict):
            batch = [batch]

        ret = self.forward_mm_encoder(batch)
        ret = self.forward_llm(ret)
        return ret
