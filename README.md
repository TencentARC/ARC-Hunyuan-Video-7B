# ARC-Hunyuan-Video-7B

[![arXiv](https://img.shields.io/badge/arXiv-2507.20939-b31b1b.svg)](https://arxiv.org/abs/2507.20939)
[![Demo](https://img.shields.io/badge/ARC-Demo-blue)](https://arc.tencent.com/en/ai-demos/multimodal)
[![Static Badge](https://img.shields.io/badge/Model-Huggingface-yellow)](https://huggingface.co/TencentARC/ARC-Hunyuan-Video-7B)
[![Blog](https://img.shields.io/badge/ARC-Blog-green)](https://tencentarc.github.io/posts/arc-video-announcement/)

<sub>
Please note that in our Demo, ARC-Hunyuan-Video-7B is the model consistent with the model checkpoint and the one described in the paper, while ARC-Hunyuan-Video-7B-V0 only supports video description and summarization in Chinese.
Due to API file size limits, our demo uses compressed input video resolutions, which may cause slight performance differences from the paper. For original results, please run locally.
</sub>

## Introduction

We introduce **ARC-Hunyuan-Video-7B**, a powerful multimodal model designed for _understanding real-world short videos_.
Understanding user-generated videos is actually challenging due to their complex visual elements, high
information density in both visuals and audio, and fast pacing that focuses on emotional expression and viewpoint delivery.
To address this challenge, ARC-Hunyuan-Video-7B
processes visual, audio, and textual signals end-to-end for a deep, structured understanding of video through integrating and reasoning over multimodal cues.
Stress test reports show an inference time of just 10 seconds for a one-minute video on H20 GPU, yielding an average of 500 tokens, with
inference accelerated by the vLLM framework.

Compared to prior arts, we introduces a new paradigm of **Structured Video Comprehension**, with capabilities including:

- **Deep Understanding of Real-World Short Videos:** ARC-Hunyuan-Video-7B excels at analyzing user-generated content from platforms like WeChat Channels and TikTok. It goes beyond surface-level descriptions to grasp the creator's intent, emotional expression, and core message by processing complex visual elements, dense audio cues, and rapid pacing.
- **Synchronized Audio-Visual Reasoning:** The synchronization of raw visual and audio signals allows our model to answer complex questions that are impossible to solve with only one modality, such as understanding humor in a skit or details in a product review.
- **Precise Temporal Awareness:** ARC-Hunyuan-Video-7B knows not just _what_ happens, but _when_ it happens. It supports multi-granularity timestamped captioning, temporal video grounding, and detailed event summarization, making it perfect for applications like video search, highlight generation, and content analysis.
- **Advanced Reasoning and Application Versatility:** Leveraging a comprehensive multi-stage training regimen including Reinforcement Learning (RL), ARC-Hunyuan-Video-7B demonstrates strong reasoning capabilities. It supports zero-shot or few-shot fine-tuning for diverse downstream applications like video tagging, recommendation, and retrieval.

The model is capable of multi-granularity timestamped video captioning and summarization, open-ended video question answering, temporal video grounding, and
video reasoning as below,

<p align="center">
    <img src="https://github.com/TencentARC/ARC-Hunyuan-Video-7B/blob/master/figures/teaser.jpg?raw=true" width="90%"/>
<p>

Specifically, ARC-Hunyuan-Video-7B is built on top of the Hunyuan-7B vision-language model with the following key designs to meet the requirements of effective structured video comprehension:

- An extra audio encoder with fine-grained visual-audio synchronization for temporally aligned visual-audio inputs
- A timestamp overlay mechanism on visual frames that explicitly provides the model with temporal awareness
- Millions of real-world videos with a totally automated bootstrapped annotation pipeline
- A comprehensive training regimen based on the finding that grounding the model in objective
  tasks with RL is key to unlocking high-quality, subjective understanding

<p align="center">
    <img src="https://github.com/TencentARC/ARC-Hunyuan-Video-7B/blob/master/figures/method.jpg?raw=true" width="95%"/>
<p>

## News

- 2025.07.25: We release the [model checkpoint](https://huggingface.co/TencentARC/ARC-Hunyuan-Video-7B) and inference code of ARC-Hunyuan-Video-7B including [vLLM](https://github.com/vllm-project/vllm) version.
- 2025.07.25: We release the [API service](https://arc.tencent.com/zh/document/ARC-Hunyuan-Video-7B) of ARC-Hunyuan-Video-7B, which is supported by [vLLM](https://github.com/vllm-project/vllm). We release two versions: one is V0, which only supports video description and summarization in Chinese; the other is the version consistent with the model checkpoint and the one described in the paper.

## TODOs
- [ ] Relase ShortVid-Bench, a specialized, human-annotated benchmark with multiple-choice questions
- [ ] Release training code for instruction tuning


## Usage
### Dependencies
- Our inference can be performed on a single NVIDIA A100 40GB GPU.
- For the vLLM deployment version, we recommend using two NVIDIA A100 40GB GPUs.
### Installation

Clone the repo and install dependent packages

```bash
git clone https://github.com/TencentARC/ARC-Hunyuan-Video-7B.git
cd ARC-Hunyuan-Video-7B
# Install torch 2.6.0
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install git+https://github.com/liyz15/transformers.git@arc_hunyuan_video

# Install flash-attention based on your python version
# If you are unable to install flash-attention, you can modify attn_implementation to "sdpa" in video_inference.py
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl


# (Optional) For vllm, please follow the instructions below,
git submodule update --init --recursive
cd model_vllm/vllm/
export SETUPTOOLS_SCM_PRETEND_VERSION="0.8.5"
wget https://wheels.vllm.ai/ed2462030f2ccc84be13d8bb2c7476c84930fb71/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
export VLLM_PRECOMPILED_WHEEL_LOCATION=$(pwd)/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
pip install --editable .
# Install flash-attention if you haven't installed it
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

### Model Weights

- Download [ARC-Hunyuan-Video-7B](https://huggingface.co/TencentARC/ARC-Hunyuan-Video-7B) including ViT and LLM and the original [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) .

### Inference
```bash
# Our model currently excels at processing short videos of up to 5 minutes.
# If your video is longer, we recommend following the approach used in our demo and API:
# split the video into segments for inference, and then use an LLM to integrate the results.
```

#### Inference without vllm

```bash
cd ARC-Hunyuan-Video-7B
python3 video_inference.py
```

#### Inference with vllm

```bash
cd ARC-Hunyuan-Video-7B
python3 video_inference_vllm.py
```

## API service

We also provide access to the model via API, which is supported by [vLLM](https://github.com/vllm-project/vllm). For details, please refer to the [documentation](https://arc.tencent.com/zh/document/ARC-Hunyuan-Video-7B).

We release two versions: one is V0, which only supports video description and summarization in Chinese; the other is the version consistent with the model checkpoint and the one described in the paper, which is capable of multi-granularity timestamped video captioning and summarization, open-ended video question answering, temporal video grounding, and video reasoning (It supports Chinese and English videos and particularly excels at Chinese). 
For videos longer than 5 minutes, we only support structured descriptions. We process these videos in 5-minute segments and use an LLM to integrate the inference results.

If you only need to understand and summarize short Chinese videos, we recommend using the V0 version.

Due to video file size limitations imposed by the deployment API, we compressed input video resolutions for our online demo and API services. Consequently, model performance in these interfaces may slightly deviate from the results reported in the paper. To reproduce the original performance, we recommend local inference.

## Future Work

We observe that incorporating generic video datasets during training may inadvertently compromise the model's capacity for real-world video understanding, potentially due to domain shift or noise introduced by non-real-world samples. To address this limitation, we plan to develop a dedicated model trained exclusively on rigorously curated real-world video data.

## Citation

If you find the work helpful, please consider citing:

```bash
@article{ge2025seed,
  title={ARC-Hunyuan-Video-7B: Structured Video Comprehension of Real-World Shorts},
  author={Ge, Yuying and Ge, Yixiao and Li, Chen and Wang, Teng and Pu, Junfu and Li, Yizhuo and Qiu, Lu and Ma, Jin and Duan, Lisheng and Zuo, Xinyu and Luo, Jinwen and Gu, Weibo and Li, Zexuan and Zhang, Xiaojing and Tao, Yangyu and Hu, Han and Wang, Di and Shan Ying},
  journal={arXiv preprint arXiv:2507.20939},
  year={2025}
}
```
