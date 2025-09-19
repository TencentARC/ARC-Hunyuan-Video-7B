# ARC-Hunyuan-Video-7B

[![arXiv](https://img.shields.io/badge/arXiv-2507.20939-b31b1b.svg)](https://arxiv.org/abs/2507.20939)
[![Demo](https://img.shields.io/badge/ARC-Demo-blue)](https://arc.tencent.com/en/ai-demos/multimodal)
[![Static Badge](https://img.shields.io/badge/Model-Huggingface-yellow)](https://huggingface.co/TencentARC/ARC-Hunyuan-Video-7B)
[![Static Badge](https://img.shields.io/badge/Model-Huggingface-yellow)](https://huggingface.co/TencentARC/ARC-Qwen-Video-7B)
[![Static Badge](https://img.shields.io/badge/Model-Huggingface-yellow)](https://huggingface.co/TencentARC/ARC-Qwen-Video-7B-Narrator)
[![Blog](https://img.shields.io/badge/ARC-Blog-green)](https://tencentarc.github.io/posts/arc-video-announcement/)
[![Benchmark](https://img.shields.io/badge/ShortVid-Bench-orange)](https://huggingface.co/datasets/TencentARC/ShortVid-Bench)

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

## ARC-Qwen-Video-7B
In this version, we have switched the base model from hunyuan VLM to [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) and introduce [ARC-Qwen-Video-7B](https://huggingface.co/TencentARC/ARC-Qwen-Video-7B). We used the same training data and training stages. Please refere to the  `arc-qwen-video` branch for details.

We are also introducing a new model, [ARC-Qwen-Video-7B-Narrator](https://huggingface.co/TencentARC/ARC-Qwen-Video-7B-Narrator). It can output **timestamped video descriptions, speaker identities, and the specific ASR (Automatic Speech Recognition) content**. By processing its output with an external LLM, you can obtain more comprehensive structured information as follows (Click to watch the video):

[<img src="https://img.youtube.com/vi/Bz1T4wCuWc8/maxresdefault.jpg" alt="视频" width="300">](https://www.youtube.com/watch?v=Bz1T4wCuWc8)

> ### 视频概述
>
> 这是一个喜剧短片，讲述了一位丈夫藏在棉衣里的私房钱被妻子意外发现，并误以为是丈夫准备的“惊喜”礼物。视频通过夫妻二人的一通电话，生动展现了丈夫从悠闲自得，到震惊错愕，再到崩溃无奈的全过程，充满了戏剧性的反转和幽默感。
>
> ### 情节发展分解
>
> 视频情节围绕一通电话展开，以下是详细的时间线、场景、说话人和对话内容：
>
> <table>
>   <thead>
>     <tr>
>       <th>时间戳</th>
>       <th>场景描述</th>
>       <th>说话人</th>
>       <th>对话内容 (ASR)</th>
>     </tr>
>   </thead>
>   <tbody>
>     <tr>
>       <td>0:00 - 0:05</td>
>       <td>丈夫头戴浴帽，围着浴巾，在室内泳池边悠闲地自拍。</td>
>       <td>无</td>
>       <td>(无对话)</td>
>     </tr>
>     <tr>
>       <td>0:05 - 0:10</td>
>       <td><b>镜头切换</b>：妻子在服装店里，满脸幸福地给丈夫打电话。</td>
>       <td>妻子</td>
>       <td>“哎，老公，老公，我爱你爱你，爱死你了，么么么。”</td>
>     </tr>
>     <tr>
>       <td rowspan="2" style="vertical-align: top;">0:10 - 0:18</td>
>       <td rowspan="2" style="vertical-align: top;">丈夫接起电话，对妻子的热情感到好奇，妻子则兴奋地揭晓了“惊喜”。</td>
>       <td>丈夫</td>
>       <td>“哎，怎么了你这是，这么高兴啊？”</td>
>     </tr>
>     <tr>
>       <td>妻子</td>
>       <td>“今天我在我的棉衣兜里，发现了你给我的惊喜，一万元哟。”</td>
>     </tr>
>     <tr>
>       <td>0:18 - 0:27</td>
>       <td>听到“一万元”，丈夫表情瞬间凝固，从疑惑变为震惊和懊悔，但仍强装镇定。</td>
>       <td>丈夫</td>
>       <td>“啊？好啊，你你你你开心高兴就行。”</td>
>     </tr>
>     <tr>
>       <td>0:27 - 0:34</td>
>       <td>妻子开心地告知钱的用途，丈夫的表情彻底僵住，震惊加剧。</td>
>       <td>妻子</td>
>       <td>“我当然高兴啊，我用它买了一件新衣裳，等晚上回去穿给你看啊。”</td>
>     </tr>
>     <tr>
>       <td rowspan="3" style="vertical-align: top;">0:34 - 0:46</td>
>       <td rowspan="3" style="vertical-align: top;">丈夫确认钱已被花掉，情绪崩溃。妻子则认为是丈夫授权的，丈夫忍不住骂了一句。</td>
>       <td>丈夫</td>
>       <td>“你已经给买成衣服了？”</td>
>     </tr>
>     <tr>
>       <td>妻子</td>
>       <td>“当然啦，不是你说的吗？说买我自己喜欢的东西。老公，你真是太好了。”</td>
>     </tr>
>     <tr>
>       <td>丈夫</td>
>       <td>“你真是败家娘们儿啊你。”</td>
>     </tr>
>     <tr>
>       <td rowspan="4" style="vertical-align: top;">0:46 - 0:59</td>
>       <td rowspan="4" style="vertical-align: top;">妻子察觉丈夫语气不对，丈夫立刻改口掩饰，并催促妻子早点回家。</td>
>       <td>妻子</td>
>       <td>“什么，老公，你说什么？”</td>
>     </tr>
>     <tr>
>       <td>丈夫</td>
>       <td>“啊？我说好啊，你漂亮我高兴。”</td>
>     </tr>
>     <tr>
>       <td>妻子</td>
>       <td>“你说的，老公。你今天呀，一定要早点回来哟，我等你哟。”</td>
>     </tr>
>     <tr>
>       <td>丈夫</td>
>       <td>“行行行行行。”</td>
>     </tr>
>   </tbody>
> </table>
>
> ### 人物与核心冲突
>
> #### 1. 人物分析
>
>    丈夫:
>        行为: 藏私房钱，事发后极力掩饰自己的真实情绪（心痛、懊悔）。
>        心理变化: 悠闲 -> 疑惑 -> 震惊 -> 崩溃 -> 无奈接受。
>        特点: 爱面子，对妻子既有爱意也有无奈，典型的“妻管严”形象。
>
>    妻子:
>        行为: 发现钱后，认为是丈夫的爱意表达，并迅速将其消费。
>        心理变化: 全程处于发现“惊喜”的幸福和喜悦中。
>        特点: 天真、消费果断，对丈夫充满信任和爱意。
>
> #### 2. 核心冲突
>
> 视频的核心冲突在于 “信息的严重不对等” 所造成的戏剧性误会：
>
> *   丈夫视角: 辛苦攒下的 $10,000$ 元私房钱被意外发现并花掉，是一场“惊吓”。
> *   妻子视角: 丈夫精心准备的 $10,000$ 元浪漫基金，是一份巨大的“惊喜”。
>
> 这个误会推动了整个故事的发展，丈夫的“打碎牙往肚里咽”和妻子的“理所当然的幸福”形成了强烈的喜剧反差，制造了密集的笑点。
>
> ### 总结
>
> 该视频通过一个关于“私房钱”的常见家庭情景，巧妙地构建了一个充满反转和幽默的故事。它利用戏剧性讽刺（观众和丈夫知道真相，而妻子蒙在鼓里）的手法，精准捕捉了丈夫在突发状况下的复杂心理活动。整个过程不仅笑料百出，也含蓄地探讨了夫妻间的沟通、信任和金钱观等话题，容易引发观众的共鸣和讨论。


## News
- 2025.09.19: We release [ARC-Qwen-Video-7B](https://huggingface.co/TencentARC/ARC-Qwen-Video-7B), which switched the base model from hunyuan VLM to [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct). We also release [ARC-Qwen-Video-7B-Narrator](https://huggingface.co/TencentARC/ARC-Qwen-Video-7B-Narrator), which can output timestamped video descriptions, speaker identities, and the specific ASR (Automatic Speech Recognition) content. Please refere to the  `arc-qwen-video` branch for details.
- 2025.08.05: We release [ShortVid-Bench](https://huggingface.co/datasets/TencentARC/ShortVid-Bench), a specialized, human-annotated benchmark with multiple-choice questions for evaluating short-video understanding.
- 2025.07.29: We release the training code for instruction tuning.
- 2025.07.25: We release the [model checkpoint](https://huggingface.co/TencentARC/ARC-Hunyuan-Video-7B) and inference code of ARC-Hunyuan-Video-7B including [vLLM](https://github.com/vllm-project/vllm) version.
- 2025.07.25: We release the [API service](https://arc.tencent.com/zh/document/ARC-Hunyuan-Video-7B) of ARC-Hunyuan-Video-7B, which is supported by [vLLM](https://github.com/vllm-project/vllm). We release two versions: one is V0, which only supports video description and summarization in Chinese; the other is the version consistent with the model checkpoint and the one described in the paper.

## TODOs

- [x] Relase ShortVid-Bench, a specialized, human-annotated benchmark with multiple-choice questions
- [x] Release training code for instruction tuning

## Usage

### Dependencies
Our inference can be performed on a single NVIDIA A100 40GB GPU.

### Installation

Clone the repo and install dependent packages

```bash
git clone https://github.com/TencentARC/ARC-Hunyuan-Video-7B.git
cd ARC-Hunyuan-Video-7B
# Install torch 2.6.0 based on your CUDA version
# CUDA 11.8
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# CUDA 12.6
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

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

## Training

### Installation

Clone the repo and install dependent packages

```bash
git clone https://github.com/TencentARC/ARC-Hunyuan-Video-7B.git
cd ARC-Hunyuan-Video-7B
# Install torch 2.6.0
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install git+https://github.com/liyz15/transformers.git@arc_hunyuan_video

# For training
pip install accelerate==1.9.0
# Upgrade the GCC version to 9.0 or above
sudo dnf install gcc-toolset-9
scl enable gcc-toolset-9 bash
source /opt/rh/gcc-toolset-9/enable
gcc -v
```

### Model Weights

- Download [ARC-Hunyuan-Video-7B](https://huggingface.co/TencentARC/ARC-Hunyuan-Video-7B) including ViT and LLM and the original [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) .

### Data Preparation

Please follow the format of "sft_data/sft_jb_sp_kd_10.json".

- "root" specifies the path of training videos (supports .mp4; videos shorter than 5 minutes yield better results).
- "audio_root" specifies the path of corresponding audios (Please use the .mp3 format). You can use the code below to extract audio from a video and save it.

```bash
from moviepy.editor import VideoFileClip
from pydub import AudioSegment

video = VideoFileClip(video_path)
if video.audio is not None:
    video.audio.write_audiofile(audio_path, logger=None)
    video.audio.close()
else:
    duration_ms = int(video.duration * 1000)
    silent_audio = AudioSegment.silent(duration=duration_ms)
    silent_audio.export(audio_path, format="mp3")
video.close()
```

- "annotation" specifies the path of the annotation in the format of ".jsonl".

### Model Fully-finetune

```bash
# We use DeepSpeed Zero-3 with two 98G-H20 GPUs.
bash scripts/arc_hunyuan_video_full_finetune.sh
```

### Model Inference

After finishing training, the model will be saved in ${OUTPUT_DIR}.

```bash
# Copy the model-related config files to the directory.
cd path of the downloaded ARC-Hunyuan-Video-7B
cp generation_config.json preprocessor_config.json ${OUTPUT_DIR}/checkpoint-500/.

cd ARC-Hunyuan-Video-7B
# Modify the prompt based on your fine-tuning data, and specify the path of the fine-tuned model.
python3 video_inference_sft.py
```

## API service

We also provide access to the model via API, which is supported by [vLLM](https://github.com/vllm-project/vllm). For details, please refer to the [documentation](https://arc.tencent.com/zh/document/ARC-Hunyuan-Video-7B).

We release two versions: one is V0, which only supports video description and summarization in Chinese; the other is the version consistent with the model checkpoint and the one described in the paper, which is capable of multi-granularity timestamped video captioning and summarization, open-ended video question answering, temporal video grounding, and video reasoning (It supports Chinese and English videos and particularly excels at Chinese).
For videos longer than 5 minutes, we only support structured descriptions. We process these videos in 5-minute segments and use an LLM to integrate the inference results.

If you only need to understand and summarize short Chinese videos, we recommend using the V0 version.

Due to video file size limitations imposed by the deployment API, we compressed input video resolutions for our online demo and API services. Consequently, model performance in these interfaces may slightly deviate from the results reported in the paper. To reproduce the original performance, we recommend local inference.


## ShortVid-Bench

Existing benchmarks often fall short in capturing the nuanced complexities
of user-generated content. To rigorously evaluate model’s ability to **understand real-world short videos**,
we construct a specialized benchmark named **ShortVid-Bench**. Specifically, we develop an automated pipeline
to generate multi-dimensional questions for each video, targeting capabilities that signify a deep, holistic
comprehension through integrating both visual and audio cues. These dimensions include: 
- Temporal Reasoning and Localization
- Affective Intent Classification
- Creator Intent Taxonomy
- Narrative Comprehension
- Humor & Meme Deconstruction
- Creative Innovation Analysis
  
For objective assessment, we employ a multiple-choice question (MCQ) format following previous work. Each question is carefully curated by human annotators who
provide the ground-truth answer and design challenging, plausible distractors. Collectively, these dimensions with a total of 1,000 multiple-choice questions
push the evaluation beyond mere descriptive captioning, demanding a genuine comprehension of the video’s
context, intent, and narrative.

<p align="center">
    <img src="https://github.com/TencentARC/ARC-Hunyuan-Video-7B/blob/master/figures/shortvid-bench.jpg?raw=true" width="70%"/>
<p>

### Model Performance
| Model | fps | #frames | think | ShortVid-Bench | 
| :--- | :--- | :--- | :--- | :--- | 
| Qwen2.5-VL-7B-Instruct | 1.0 | 150 | × | 69.3 |
| Qwen2.5-Omni-7B | 1.0 | 150 | × | 69.7 |
| Keye-VL-8B | 1.0 | 150 | ✓ | 56.3 |
| ARC-Hunyuan-Video-7B | 1.0 | 150 | ✓ | **73.0** | 

<sub>
Please note that the results in the table above are different from those in 
<a href="https://arxiv.org/abs/2507.20939" target="_blank">ARC-Hunyuan-Video-7B</a>. 
This is because, after releasing the technical report, we expanded the benchmark dataset to 1,000 samples, whereas the results in the paper were based on 400 samples.
</sub>

## Future Work

We observe that incorporating generic video datasets during training may inadvertently compromise the model's capacity for real-world video understanding, potentially due to domain shift or noise introduced by non-real-world samples. To address this limitation, we plan to develop a dedicated model trained exclusively on rigorously curated real-world video data.

## Citation

If you find the work helpful, please consider citing:

```bash
@article{ge2025arc,
  title={ARC-Hunyuan-Video-7B: Structured Video Comprehension of Real-World Shorts},
  author={Ge, Yuying and Ge, Yixiao and Li, Chen and Wang, Teng and Pu, Junfu and Li, Yizhuo and Qiu, Lu and Ma, Jin and Duan, Lisheng and Zuo, Xinyu and others},
  journal={arXiv preprint arXiv:2507.20939},
  year={2025}
}
```

## Acknowledge

Our training code is built upon [InternVL](https://github.com/OpenGVLab/InternVL). Thanks for their excellent work!
