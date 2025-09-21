# ARC-Qwen-Video-7B

[![arXiv](https://img.shields.io/badge/arXiv-2507.20939-b31b1b.svg)](https://arxiv.org/abs/2507.20939)
[![Static Badge](https://img.shields.io/badge/Model-Huggingface-yellow)](https://huggingface.co/TencentARC/ARC-Qwen-Video-7B)
[![Static Badge](https://img.shields.io/badge/Model-Huggingface-yellow)](https://huggingface.co/TencentARC/ARC-Qwen-Video-7B-Narrator)
[![Blog](https://img.shields.io/badge/ARC-Blog-green)](https://tencentarc.github.io/posts/arc-video-announcement/)
[![Benchmark](https://img.shields.io/badge/ShortVid-Bench-orange)](https://huggingface.co/datasets/TencentARC/ShortVid-Bench)

In this version, we have switched the base model from hunyuan VLM to [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) and introduce [ARC-Qwen-Video-7B](https://huggingface.co/TencentARC/ARC-Qwen-Video-7B). We used the same training data and training stages. For a detailed introduction, please refer to the `master` branch. The main distinctions are listed as below,

| Feature                                                                                                                | `ARC-Hunyuan-Video-7B`                                                                                                                                                                                          | `ARC-Qwen-Video-7B`                                                                                                                                                                                          |
| ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Base VLM**                                                                                                           | Hunyuan-VL-7B-Pretrain                                                                                                                                                                                 | Qwen2.5-VL-7B-Instruct                                                                                                                                                                                 |
| **Frame Resolution** <br> <small>*Each model uses a fixed frame resolution to maintain audio-video synchronization.*</small> | Fixed at `640 x 640`                                                                                                                                                                                     | Fixed at `392 x 292`                                                                                                                                                                                    |
| **Frame Sampling**                                                                                                     | • **< 150s**: 1 FPS <br> • **> 150s**: Uniformly sample 150 frames.                                                                                                                                               | • **< 300s**: 1 FPS <br> • **> 300s**: Uniformly sample 300 frames.                                                                                                                                             |
| **Audio-Video Synchronization**                                                                                                | • **< 150s**: Sum tokens from 1s audio + 1s video frame. <br> • **150-300s**: Sum tokens from corresponding audio segment + video frame. <br> • **> 300s**: Split audio into 300 segments, use first 2s of each.         | • **< 300s**: Sum tokens from 1s audio + 1s video. <br> • **> 300s**: Split audio into 300 segments, use middle 1s of each.                                                                               |

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


## Usage

### Dependencies
The installation has been tested and verified on the following environments:
*   NVIDIA H20 with CUDA 12.4
*   NVIDIA A100 with CUDA 12.1

### Installation

Clone the repo and install dependent packages

```bash
git clone -b arc-qwen-video https://github.com/TencentARC/ARC-Hunyuan-Video-7B.git
cd ARC-Hunyuan-Video-7B

# Install torch 2.6.0 based on your CUDA version
# CUDA 11.8
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# CUDA 12.6
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

pip install librosa decord av accelerate
pip uninstall transformers
pip install git+https://github.com/geyuying/transformers.git@arc-qwen-video
pip install deepspeed==0.16.9
pip install flash_attn==2.7.1.post4

# Install FFmpeg according to your system, and ensure that the following command produces a normal version output:
ffmpeg -version

# (Optional) For vllm, please follow the instructions below,
pip uninstall vllm
pip install git+https://github.com/geyuying/vllm.git@arc-qwen-video
```

### Inference

```bash
# Our model currently excels at processing short videos of up to 5 minutes.
# If your video is longer, we recommend following the approach used in our demo and API:
# split the video into segments for inference, and then use an LLM to integrate the results.
```

To quickly verify that your environment is set up correctly and that video and audio information are being processed as expected, you can run the following test case with ARC-Qwen-Video-7B.

```bash
video_path = "examples/猪排.mp4"
task = "QA"
question = "What did the man say at the beginning of the video after measuring the thickness of the fried pork cutlet?"
```
Expected Result: If the model's output contains the phrase "So thin", it indicates that your installation is successful.

#### Inference without vllm

```bash
cd ARC-Hunyuan-Video-7B

# For ARC-Qwen-Video-7B
python3 inference_arc_qwen_video.py

# For ARC-Qwen-Video-7B-Narrator
python3 inference_arc_qwen_video_narrator.py
```

#### Inference with vllm

```bash
cd ARC-Hunyuan-Video-7B

# For ARC-Qwen-Video-7B
python3 vllm_arc_qwen_vl_video_batch.py --batch_inference

# For ARC-Qwen-Video-7B-Narrator
python3 vllm_arc_qwen_vl_video_batch_narrator.py --batch_inference
```

## Benchmark Performance
| | Video-MMMU | MMVU | Temp-Compass | Video-Holmes | Video-MME | VCR-Bench | MV-Bench | ShortVid-Bench | Charades-STA |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ARC-Hunyuan-Video-7B | 31.1 | 49.1 | 66.0 | 40.9 | 58.7 | 50.5 | **62.6** | **73.0** | **54.8** |
| ARC-Qwen-Video-7B | **41.3** | **55.5** | **68.7** | **51.1** | **61.0** | **52.3** | 60.8 | 72.6 | 52.8 |

Quantitative evaluation is performed on different benchmarks using accuracy as the evaluation metric, except for the grounding task on Charades-STA, which uses mIoU. For all benchmarks other than VideoMMMU and Charades-STA, we only evaluated the multiple-choice questions.

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
