import torch
import os
import math
import time

# import sys
# transformers_path = "/apdcephfs_gy4/share_303218624/yuyingge/ARC_Qwen_Video_7B"
# sys.path.insert(0, transformers_path)

import transformers
from transformers import (
    ARC_Qwen2_5_VL_VideoForConditionalGeneration,
    AutoProcessor,
    WhisperFeatureExtractor
)
os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"
from vision_process import process_vision_info
from vision_utils import load_audio_from_video

model_path = 'TencentARC/ARC-Qwen-Video-7B'
whisper_path = 'openai/whisper-large-v3'

device = "cuda:0"

print("Loading model and processors...")
load_start_time = time.time()

model = ARC_Qwen2_5_VL_VideoForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=device,
).eval()

processor = AutoProcessor.from_pretrained(model_path)
wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)

print(f"Model and processors loaded in {time.time() - load_start_time:.2f} seconds.")


def build_prompt(title, task_type):
    if task_type == "MCQ":
        return f"{title}\nOutput the thinking process in <think> </think> and final answer (only option index) in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>."
    elif task_type == "Grounding":
        return f"{title}\nOutput the thinking process in <think> </think> and final answer (only time range) in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>."
    else:
        return f"{title}\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>."

def run_inference(video_path, title="", task_type="QA"):

    print(f"🤖 Thinking...")

    try:
        start_time = time.time()

        prompt = build_prompt(title, task_type)

        print("Processing audio...")
        audios, duration = load_audio_from_video(video_path)
        sr = 16000
        max_num_segments = 10
        segment_length = sr * 30
        num_segments = math.ceil(len(audios) / segment_length)

        assert num_segments <= max_num_segments

        all_spectrograms = []
        for i in range(num_segments):
            start = i * segment_length
            end = min((i + 1) * segment_length, len(audios))
            segment = audios[start:end]
            if len(segment) > 0:
                spectrogram = wav_processor(segment, sampling_rate=sr, return_tensors="pt")["input_features"]
                all_spectrograms.append(spectrogram)

        values_audios = torch.cat(all_spectrograms, dim=1).squeeze() if all_spectrograms else None
        print("Audio processing done.")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        print(f"{task_type} task selected. Processing video with process_vision_info.")
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        if video_inputs and len(video_inputs) > 0:
            print(f"Video input shape: {video_inputs[0].shape}, kwargs: {video_kwargs}, Audio shape: {audios.shape if audios is not None else 'None'}")
        else:
             print(f"No video inputs. kwargs: {video_kwargs}, Audio shape: {audios.shape if audios is not None else 'None'}")

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )

        if values_audios is not None:
            inputs['values_audios'] = values_audios.to(model.device, dtype=torch.bfloat16)

        inputs = inputs.to(model.device)

        print("Generating response...")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        full_response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        end_time = time.time()
        print(f"Model inference took: {end_time - start_time:.2f} seconds.")

        return full_response

    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = f"error：{str(e)}"
        return error_msg


if __name__ == "__main__":
    example_base_path = "examples"
        
    examples = [
        ["寿司.mp4", "该视频标题为白金枪鱼寿司的陷阱\n描述视频内容.", "Summary"],
        ["寿司.mp4", "我们何时能看到一个穿制服的男人站在菊花门前?", "Grounding"],
        ["寿司.mp4", "这个视频有哪些幽默的地方？", "QA"],
        ["寿司.mp4", "视频中最后老板提供了什么给顾客？\nA.纸尿裤\nB.寿司\nC.现金\nD.面巾纸", "MCQ"],
        ["开关.mov", "请按时间顺序给出视频的章节摘要和对应时间点", "Segment"],
        ["猪排.mp4", "When will we be able to see the man in the video eat the pork cutlet in the restaurant?", "Grounding"],
        ["猪排.mp4", "Is the man satisfied with the pork cutlet he cooked at the beginning of the video?", "QA"],
        ["猪排.mp4", "Localize video chapters with temporal boundaries and the corresponding sentence description.", "Segment"],
    ]

    start_time = time.time()
    for video_file, prompt, task in examples:
        full_video_path = os.path.join(example_base_path, video_file)
        
        print("="*80)
        print(f"   - Video: {full_video_path}")
        print(f"   - Task: {task}")
        print(f"   - Prompt: {prompt if prompt else '无'}")
        print("="*80)
        
        response = run_inference(video_path=full_video_path, title=prompt, task_type=task)
        
        print("\n" + "-"*30 + " Model output " + "-"*30)
        print(response)
        print("-" * (62 + len("Model output")))
        print("\n\n")

    end_time = time.time()
    print(f"Total inference time: {end_time - start_time:.4f} seconds")
