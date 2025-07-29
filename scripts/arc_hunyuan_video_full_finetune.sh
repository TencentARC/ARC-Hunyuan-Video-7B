set -x

export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECKS_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_DEBUG=info

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=8005
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='work_dirs/brief_summary_sft'
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

NODE_RANK=$1
torchrun --nproc_per_node=2 \
  model_train/train/arc_hunyuan_video_finetune.py \
  --model_name_or_path "TencentARC/ARC-Hunyuan-Video-7B" \
  --conv_style "hunyuan" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "sft_data/sft_jb_sp_kd_10.json" \
  --overwrite_output_dir True \
  --force_image_size 640 \
  --num_image_token 112 \
  --freeze_llm False \
  --freeze_speech_encoder True \
  --freeze_backbone True \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 500 \
  --save_total_limit 100 \
  --learning_rate 1e-5 \
  --warmup_steps 100 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 20000 \
  --max_num_frame 150 \
  --do_train True \
  --grad_checkpoint True \
  --dynamic_image_size False \
  --normalize_type hunyuan \
  --seed 42 \
  --deepspeed "config/zero_stage3_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
