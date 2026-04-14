#!/usr/bin/env bash
set -euo pipefail

# 用法:
# DATA_PATH=/path/to/time300b_data bash train_overfit_first_seq.sh
# 或直接编辑下面的默认 DATA_PATH。

source /root/miniconda3/bin/activate MOE

sanitize_ld_library_path() {
    local original_ld="${LD_LIBRARY_PATH:-}"
    local sanitized_ld=""
    local IFS=':'

    for p in $original_ld; do
        [ -z "$p" ] && continue
        case "$p" in
            *"/stubs"|*"/stubs/")
                ;;
            *)
                if [ -z "$sanitized_ld" ]; then
                    sanitized_ld="$p"
                else
                    sanitized_ld="${sanitized_ld}:$p"
                fi
                ;;
        esac
    done

    local preferred_cuda_paths=""
    for p in /usr/local/cuda/lib64 /usr/local/cuda/compat /usr/lib/x86_64-linux-gnu; do
        if [ -d "$p" ]; then
            if [ -z "$preferred_cuda_paths" ]; then
                preferred_cuda_paths="$p"
            else
                preferred_cuda_paths="${preferred_cuda_paths}:$p"
            fi
        fi
    done

    if [ -n "$sanitized_ld" ] && [ -n "$preferred_cuda_paths" ]; then
        export LD_LIBRARY_PATH="${preferred_cuda_paths}:${sanitized_ld}"
    elif [ -n "$preferred_cuda_paths" ]; then
        export LD_LIBRARY_PATH="$preferred_cuda_paths"
    else
        export LD_LIBRARY_PATH="$sanitized_ld"
    fi
}

sanitize_ld_library_path

cd /mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE

DATA_PATH="/mnt/shared-storage-gpfs2/speechllm-share/lishenyi/datasets/time300b/b52d0ca9de8da5202f73a5057681fca6b48906fb"
OUTPUT_PATH="logs_2/time300b_overfit_3"

python main.py \
  -d "$DATA_PATH" \
  --model_family timesfm_2p5 \
  --from_scratch \
  --output_path "$OUTPUT_PATH" \
  --max_length 512 \
  --stride 1000000000 \
  --max_train_sequences 50 \
  --enable_overfit_fixed_window \
  --overfit_hist_length 384 \
  --overfit_gt_length 128 \
  --global_batch_size 16 \
  --micro_batch_size 16 \
    --train_steps 5000 \
    --learning_rate 1e-3 \
    --lr_scheduler_type constant \
    --min_learning_rate 1e-3 \
  --warmup_ratio 0.0 \
  --weight_decay 0.0 \
  --normalization_method none \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps 100 \
  --no-enable_validation_split \
  --no-load_best_model_at_end \
  --dataloader_num_workers 1 \
  --logging_steps 10 \
  --precision fp32 \
  --ddp_find_unused_parameters
