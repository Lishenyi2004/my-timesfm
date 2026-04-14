#!/usr/bin/env bash
set -e

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

if [ -z "${NODE_RANK}" ] || [ -z "${MASTER_ADDR}" ]; then
    echo "Error: 平台环境变量缺失！请确保使用了 -e DISTRIBUTED_JOB=true"
    env | grep -E "NODE|MASTER"
    exit 1
fi

export MASTER_PORT=29500

if [ -n "${NCCL_SOCKET_IFNAME}" ]; then
    export GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME%%,*}
    export TP_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME%%,*}
fi

export RANK=${NODE_RANK}
export WORLD_SIZE=${NODE_COUNT}
export NNODES=${NODE_COUNT}

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7
export NCCL_DEBUG=WARN

cd /mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE

DATA_PATH="/mnt/shared-storage-gpfs2/speechllm-share/lishenyi/datasets/time300b/b52d0ca9de8da5202f73a5057681fca6b48906fb_m4_symlinks/m4_hourly"
TEACHER_PATH="/mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE/timesfm-2.5-200m-pytorch"

exec python torch_dist_run.py distill_main.py \
    -d "$DATA_PATH" \
    --teacher_model_path "$TEACHER_PATH" \
    --student_num_layers 8 \
    --init_student_from_teacher \
    --distill_supervised_weight 1.0 \
    --distill_point_weight 1.0 \
    --distill_quantile_weight 0.5 \
    --global_batch_size 1024 \
    --micro_batch_size 1024 \
    --evaluation_strategy steps \
    --eval_steps 4000 \
    --save_strategy steps \
    --save_steps 4000 \
    --load_best_model_at_end \
    --dataloader_num_workers 1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --cosine_num_cycles 0.5 \
    --learning_rate 1e-3 \
    --min_learning_rate 1e-5 \
    --weight_decay 0.05 \
    --ddp_timeout 7200 \
    --precision fp32 \
    --num_train_epochs 1000 \
    --normalization_method none \
    --output_path logs_2/time300b_distill_m4_hourly_t_all \
    --ddp_find_unused_parameters \
    --enable_overfit_fixed_window \
    --overfit_hist_length 384 \
    --overfit_gt_length 128
