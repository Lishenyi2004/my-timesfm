JOB_NAME="time-moe-train8"
WORKDIR="/mnt/shared-storage-gpfs2/speechllm-share/lishenyi/Time-MoE"
TRAIN_SCRIPT="${WORKDIR}/train.sh"

# 资源配置
GPU_PER_NODE=8
NUM_NODES=8
NAMESPACE="ailab-speechllm2"
CHARGED_GROUP="speechllm2_gpu"

# 计算资源
GPU=$GPU_PER_NODE
CPU=$((GPU * 16))
MEMORY=$((GPU * 2 * 120000))

rjob submit \
    --gpu=$GPU \
    --name=$JOB_NAME \
    --memory=$MEMORY \
    --cpu=$CPU \
    -P $NUM_NODES \
    --namespace=$NAMESPACE \
    --charged-group=$CHARGED_GROUP \
    -e GROUP=$CHARGED_GROUP \
    -e DISTRIBUTED_JOB=true \
    --private-machine=group \
    --image=registry.h.pjlab.org.cn/ailab-speechllm-speechllm_cpu/lsy:moe-fast \
    --mount=gpfs://gpfs1/lishenyi:/mnt/shared-storage-user/lishenyi \
    --mount=gpfs://gpfs1/brainllm-share:/mnt/shared-storage-user/brainllm-share \
    --mount=gpfs://gpfs2/speechllm-share:/mnt/shared-storage-gpfs2/speechllm-share \
    --custom-resources rdma/mlnx_shared=8 \
    --custom-resources brainpp.cn/fuse=1 \
    --custom-resources mellanox.com/mlnx_rdma=1 \
    --host-network=true \
    --preemptible=no \
    --gang-start=true \
    --auto-restart=false \
    -- bash "$TRAIN_SCRIPT"
