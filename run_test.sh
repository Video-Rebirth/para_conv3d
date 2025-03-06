#!/bin/bash

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=4,5,6,7

# export TORCH_NCCL_DESYNC_DEBUG=1  # Helps identify rank desync issues:cite[5]
# export CUBLAS_WORKSPACE_CONFIG=:4096:8

torchrun \
        --nnodes=1 \
        --node_rank=0 \
        --nproc_per_node=4 \
        --master_addr=127.0.0.1 \
        --master_port=12345 \
        test_para_conv3d_train.py