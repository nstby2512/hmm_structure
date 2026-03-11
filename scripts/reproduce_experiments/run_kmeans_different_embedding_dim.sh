#!/bin/bash

# Set CUDA device if needed
# export CUDA_VISIBLE_DEVICES=0

experiments=(
    "kmeans_upos_qwen_embedding_dim_16;train-test kmeans upos --experiment-name kmeans_upos_qwen_embedding_dim_16 --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings middle --word-embedding-dim 16"
    "kmeans_upos_qwen_embedding_dim_32;train-test kmeans upos --experiment-name kmeans_upos_qwen_embedding_dim_32 --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings middle --word-embedding-dim 32"
    "kmeans_upos_qwen_embedding_dim_64;train-test kmeans upos --experiment-name kmeans_upos_qwen_embedding_dim_64 --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings middle --word-embedding-dim 64"
    "kmeans_upos_qwen_embedding_dim_128;train-test kmeans upos --experiment-name kmeans_upos_qwen_embedding_dim_128 --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings middle --word-embedding-dim 128"
    "kmeans_upos_qwen_embedding_dim_256;train-test kmeans upos --experiment-name kmeans_upos_qwen_embedding_dim_256 --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings middle --word-embedding-dim 256"
    "kmeans_upos_qwen_embedding_dim_384;train-test kmeans upos --experiment-name kmeans_upos_qwen_embedding_dim_384 --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings middle --word-embedding-dim 384"
    "kmeans_upos_qwen_embedding_dim_512;train-test kmeans upos --experiment-name kmeans_upos_qwen_embedding_dim_512 --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings middle --word-embedding-dim 512"
    "kmeans_upos_qwen_embedding_dim_640;train-test kmeans upos --experiment-name kmeans_upos_qwen_embedding_dim_640 --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings middle --word-embedding-dim 640"
    "kmeans_upos_qwen_embedding_dim_768;train-test kmeans upos --experiment-name kmeans_upos_qwen_embedding_dim_768 --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings middle --word-embedding-dim 768"
    "kmeans_upos_qwen_embedding_dim_896;train-test kmeans upos --experiment-name kmeans_upos_qwen_embedding_dim_896 --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings middle --word-embedding-dim 896"
    "kmeans_upos_qwen_embedding_dim_1024;train-test kmeans upos --experiment-name kmeans_upos_qwen_embedding_dim_1024 --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings middle --word-embedding-dim 1024"
)

echo "Running kmeans with different embedding dimensions..."

for run in "${experiments[@]}"; do
    IFS=";" read -r name args <<< "$run"
    echo "---------------------------------------"
    echo "Running: $name"

    python main.py $args

    echo "Finished $name"
done
