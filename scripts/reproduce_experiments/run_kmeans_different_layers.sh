#!/bin/bash

# Set CUDA device if needed
# export CUDA_VISIBLE_DEVICES=0

experiments=(
    "kmeans_upos_bert_first;train-test kmeans upos --experiment-name kmeans_upos_bert_first --max-epochs 50 --model-for-embeddings bert-base-uncased --layer-for-embeddings first"
    "kmeans_upos_bert_middle;train-test kmeans upos --experiment-name kmeans_upos_bert_middle --max-epochs 50 --model-for-embeddings bert-base-uncased --layer-for-embeddings middle"
    "kmeans_upos_bert_last;train-test kmeans upos --experiment-name kmeans_upos_bert_last --max-epochs 50 --model-for-embeddings bert-base-uncased --layer-for-embeddings last"
    "kmeans_upos_opt_first;train-test kmeans upos --experiment-name kmeans_upos_opt_first --max-epochs 50 --model-for-embeddings facebook/opt-125m --layer-for-embeddings first"
    "kmeans_upos_opt_middle;train-test kmeans upos --experiment-name kmeans_upos_opt_middle --max-epochs 50 --model-for-embeddings facebook/opt-125m --layer-for-embeddings middle"
    "kmeans_upos_opt_last;train-test kmeans upos --experiment-name kmeans_upos_opt_last --max-epochs 50 --model-for-embeddings facebook/opt-125m --layer-for-embeddings last"
    "kmeans_upos_qwen_first;train-test kmeans upos --experiment-name kmeans_upos_qwen_first --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings first"
    "kmeans_upos_qwen_middle;train-test kmeans upos --experiment-name kmeans_upos_qwen_middle --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings middle"
    "kmeans_upos_qwen_last;train-test kmeans upos --experiment-name kmeans_upos_qwen_last --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings last"
)

echo "Running kmeans with different layers..."

for run in "${experiments[@]}"; do
    IFS=";" read -r name args <<< "$run"
    echo "---------------------------------------"
    echo "Running: $name"

    python main.py $args

    echo "Finished $name"
done
