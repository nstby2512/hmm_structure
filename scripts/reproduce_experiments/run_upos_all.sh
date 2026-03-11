#!/bin/bash

# Set CUDA device if needed
# export CUDA_VISIBLE_DEVICES=0

experiments=(
    "standard_EM_upos;train-test hmm-EM upos --experiment-name standardEM_upos --max-epochs 10 5"
    "hard_EM_upos;train-test hmm-hardEM upos --experiment-name hard_EM_upos --max-epochs 10 5"
    "stepwise_EM_upos_0_6;train-test hmm-sEM upos --experiment-name stepwise_EM_upos_0_6 --max-epochs 10 5 --sEM-alpha 0.6"
    "stepwise_EM_upos_0_8;train-test hmm-sEM upos --experiment-name stepwise_EM_upos_0_8 --max-epochs 10 5 --sEM-alpha 0.8"
    "stepwise_EM_upos_1_0;train-test hmm-sEM upos --experiment-name stepwise_EM_upos_1_0 --max-epochs 10 5 --sEM-alpha 1.0"
    "neural_hmm_upos;train-test nhmm upos --experiment-name neural_hmm_upos --max-epochs 10 5"
    "kmeans_upos_bert;train-test kmeans upos --experiment-name kmeans_upos_bert --max-epochs 50 --model-for-embeddings bert-base-uncased --layer-for-embeddings middle"
    "kmeans_upos_opt;train-test kmeans upos --experiment-name kmeans_upos_opt --max-epochs 50 --model-for-embeddings facebook/opt-125m --layer-for-embeddings middle"
    "kmeans_upos_qwen;train-test kmeans upos --experiment-name kmeans_upos_qwen --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings middle"
)

echo "Running all upos experiments..."

for run in "${experiments[@]}"; do
    IFS=";" read -r name args <<< "$run"
    echo "---------------------------------------"
    echo "Running: $name"

    python main.py $args

    echo "Finished $name"
done
