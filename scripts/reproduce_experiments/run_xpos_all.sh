#!/bin/bash

# Set CUDA device if needed
# export CUDA_VISIBLE_DEVICES=0

experiments=(
    "standard_EM_xpos;train-test hmm-EM xpos --experiment-name standardEM_xpos --max-epochs 10 5"
    "hard_EM_xpos;train-test hmm-hardEM xpos --experiment-name hard_EM_xpos --max-epochs 10 5"
    "stepwise_EM_xpos_0_6;train-test hmm-sEM xpos --experiment-name stepwise_EM_xpos_0_6 --max-epochs 10 5 --sEM-alpha 0.6"
    "stepwise_EM_xpos_0_8;train-test hmm-sEM xpos --experiment-name stepwise_EM_xpos_0_8 --max-epochs 10 5 --sEM-alpha 0.8"
    "stepwise_EM_xpos_1_0;train-test hmm-sEM xpos --experiment-name stepwise_EM_xpos_1_0 --max-epochs 10 5 --sEM-alpha 1.0"
    "neural_hmm_xpos;train-test nhmm xpos --experiment-name neural_hmm_xpos --max-epochs 10 5"
    "kmeans_xpos_bert;train-test kmeans xpos --experiment-name kmeans_xpos_bert --max-epochs 50 --model-for-embeddings bert-base-uncased --layer-for-embeddings middle"
    "kmeans_xpos_opt;train-test kmeans xpos --experiment-name kmeans_xpos_opt --max-epochs 50 --model-for-embeddings facebook/opt-125m --layer-for-embeddings middle"
    "kmeans_xpos_qwen;train-test kmeans xpos --experiment-name kmeans_xpos_qwen --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings middle"
)

echo "Running all xpos experiments..."

for run in "${experiments[@]}"; do
    IFS=";" read -r name args <<< "$run"
    echo "---------------------------------------"
    echo "Running: $name"

    python main.py $args

    echo "Finished $name"
done
