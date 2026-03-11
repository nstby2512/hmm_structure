#!/bin/bash

# Set CUDA device if needed
# export CUDA_VISIBLE_DEVICES=0

experiments=(
    "neural_hmm_upos_hidden_64;train-test nhmm upos --experiment-name neural_hmm_upos_hidden_64 --max-epochs 10 5 --nhmm-hidden-dim 64"
    "neural_hmm_upos_hidden_128;train-test nhmm upos --experiment-name neural_hmm_upos_hidden_128 --max-epochs 10 5 --nhmm-hidden-dim 128"
    "neural_hmm_upos_hidden_192;train-test nhmm upos --experiment-name neural_hmm_upos_hidden_192 --max-epochs 10 5 --nhmm-hidden-dim 192"
    "neural_hmm_upos_hidden_256;train-test nhmm upos --experiment-name neural_hmm_upos_hidden_256 --max-epochs 10 5 --nhmm-hidden-dim 256"
)

echo "Running neural HMM with different hidden dimensions..."

for run in "${experiments[@]}"; do
    IFS=";" read -r name args <<< "$run"
    echo "---------------------------------------"
    echo "Running: $name"

    python main.py $args

    echo "Finished $name"
done
