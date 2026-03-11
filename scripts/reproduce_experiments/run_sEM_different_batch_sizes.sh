#!/bin/bash

# Set CUDA device if needed
# export CUDA_VISIBLE_DEVICES=0

experiments=(
    "stepwise_EM_upos_1_0_bs_32;train-test hmm-sEM upos --experiment-name stepwise_EM_upos_1_0_bs_32 --max-epochs 10 5 --sEM-alpha 1.0 --batch-size 32"
    "stepwise_EM_upos_1_0_bs_64;train-test hmm-sEM upos --experiment-name stepwise_EM_upos_1_0_bs_64 --max-epochs 10 5 --sEM-alpha 1.0 --batch-size 64"
    "stepwise_EM_upos_1_0_bs_128;train-test hmm-sEM upos --experiment-name stepwise_EM_upos_1_0_bs_128 --max-epochs 10 5 --sEM-alpha 1.0 --batch-size 128"
    "stepwise_EM_upos_1_0_bs_256;train-test hmm-sEM upos --experiment-name stepwise_EM_upos_1_0_bs_256 --max-epochs 10 5 --sEM-alpha 1.0 --batch-size 256"
    "stepwise_EM_upos_1_0_bs_512;train-test hmm-sEM upos --experiment-name stepwise_EM_upos_1_0_bs_512 --max-epochs 10 5 --sEM-alpha 1.0 --batch-size 512"
)

echo "Running stepwise EM with different batch sizes..."

for run in "${experiments[@]}"; do
    IFS=";" read -r name args <<< "$run"
    echo "---------------------------------------"
    echo "Running: $name"

    python main.py $args

    echo "Finished $name"
done
