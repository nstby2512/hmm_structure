EXPERIMENT_NAME=sEM_upos_1_0_bs_128
python main.py \
    train-test \
    hmm-sEM \
    upos \
    --experiment-name $EXPERIMENT_NAME \
    --max-epochs 10 5 \
    --sEM-alpha 1.0 \
    --batch-size 128 \