EXPERIMENT_NAME=hardEM_xpos
python main.py \
    train-test \
    hmm-hardEM \
    xpos \
    --experiment-name $EXPERIMENT_NAME \
    --max-epochs 10 5 \
