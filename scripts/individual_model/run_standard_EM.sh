EXPERIMENT_NAME=bEM_upos
python main.py \
    train-test \
    hmm-EM \
    upos \
    --experiment-name $EXPERIMENT_NAME \
    --max-epochs 10 5 \
