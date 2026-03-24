export CUDA_VISIBLE_DEVICES=3
EXPERIMENT_NAME=neural_hmm_upos_192
python main.py \
    train-test \
    nhmm \
    upos \
    ptb-trains \
    --experiment-name $EXPERIMENT_NAME \
    --max-epochs 20 5 \
    --nhmm-hidden-dim 192 \
