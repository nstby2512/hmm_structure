# export HF_CACHE_DIR=
EXPERIMENT_NAME=kmeans_qwen_upos_middle_dim_512
export CUDA_VISIBLE_DEVICES=3
python main.py \
    train-test \
    kmeans \
    upos \
    --experiment-name $EXPERIMENT_NAME \
    --max-epochs 50 \
    --model-for-embeddings Qwen/Qwen3-0.6B \
    --layer-for-embeddings middle \
    --word-embedding-dim 512 \
    # --batch-size 256 \
    # hmm-EM \
    # --load-path logs/bEM_forward_backward_20_epochs/checkpoint.pt
    # hmm-hardEM \
