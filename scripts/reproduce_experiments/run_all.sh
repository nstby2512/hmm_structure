#!/bin/bash

# Set CUDA device if needed
export CUDA_VISIBLE_DEVICES=0

experiments=(
    # UPOS results, Table 1 and Figure 1
    "standard_EM_upos;train-test hmm-EM upos --experiment-name standardEM_upos --max-epochs 10 5"
    "hard_EM_upos;train-test hmm-hardEM upos --experiment-name hard_EM_upos --max-epochs 10 5"
    "stepwise_EM_upos_0_6;train-test hmm-sEM upos --experiment-name stepwise_EM_upos_0_6 --max-epochs 10 5 --sEM-alpha 0.6"
    "stepwise_EM_upos_0_8;train-test hmm-sEM upos --experiment-name stepwise_EM_upos_0_8 --max-epochs 10 5 --sEM-alpha 0.8"
    "stepwise_EM_upos_1_0;train-test hmm-sEM upos --experiment-name stepwise_EM_upos_1_0 --max-epochs 10 5 --sEM-alpha 1.0"
    "neural_hmm_upos;train-test nhmm upos --experiment-name neural_hmm_upos --max-epochs 10 5"
    "kmeans_upos_bert;train-test kmeans upos --experiment-name kmeans_upos_bert --max-epochs 50 --model-for-embeddings bert-base-uncased --layer-for-embeddings middle"
    "kmeans_upos_opt;train-test kmeans upos --experiment-name kmeans_upos_opt --max-epochs 50 --model-for-embeddings facebook/opt-125m --layer-for-embeddings middle"
    "kmeans_upos_qwen;train-test kmeans upos --experiment-name kmeans_upos_qwen --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings middle"
    # XPOS results, Table 2 and Figure 1
    "standard_EM_xpos;train-test hmm-EM xpos --experiment-name standardEM_xpos --max-epochs 10 5"
    "hard_EM_xpos;train-test hmm-hardEM xpos --experiment-name hard_EM_xpos --max-epochs 10 5"
    "stepwise_EM_xpos_0_6;train-test hmm-sEM xpos --experiment-name stepwise_EM_xpos_0_6 --max-epochs 10 5 --sEM-alpha 0.6"
    "stepwise_EM_xpos_0_8;train-test hmm-sEM xpos --experiment-name stepwise_EM_xpos_0_8 --max-epochs 10 5 --sEM-alpha 0.8"
    "stepwise_EM_xpos_1_0;train-test hmm-sEM xpos --experiment-name stepwise_EM_xpos_1_0 --max-epochs 10 5 --sEM-alpha 1.0"
    "neural_hmm_xpos;train-test nhmm xpos --experiment-name neural_hmm_xpos --max-epochs 10 5"
    "kmeans_xpos_bert;train-test kmeans xpos --experiment-name kmeans_xpos_bert --max-epochs 50 --model-for-embeddings bert-base-uncased --layer-for-embeddings middle"
    "kmeans_xpos_opt;train-test kmeans xpos --experiment-name kmeans_xpos_opt --max-epochs 50 --model-for-embeddings facebook/opt-125m --layer-for-embeddings middle"
    "kmeans_xpos_qwen;train-test kmeans xpos --experiment-name kmeans_xpos_qwen --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings middle"
    # Stepwise EM with different batch sizes, Figure 2
    "stepwise_EM_upos_1_0_bs_32;train-test hmm-sEM upos --experiment-name stepwise_EM_upos_1_0_bs_32 --max-epochs 10 5 --sEM-alpha 1.0 --batch-size 32"
    "stepwise_EM_upos_1_0_bs_64;train-test hmm-sEM upos --experiment-name stepwise_EM_upos_1_0_bs_64 --max-epochs 10 5 --sEM-alpha 1.0 --batch-size 64"
    "stepwise_EM_upos_1_0_bs_128;train-test hmm-sEM upos --experiment-name stepwise_EM_upos_1_0_bs_128 --max-epochs 10 5 --sEM-alpha 1.0 --batch-size 128"
    "stepwise_EM_upos_1_0_bs_256;train-test hmm-sEM upos --experiment-name stepwise_EM_upos_1_0_bs_256 --max-epochs 10 5 --sEM-alpha 1.0 --batch-size 256"
    "stepwise_EM_upos_1_0_bs_512;train-test hmm-sEM upos --experiment-name stepwise_EM_upos_1_0_bs_512 --max-epochs 10 5 --sEM-alpha 1.0 --batch-size 512"
    # Neural HMM with different hidden dimensions, Figure 3
    "neural_hmm_upos_hidden_64;train-test nhmm upos --experiment-name neural_hmm_upos_hidden_64 --max-epochs 10 5 --nhmm-hidden-dim 64"
    "neural_hmm_upos_hidden_128;train-test nhmm upos --experiment-name neural_hmm_upos_hidden_128 --max-epochs 10 5 --nhmm-hidden-dim 128"
    "neural_hmm_upos_hidden_192;train-test nhmm upos --experiment-name neural_hmm_upos_hidden_192 --max-epochs 10 5 --nhmm-hidden-dim 192"
    "neural_hmm_upos_hidden_256;train-test nhmm upos --experiment-name neural_hmm_upos_hidden_256 --max-epochs 10 5 --nhmm-hidden-dim 256"
    # K-means with different embedding layers, Figure 4
    "kmeans_upos_bert_first;train-test kmeans upos --experiment-name kmeans_upos_bert_first --max-epochs 50 --model-for-embeddings bert-base-uncased --layer-for-embeddings first"
    "kmeans_upos_bert_middle;train-test kmeans upos --experiment-name kmeans_upos_bert_middle --max-epochs 50 --model-for-embeddings bert-base-uncased --layer-for-embeddings middle"
    "kmeans_upos_bert_last;train-test kmeans upos --experiment-name kmeans_upos_bert_last --max-epochs 50 --model-for-embeddings bert-base-uncased --layer-for-embeddings last"
    "kmeans_upos_opt_first;train-test kmeans upos --experiment-name kmeans_upos_opt_first --max-epochs 50 --model-for-embeddings facebook/opt-125m --layer-for-embeddings first"
    "kmeans_upos_opt_middle;train-test kmeans upos --experiment-name kmeans_upos_opt_middle --max-epochs 50 --model-for-embeddings facebook/opt-125m --layer-for-embeddings middle"
    "kmeans_upos_opt_last;train-test kmeans upos --experiment-name kmeans_upos_opt_last --max-epochs 50 --model-for-embeddings facebook/opt-125m --layer-for-embeddings last"
    "kmeans_upos_qwen_first;train-test kmeans upos --experiment-name kmeans_upos_qwen_first --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings first"
    "kmeans_upos_qwen_middle;train-test kmeans upos --experiment-name kmeans_upos_qwen_middle --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings middle"
    "kmeans_upos_qwen_last;train-test kmeans upos --experiment-name kmeans_upos_qwen_last --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings last"
    # K-means with different embedding dimensions, Figure 5
    "kmeans_upos_qwen_embedding_dim_16;train-test kmeans upos --experiment-name kmeans_upos_qwen_embedding_dim_16 --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings middle --word-embedding-dim 16"
    "kmeans_upos_qwen_embedding_dim_32;train-test kmeans upos --experiment-name kmeans_upos_qwen_embedding_dim_32 --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings middle --word-embedding-dim 32"
    "kmeans_upos_qwen_embedding_dim_64;train-test kmeans upos --experiment-name kmeans_upos_qwen_embedding_dim_64 --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings middle --word-embedding-dim 64"
    "kmeans_upos_qwen_embedding_dim_128;train-test kmeans upos --experiment-name kmeans_upos_qwen_embedding_dim_128 --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings middle --word-embedding-dim 128"
    "kmeans_upos_qwen_embedding_dim_256;train-test kmeans upos --experiment-name kmeans_upos_qwen_embedding_dim_256 --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings middle --word-embedding-dim 256"
    "kmeans_upos_qwen_embedding_dim_384;train-test kmeans upos --experiment-name kmeans_upos_qwen_embedding_dim_384 --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings middle --word-embedding-dim 384"
    "kmeans_upos_qwen_embedding_dim_512;train-test kmeans upos --experiment-name kmeans_upos_qwen_embedding_dim_512 --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings middle --word-embedding-dim 512"
    "kmeans_upos_qwen_embedding_dim_640;train-test kmeans upos --experiment-name kmeans_upos_qwen_embedding_dim_640 --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings middle --word-embedding-dim 640"
    "kmeans_upos_qwen_embedding_dim_768;train-test kmeans upos --experiment-name kmeans_upos_qwen_embedding_dim_768 --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings middle --word-embedding-dim 768"
    "kmeans_upos_qwen_embedding_dim_896;train-test kmeans upos --experiment-name kmeans_upos_qwen_embedding_dim_896 --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings middle --word-embedding-dim 896"
    "kmeans_upos_qwen_embedding_dim_1024;train-test kmeans upos --experiment-name kmeans_upos_qwen_embedding_dim_1024 --max-epochs 50 --model-for-embeddings Qwen/Qwen3-0.6B --layer-for-embeddings middle --word-embedding-dim 1024"
)

echo "Running experiments..."

for run in "${experiments[@]}"; do
    IFS=";" read -r name args <<< "$run"
    echo "---------------------------------------"
    echo "Running: $name"

    python main.py $args

    echo "Finished $name"
done
