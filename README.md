# Unsupervised Part-of-Speech Tagging

This project employs a subset of Penn Treebank dataset and evaluates HMM and K-means on the PoS tagging problem.

## Installation
```bash
conda create -n nlp python=3.12 -y
conda activate nlp
pip install -r requirements.txt
```

## Reproduce Results
To reproduce results in the report, please set the corresponding GPU device and execute:
```bash
bash scripts/reproduce_experiments/run_all.sh
```

## Repository Structure
```
.
├── figs
│   ├── embedding_layers_analysis_upos.pdf
│   ├── kmeans_embedding_dim_upos.pdf
│   ├── nhmm_hidden_dim_upos.pdf
│   ├── stepwise_EM_batch_sizes_upos.pdf
│   └── training_curves.pdf
├── main.py
├── plot
│   ├── fig_1_plot.py
│   ├── fig_2_plot_stepwise_EM_batch_sizes.py
│   ├── fig_3_plot_nhmm_hidden_dim.py
│   ├── fig_4_plot_kmeans_layer_analysis.py
│   └── fig_5_plot_kmeans_embedding_dim.py
├── pos_tagging
│   ├── __init__.py
│   ├── base.py
│   ├── hmm_pipeline.py
│   ├── hmm.py
│   ├── kmeans.py
│   └── nhmm.py
├── ptb-train.conllu
├── README.md
├── requirements.txt
├── scripts
│   ├── individual_model
│   │   ├── run_hard_EM.sh
│   │   ├── run_kmeans.sh
│   │   ├── run_neural_hmm.sh
│   │   ├── run_standard_EM.sh
│   │   └── run_stepwise_EM.sh
│   └── reproduce_experiments
│       ├── run_all.sh
│       ├── run_kmeans_different_embedding_dim.sh
│       ├── run_kmeans_different_layers.sh
│       ├── run_neural_hmm_different_hidden_dim.sh
│       ├── run_sEM_different_batch_sizes.sh
│       ├── run_upos_all.sh
│       └── run_xpos_all.sh
└── utils
    ├── argparser.py
    ├── logging_nlp.py
    ├── preprocess_dataset.py
    └── utils.py
```