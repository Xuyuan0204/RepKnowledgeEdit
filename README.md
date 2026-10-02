# RILKE: Representation Interventions Enable Lifelong Knowledge Memory Control in LLMs

[![ACL 2026](https://img.shields.io/badge/ACL%202026-Oral-blue.svg)](https://arxiv.org/abs/2511.20892)

Official implementation of **"Representation Interventions Enable Lifelong Knowledge Memory Control in LLMs"** (ACL 2026, Oral).

## Overview

**RILKE** (**R**epresentation **I**ntervention for **L**ifelong **K**nowledg**E** Control) updates knowledge in large language models efficiently and accurately, without retraining. It operates in the model's representation space through two components:

- **Training** learns paraphrase-robust, edit-localized intervention modules, each confined to a low-dimensional subspace to minimize cross-edit interference.
- **Inference** uses a query-adaptive router that selects the right module via activation-based cosine similarity.

RILKE supports **individual training** (one module per edit) and **clustered training** (one shared module per group of similar edits). Evaluated on LLaMA-3.1-8B-Instruct and Qwen2.5-7B-Instruct, it achieves high edit success and strong generalization with modest memory overhead.

## Installation

```bash
conda create -n rilke python=3.10
conda activate rilke
pip install -r requirements.txt
```

**Core dependencies**: PyTorch, Transformers, [pyreft](https://github.com/stanfordnlp/pyreft), [pyvene](https://github.com/stanfordnlp/pyvene), sentence-transformers, rouge-score, scikit-learn, wandb (optional).

## Project Structure

```
RILKE/
├── REFT_module.py           # LoReFT intervention modules (Explicit)
├── REFT_trainer.py          # Custom trainers with consistency/adversarial losses
├── utils.py                 # Weight loading, reinitialization, BERT-score utilities
├── store_activation.py      # Store per-sample activations at the target layer
├── cluster_activation.py    # Cluster activations (KMeans, HAC)
├── no_batched_train.py      # Individual training (one module per edit)
├── train_test.py            # Unified train + test for the individual setting
├── test_single_rep.py       # Individual-setting evaluation with activation retrieval
├── test_rep.py              # Basic evaluation with pre-trained interventions
├── train_cluster.py         # Clustered training (one shared module per cluster)
├── test_cluster_rep.py      # Clustered-setting evaluation with activation retrieval
├── configs/cluster.yaml     # Settings for the clustered pipeline
├── train_test_single.sh     # Example individual train+test script
├── run_cluster_pipeline.sh  # End-to-end clustered pipeline (store → cluster → train → eval)
├── src/dataset/             # UnKE dataset loader
├── datasets/UnKE/           # UnKE data (final_data_v2.json, final_data_v3.json)
├── cluster_index/           # Precomputed cluster index for clustered training
└── result/                  # Example outputs
```

Every workflow follows three stages: **store activations** → **train** intervention modules (individual or clustered) → **evaluate** via activation-based retrieval.

---

## Individual Training

One lightweight module per edit; at test time, each query is routed to the most similar stored module by cosine similarity.

**1. Store activations** for the original and paraphrased queries:

```bash
python store_activation.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset_name unke_v3 --data_src original
python store_activation.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset_name unke_v3 --data_src rephrased
```

**2. Train** one module per edit (each is reinitialized, trained on its sample, and saved):

```bash
python no_batched_train.py \
  --dataset unke_v3 \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --adv_train_method Explicit \
  --rank 4 \
  --epochs 1000 \
  --learning_rate 1e-2 \
  --noise_std 0.02 \
  --lambda_consistency 0.001 \
  --save_weights_dir single_unke_llama \
  --record True \
  --wandb_project rilke_individual
```

**3. Evaluate** (each query is routed to its most similar stored module):

```bash
python test_single_rep.py \
  --dataset unke_v3 \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --adapter_weights_dir ./Stored_weights/single_unke_llama \
  --activation_path ./activation/unke_v3/llama_3_8b_layer15_no_answer_last_original.pt \
  --original_query_activation_path ./activation/unke_v3/llama_3_8b_layer15_no_answer_last_original.pt \
  --rephrased_query_activation_path ./activation/unke_v3/llama_3_8b_layer15_no_answer_last_rephrased.pt \
  --target_layer 15 \
  --rank 4 \
  --num_samples 1000 \
  --save_path individual_unke_results
```

`train_test.py` runs training and evaluation in a single command (shown here for Qwen at layer 18):

```bash
python train_test.py \
  --num_samples 1000 \
  --dataset unke_v3 \
  --adv_train_method Explicit \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --rank 4 \
  --epochs 1000 \
  --target_layer 18 \
  --save_weights_dir single_unke_qwen_layer18 \
  --activation_path ./activation/unke_v3/qwen_2_5_7b_layer18_no_answer_last_original.pt \
  --original_query_activation_path ./activation/unke_v3/qwen_2_5_7b_layer18_no_answer_last_original.pt \
  --rephrased_query_activation_path ./activation/unke_v3/qwen_2_5_7b_layer18_no_answer_last_rephrased.pt
```

---

## Clustered Training

Groups semantically similar edits and trains one shared module per cluster, reducing storage while preserving performance. Run everything with `bash run_cluster_pipeline.sh`, or follow the steps below. All settings live in [`configs/cluster.yaml`](configs/cluster.yaml); any CLI flag overrides the file.

**1. Cluster activations** with size-bounded Hierarchical Agglomerative Clustering (HAC):

```bash
python cluster_activation.py   # → cluster_index/unke/unke_v3_3_hac_maxsize8.json
```

Defaults: cosine threshold `tau=0.9` and `max_cluster_size=8`; oversized clusters are split recursively.

**2. Train**:

```bash
python train_cluster.py --config configs/cluster.yaml
```

**3. Evaluate**:

```bash
python test_cluster_rep.py --config configs/cluster.yaml
```

---

## Evaluation Metrics

- **ROUGE-L** (recall) — lexical overlap with the reference answer.
- **BERT Score** — semantic similarity via sentence-transformers (`all-MiniLM-L6-v2`).

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{DBLP:conf/acl/LiuCDLZWYCC26,
  author       = {Xuyuan Liu and
                  Shengyu Chen and
                  Xinshuai Dong and
                  Yanchi Liu and
                  Xujiang Zhao and
                  Haoyu Wang and
                  Yujun Yan and
                  Haifeng Chen and
                  Zhengzhang Chen},
  editor       = {Maria Liakata and
                  Viviane P. Moreira and
                  Jiajun Zhang and
                  David Jurgens},
  title        = {Representation Interventions Enable Lifelong Knowledge Memory Control
                  in LLMs},
  booktitle    = {Proceedings of the 64th Annual Meeting of the Association for Computational
                  Linguistics (Volume 1: Long Papers), {ACL} 2026, San Diego, California,
                  United States, July 2-7, 2026},
  pages        = {5414--5436},
  publisher    = {Association for Computational Linguistics},
  year         = {2026},
  url          = {https://doi.org/10.18653/v1/2026.acl-long.246},
  doi          = {10.18653/V1/2026.ACL-LONG.246},
  timestamp    = {Thu, 30 Jul 2026 17:33:57 +0200},
  biburl       = {https://dblp.org/rec/conf/acl/LiuCDLZWYCC26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## Acknowledgements

Built on [pyreft](https://github.com/stanfordnlp/pyreft). We thank the authors for their open-source work.

## Contact

Questions or suggestions reports: xuyuan.liu.gr@dartmouth.edu
