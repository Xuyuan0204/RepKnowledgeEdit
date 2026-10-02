# RILKE: Representation Interventions Enable Lifelong Knowledge Memory Control in LLMs

[![ACL 2026](https://img.shields.io/badge/ACL%202026-Oral-blue.svg)](https://arxiv.org/abs/2511.20892)

Official implementation of **"Representation Interventions Enable Lifelong Knowledge Memory Control in LLMs"** (ACL 2026, Oral).

## Overview

**RILKE** (**R**epresentation **I**ntervention for **L**ifelong **K**nowledg**E** Control) enables efficient and accurate knowledge updates in large language models without costly retraining. The method operates within the model's representation space using two key components:

- **Training**: Learns paraphrase-robust and edit-localized intervention modules that confine each update to a low-dimensional subspace, minimizing cross-edit interference.
- **Inference**: A query-adaptive router dynamically selects the appropriate intervention module via activation-based cosine similarity retrieval.

RILKE supports both **individual training** (one module per edit) and **clustered training** (shared modules for semantically similar edits), and has been evaluated on LLaMA-3.1-8B-Instruct and Qwen2.5-7B-Instruct, achieving high edit success rates and strong generalization with modest memory overhead.

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
├── REFT_module.py           # Intervention modules (Vanilla, Explicit, Implicit, Adv_Explicit)
├── REFT_trainer.py          # Custom trainers with consistency/adversarial losses
├── utils.py                 # Weight loading, reinitialization, BERT score utilities
├── store_activation.py      # Store per-sample activations at target layer
├── cluster_activation.py    # Cluster activations (KMeans, HAC)
├── no_batched_train.py      # Individual training (one module per data point)
├── train_test.py            # Unified train + test for individual setting
├── test_single_rep.py       # Evaluation with activation-based retrieval (individual)
├── test_rep.py              # Basic evaluation with pre-trained interventions
├── train_cluster.py         # Clustered training (shared module per cluster)
├── test_cluster_rep.py      # Evaluation with cluster-based retrieval
├── train_test_single.sh     # Example individual train+test script
├── run_cluster_pipeline.sh  # End-to-end clustered pipeline (store → cluster → train → eval)
├── src/dataset/             # UnKE dataset loader
├── datasets/UnKE/           # UnKE data (final_data_v2.json, final_data_v3.json)
├── cluster_index/           # Precomputed cluster index for batched training
└── result/                  # Example outputs
```

## Quick Start

The RILKE pipeline consists of three stages:

1. **Store activations** at the target layer for each data point
2. **Train** intervention modules (individually or per cluster)
3. **Evaluate** using activation-based retrieval to select the appropriate module at inference

---

## Individual Training

Individual training learns one lightweight intervention module per data point, then retrieves the best-matching module at test time via cosine similarity over stored activations.

### Step 1: Store Activations

Extract hidden-state activations at the target layer for both original and paraphrased queries:

```bash
# Original queries
python store_activation.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --dataset_name unke_v3 \
  --data_src original

# Paraphrased queries (for generalization evaluation)
python store_activation.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --dataset_name unke_v3 \
  --data_src rephrased
```

### Step 2: Train

Train one intervention module per data point. Each module is reinitialized before training on its corresponding sample, then saved for later retrieval.

```bash
python no_batched_train.py \
  --dataset unke_v3 \
  --adv_train_method Explicit \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --rank 4 \
  --epochs 1000 \
  --learning_rate 1e-2 \
  --noise_std 0.02 \
  --lambda_consistency 0.001 \
  --save_weights_dir single_unke_llama \
  --record True \
  --wandb_project rilke_individual
```

**Supported training methods** (`--adv_train_method`):
| Method | Description |
|---|---|
| `Vanilla` | Standard LoReFT intervention |
| `Explicit` | LoReFT with noise-based explicit regularization (Section 4.1) |
| `Implicit` | LoReFT with consistency loss on rotated representations |
| `Adv_Explicit` | LoReFT with adversarial perturbation training |

### Step 3: Evaluate

At inference, each query's activation is compared against stored training activations. The intervention module with the highest cosine similarity is loaded and applied:

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

### Unified Train + Test

`train_test.py` provides a single-command interface that trains all individual modules and then evaluates them:

```bash
python train_test.py \
  --num_samples 1000 \
  --dataset unke_v3 \
  --adv_train_method Explicit \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --rank 4 \
  --epochs 1000 \
  --save_weights_dir single_unke_qwen_layer18 \
  --activation_path ./activation/unke_v3/qwen_2_5_7b_layer18_no_answer_last_original.pt \
  --original_query_activation_path ./activation/unke_v3/qwen_2_5_7b_layer18_no_answer_last_original.pt \
  --rephrased_query_activation_path ./activation/unke_v3/qwen_2_5_7b_layer18_no_answer_last_rephrased.pt \
  --target_layer 18
```

---

## Clustered Training

Clustered training groups semantically similar data points and trains a shared intervention module per cluster, reducing storage while maintaining performance.

> **End-to-end:** `bash run_cluster_pipeline.sh` runs the four stages below in order (store → cluster → train → evaluate). The clustered scripts are configured for LLaMA-3.1-8B-Instruct at layer 15 on UnKE-v3; tune `RANK`, `EPOCHS`, `NUM_SAMPLES`, `ADV_METHOD`, and `SAVE_WEIGHTS_DIR` via environment variables.

### Step 1: Cluster Activations

Group stored activations into size-bounded clusters using Hierarchical Agglomerative Clustering (HAC) with a cosine similarity threshold:

```bash
python cluster_activation.py
```

Output: `cluster_index/unke/unke_v3_3_hac_maxsize8.json`

The clustering uses `tau=0.9` (cosine similarity threshold) and `max_cluster_size=8` by default. Large clusters are recursively split to respect the size bound.

### Step 2: Train

All clustered-training settings live in [`configs/cluster.yaml`](configs/cluster.yaml) — edit them there instead of passing long flag lists. Any flag you also pass on the command line overrides the file.

```bash
python train_cluster.py --config configs/cluster.yaml
```

### Step 3: Evaluate

```bash
python test_cluster_rep.py --config configs/cluster.yaml
```

---

## Evaluation Metrics

- **ROUGE-L** (recall): Lexical overlap between generated and reference answers
- **BERT Score** (cosine similarity): Semantic similarity using sentence-transformers (`all-MiniLM-L6-v2`)

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

This project builds on [pyreft](https://github.com/stanfordnlp/pyreft). We thank the authors for their open-source contributions.

## Contact
If you have any questions, suggestions, or bug reports, please contact

xuyuan.liu.gr@dartmouth.edu
