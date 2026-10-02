#!/bin/bash
set -e

# End-to-end clustered-training pipeline for RILKE.
#
# NOTE: the clustered scripts (cluster_activation.py / train_cluster.py /
# test_cluster_rep.py) are built around LLaMA-3.1-8B-Instruct at layer 15 on
# UnKE-v3. These are fixed below so every stage stays consistent (the cluster
# index, the retrieval activations, and the trained modules must all come from
# the same model / layer / dataset). The individual pipeline (train_test.py)
# is the one that also supports Qwen.

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
DATASET="unke_v3"
TARGET_LAYER=15
MODEL_PREFIX="llama_3_8b"

# Safe-to-tune knobs.
RANK=${RANK:-4}
EPOCHS=${EPOCHS:-1000}
NUM_SAMPLES=${NUM_SAMPLES:-1000}
ADV_METHOD=${ADV_METHOD:-"Explicit"}
SAVE_WEIGHTS_DIR=${SAVE_WEIGHTS_DIR:-"cluster_unke_explicit_llama"}
WANDB_PROJECT=${WANDB_PROJECT:-"rilke_cluster"}

ACTIVATION_DIR="./activation/${DATASET}"
ACTIVATION_ORIG="${ACTIVATION_DIR}/${MODEL_PREFIX}_layer${TARGET_LAYER}_no_answer_last_original.pt"
ACTIVATION_REPHRASE="${ACTIVATION_DIR}/${MODEL_PREFIX}_layer${TARGET_LAYER}_no_answer_last_rephrased.pt"
CLUSTER_INDEX="./cluster_index/unke/unke_v3_3_hac_maxsize8.json"

echo "============================================"
echo "RILKE Cluster Pipeline"
echo "Model:        ${MODEL_NAME}"
echo "Dataset:      ${DATASET}"
echo "Target Layer: ${TARGET_LAYER}"
echo "============================================"

echo ""
echo "[Step 1/5] Storing activations (original queries)..."
if [ -f "$ACTIVATION_ORIG" ]; then
  echo "  exists, skipping: $ACTIVATION_ORIG"
else
  python store_activation.py \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET" \
    --target_layer "$TARGET_LAYER" \
    --data_src original
fi

echo ""
echo "[Step 2/5] Storing activations (rephrased queries)..."
if [ -f "$ACTIVATION_REPHRASE" ]; then
  echo "  exists, skipping: $ACTIVATION_REPHRASE"
else
  python store_activation.py \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET" \
    --target_layer "$TARGET_LAYER" \
    --data_src rephrased
fi

echo ""
echo "[Step 3/5] Clustering activations (HAC) -> ${CLUSTER_INDEX}..."
if [ -f "$CLUSTER_INDEX" ]; then
  echo "  exists, skipping: $CLUSTER_INDEX"
else
  python cluster_activation.py
fi

echo ""
echo "[Step 4/5] Training cluster-based intervention modules..."
CUDA_VISIBLE_DEVICES=2 \
python train_cluster.py \
  --dataset "$DATASET" \
  --model_name "$MODEL_NAME" \
  --adv_train_method "$ADV_METHOD" \
  --learning_rate 1e-2 \
  --drop_out 0.05 \
  --noise_std 0.02 \
  --lambda_consistency 0.001 \
  --batch_size 8 \
  --rank "$RANK" \
  --epochs "$EPOCHS" \
  --cluster_method hac \
  --attn_impl sdpa \
  --cluster_indices_path "$CLUSTER_INDEX" \
  --save_weights_dir "$SAVE_WEIGHTS_DIR" \
  --record True \
  --wandb_project "$WANDB_PROJECT"

echo ""
echo "[Step 5/5] Evaluating with cluster-based retrieval..."
CUDA_VISIBLE_DEVICES=2 \
python test_cluster_rep.py \
  --dataset "$DATASET" \
  --model_name "$MODEL_NAME" \
  --adapter_weights_dir "./Stored_weights/${SAVE_WEIGHTS_DIR}" \
  --activation_path "$ACTIVATION_ORIG" \
  --original_query_activation_path "$ACTIVATION_ORIG" \
  --rephrased_query_activation_path "$ACTIVATION_REPHRASE" \
  --target_layer "$TARGET_LAYER" \
  --rank "$RANK" \
  --num_samples "$NUM_SAMPLES" \
  --cluster_indices_path "$CLUSTER_INDEX" \
  --save_path "cluster_${DATASET}_results"

echo ""
echo "============================================"
echo "Pipeline complete!"
echo "============================================"
