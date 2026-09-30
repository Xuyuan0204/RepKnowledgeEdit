python train_test.py \
 --num_samples 1000 \
 --dataset unke_v3 \
 --record True  \
 --adv_train_method Explicit \
 --wandb_project unke_single_qwen \
 --output_dir single_explicit_unke \
 --epochs 1000  \
 --noise_std 0.002 \
 --drop_out 0.05 \
 --rank 4 \
 --save_weights_dir single_unke_qwen_layer18 \
 --activation_path ./activation/unke_v3/qwen_2_5_7b_layer18_no_answer_last_original.pt \
 --original_query_activation_path  ./activation/unke_v3/qwen_2_5_7b_layer18_no_answer_last_original.pt \
 --rephrased_query_activation_path ./activation/unke_v3/qwen_2_5_7b_layer18_no_answer_last_rephrased.pt \
 --target_layer 18 \
 --model_name Qwen/Qwen2.5-7B-Instruct
 
 
