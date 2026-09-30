from ast import Try

from dataclasses import dataclass, field
import pandas as pd
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
import torch
import transformers
from datasets import Dataset
from transformers import Trainer, TrainerCallback
import os
import json
from pyreft import (
    TaskType,
    get_reft_model,
    ReftConfig,
    ReftTrainerForCausalLM, 
    ReftDataCollator,
    ReftSupervisedDataset,
    make_last_position_supervised_data_module,
    ConsreftIntervention,
    LoreftIntervention
)
from src.dataset.unke import UnkeForDirectOpt
import wandb
from REFT_module import LoreftIntervention_Implicit, LoreftIntervention_Explicit, LoreftIntervention_Adv_Explicit
from REFT_trainer import ReftTrainerImplicit, ReftTrainerAdv, ReftTrainerCertified
from utils import reinit_intervention_weights
import argparse
from itertools import islice


seed = 1024
torch.manual_seed(seed)


@dataclass
class ReftHyperparameters:
    num_samples: int = 8  
    rank: int = 4  # low_rank_dimension

    target_layer: int = 15
    batch_size: int = 8
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    record: bool = False
    wandb_project: str = "reft_cluster_batch"
    dataset: str = "unke"
    noise_std: float = 0.005  
    drop_out: float = 0.05
    adv_train_method: str = "Adv_Explicit"
    epochs: int = 1000
    learning_rate: float = 2e-3
    adv_epsilon: float = 0.001
    adv_steps: int = 3
    adv_norm: str = "l2"
    lambda_consistency: float = 0.01
    output_dir: str = "reft_results"
    cluster_method: str = "kmeans"
    cluster_indices_path: str = "./cluster_index/unke/unke_v3_3_hac_maxsize8.json"
    save_weights_dir: str = None


def get_config(args):
    config = ReftHyperparameters()
    config.num_samples = args.num_samples
    config.batch_size = args.batch_size
    config.record = args.record
    config.wandb_project = args.wandb_project
    config.dataset = args.dataset
    config.noise_std = args.noise_std
    config.drop_out = args.drop_out
    config.adv_train_method = args.adv_train_method
    config.epochs = args.epochs
    config.learning_rate = args.learning_rate
    config.lambda_consistency = args.lambda_consistency
    config.adv_steps = args.adv_steps
    config.adv_norm = args.adv_norm
    config.adv_epsilon = args.adv_epsilon
    config.rank = args.rank
    config.output_dir = args.output_dir
    config.cluster_method = args.cluster_method
    config.cluster_indices_path = args.cluster_indices_path
    config.save_weights_dir = args.save_weights_dir
    return config


scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


def Reft_train(config):

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name, model_max_length=512, 
        padding_side="right", use_fast=False
        
        )

    if config.dataset == "unke":
        edit_dataset = UnkeForDirectOpt().get_dataset()
    elif config.dataset == "unke_v3":
        edit_dataset = UnkeForDirectOpt().get_dataset_v3()


    print(f"Training on full dataset with {config.num_samples} samples")

    data_examples = edit_dataset
   
    edit_questions = [sample["question"] for sample in data_examples]
    edit_answers = [sample["answer"] for sample in data_examples]
    edit_rephrase_questions = [sample["para_question"] for sample in data_examples]


    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.bfloat16, device_map=device,attn_implementation="flash_attention_2",low_cpu_mem_usage=True,)


    if config.record:
        wandb.init(project=config.wandb_project)


    tokenizer.pad_token = tokenizer.unk_token

    if config.adv_train_method == "Explicit":
        print("Using Explicit ReFT")
        reft_config = ReftConfig(representations={
            "layer": config.target_layer, "component": "block_output",
            "intervention": LoreftIntervention_Explicit(
                embed_dim=model.config.hidden_size,
                low_rank_dimension=config.rank,
                dropout_rate=config.drop_out,
                init_noise_std=config.noise_std
            )})
    elif config.adv_train_method == "Implicit":
        print("Using Implicit ReFT")
        reft_config = ReftConfig(representations={
            "layer": config.target_layer, "component": "block_output",
            "intervention": LoreftIntervention_Implicit(
                embed_dim=model.config.hidden_size,
                low_rank_dimension=config.rank,
                dropout_rate=config.drop_out,
                init_noise_std=config.noise_std
            )})
    elif config.adv_train_method == "Adv_Explicit":
        print("Using Adversarial Explicit ReFT")
        reft_config = ReftConfig(representations={
            "layer": config.target_layer, "component": "block_output",
            "intervention": LoreftIntervention_Adv_Explicit(
                embed_dim=model.config.hidden_size,
                low_rank_dimension=config.rank,
                dropout_rate=config.drop_out,
                adv_epsilon=config.adv_epsilon,       # magnitude of adversarial ball
                adv_steps=config.adv_steps,            # PGD steps (1 = FGSM)
                adv_norm=config.adv_norm,          # "l2" or "linf"
                init_noise_std=config.noise_std
            )})
    elif config.adv_train_method == "Vanilla" or config.adv_train_method == "Certified":
        reft_config = ReftConfig(representations={
            "layer": config.target_layer, "component": "block_output",
            "intervention": LoreftIntervention(
                embed_dim=model.config.hidden_size,
                low_rank_dimension=config.rank,
                dropout=config.drop_out,
            )})

        print("Using Vanilla ReFT")
    else:
        raise ValueError(f"Invalid adv_train_method: {config.adv_train_method}")



    reft_model = get_reft_model(model, reft_config)
    reft_model.print_trainable_parameters()

    batch_triggers = [tokenizer.apply_chat_template([ {"role": "user", "content": f"{q}"}], tokenize=False) for q in edit_questions]
    batch_sequences = [f"{a}" for a in edit_answers]
    batch_rephrase_questions = [tokenizer.apply_chat_template([{"role": "user", "content": f"{q}"}], tokenize=False) for q in edit_rephrase_questions]
    
    cluster_indices_path = config.cluster_indices_path

    print(f"Loading cluster indices from {cluster_indices_path}")
    with open(cluster_indices_path, 'r') as f:
        cluster_data = json.load(f)
    
    ordered_indices = []
    cluster_info = []
    if config.num_samples is not None:
        cluster_data = dict(islice(cluster_data.items(), config.num_samples))
    
    for cluster_id, indices in cluster_data.items():
        
        cluster_info.append({"cluster_id": cluster_id, "start_idx": len(ordered_indices), "size": len(indices)})
        ordered_indices.extend(indices)
    
    print(f"Reordering dataset based on {len(cluster_data)} clusters")
    print(f"Total samples to reorder: {len(ordered_indices)}")
    
  
    batch_triggers = [batch_triggers[i] for i in ordered_indices]
    batch_sequences = [batch_sequences[i] for i in ordered_indices]
    batch_rephrase_questions = [batch_rephrase_questions[i] for i in ordered_indices]


    data_module = make_last_position_supervised_data_module(
        tokenizer, model, batch_triggers, batch_sequences)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    # train
    training_args = transformers.TrainingArguments(
        num_train_epochs=config.epochs, 
        output_dir="./tmp", 
        learning_rate=config.learning_rate, 
        report_to=[], 
        per_device_train_batch_size=config.batch_size,
        logging_steps=100,
        logging_strategy="steps",
        save_strategy="no")

    if config.record:
        wandb.config.update({
            "learning_rate": training_args.learning_rate,
            "noise_std": config.noise_std,
            "drop_out": config.drop_out,
            "lambda_consistency": config.lambda_consistency,
            "method": config.adv_train_method,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": training_args.learning_rate,
            "rank": config.rank,
            "cluster_method": config.cluster_method,
            "cluster_indices_path": cluster_indices_path
        })
        if config.adv_train_method == "Adv_Explicit":
            wandb.config.update({
                "adv_epsilon": config.adv_epsilon,
                "adv_steps": config.adv_steps,
                "adv_norm": config.adv_norm
            })
    act_fn = None
    if "llama" in config.model_name.lower():
        if config.adv_train_method == "Adv_Explicit" or config.adv_train_method == "Explicit": 
            trainer = ReftTrainerAdv(
                lambda_consistency=config.lambda_consistency,
                model=reft_model, tokenizer=tokenizer,
                args=training_args, **data_module)
        elif config.adv_train_method == "Implicit":
            trainer = ReftTrainerImplicit(
                lambda_consistency=config.lambda_consistency,
                model=reft_model, tokenizer=tokenizer,
                args=training_args, **data_module)
        elif config.adv_train_method == "Certified":
            print("Using Certified ReFT")
            trainer = ReftTrainerCertified(
                lambda_consistency=config.lambda_consistency,
                model=reft_model, tokenizer=tokenizer,
                args=training_args, **data_module)
        elif config.adv_train_method == "Vanilla":
            trainer = ReftTrainerForCausalLM(
                model=reft_model, tokenizer=tokenizer,
                args=training_args, **data_module)
            print("Using Vanilla ReFT")
    elif "qwen3" in config.model_name.lower():
        if config.adv_train_method == "Explicit":
            print("Using Explicit ReFT")
            reft_config = ReftConfig(representations={
               "layer": config.target_layer, "component":  f"model.layers[{config.target_layer}].output",
                "intervention": LoreftIntervention_Explicit(
                    embed_dim=model.config.hidden_size,
                    low_rank_dimension=config.rank,
                    act_fn=act_fn,
                    dropout_rate=config.drop_out,
                    init_noise_std=config.noise_std
                )})
    else:
        raise ValueError(f"Invalid adv_train_method: {config.adv_train_method}")

    print("Starting cluster-based batch training...")
    print(f"Training on {len(cluster_info)} clusters")
    print("=" * 50)

    all_results = []
    rouge_scores = []
    rephrased_rouge_scores = []
    
    for cluster_idx, cluster in enumerate(cluster_info):
        cluster_id = cluster["cluster_id"]
        start_idx = cluster["start_idx"]
        cluster_size = cluster["size"]
        end_idx = start_idx + cluster_size
        
        print(f"\n{'='*20} Training on Cluster {cluster_id} ({cluster_idx + 1}/{len(cluster_info)}) {'='*20}")
        print(f"Cluster size: {cluster_size} samples")
        
       
        print(f"Reinitializing REFT module for cluster {cluster_id}...")
        reft_model = reinit_intervention_weights(reft_model)
        
        trigger = batch_triggers[start_idx:end_idx]
        sequence = batch_sequences[start_idx:end_idx]
        rephrase = batch_rephrase_questions[start_idx:end_idx]
        
       
        cluster_data_module = make_last_position_supervised_data_module(
            tokenizer, model, trigger, sequence)
        
     
        if config.adv_train_method == "Adv_Explicit" or config.adv_train_method == "Explicit": 
            trainer = ReftTrainerAdv(
                lambda_consistency=config.lambda_consistency,
                model=reft_model, tokenizer=tokenizer,
                args=training_args, **cluster_data_module)
        elif config.adv_train_method == "Implicit":
            trainer = ReftTrainerImplicit(
                lambda_consistency=config.lambda_consistency,
                model=reft_model, tokenizer=tokenizer,
                args=training_args, **cluster_data_module)
        elif config.adv_train_method == "Certified":
            trainer = ReftTrainerCertified(
                lambda_consistency=config.lambda_consistency,
                model=reft_model, tokenizer=tokenizer,
                args=training_args, **cluster_data_module)
        elif config.adv_train_method == "Vanilla":
            trainer = ReftTrainerForCausalLM(
                model=reft_model, tokenizer=tokenizer,
                args=training_args, **cluster_data_module)
        else:
            raise ValueError(f"Invalid adv_train_method: {config.adv_train_method}")
      
        print(f"Training REFT model on cluster {cluster_id} with {cluster_size} samples...")
        _ = trainer.train()
        print(f"Evaluating on trained cluster {cluster_id}...")
        
        for index in range(len(trigger)):
            test_prompt_tokens = tokenizer(trigger[index], return_tensors="pt").to(device)
            base_unit_location = test_prompt_tokens["input_ids"].shape[-1] - 1
        
            _, steered_response = reft_model.generate(
                test_prompt_tokens, 
                unit_locations={"sources->base": (None, [[[base_unit_location]]])},
                intervene_on_prompt=True, 
                max_new_tokens=512, 
                do_sample=False, 
                eos_token_id=tokenizer.eos_token_id, 
                early_stopping=True,
                temperature=0.001
            )
            generated_text = tokenizer.decode(steered_response[0][len(test_prompt_tokens["input_ids"][0]):], skip_special_tokens=True)
            
            score = scorer.score(sequence[index], generated_text)
            rouge_score = score["rougeL"].recall
            
            # Test on rephrased question
            rephrase_prompt_tokens = tokenizer(rephrase[index], return_tensors="pt").to(device)
            base_unit_location = rephrase_prompt_tokens["input_ids"].shape[-1] - 1
            _, steered_response = reft_model.generate(
                rephrase_prompt_tokens, 
                unit_locations={"sources->base": (None, [[[base_unit_location]]])},
                intervene_on_prompt=True, 
                max_new_tokens=512, 
                do_sample=False, 
                eos_token_id=tokenizer.eos_token_id, 
                early_stopping=True,
                temperature=0.001
            )
            generated_rephrased_text = tokenizer.decode(steered_response[0][len(rephrase_prompt_tokens["input_ids"][0]):], skip_special_tokens=True)
            rephrased_score = scorer.score(sequence[index], generated_rephrased_text)
            rephrased_rouge_score = rephrased_score["rougeL"].recall
            
            global_idx = start_idx + index
            result = {
                "data_idx": global_idx,
                "cluster_id": cluster_id,
                "cluster_local_idx": index,
                "question": trigger[index], 
                "reference": sequence[index], 
                "generated_text": generated_text, 
                "rouge_score": rouge_score, 
                "rephrased_question": rephrase[index], 
                "rephrased_text": generated_rephrased_text, 
                "rephrased_rouge_score": rephrased_rouge_score
            }
            all_results.append(result)
            rouge_scores.append(rouge_score)
            rephrased_rouge_scores.append(rephrased_rouge_score)
        
            print(f"Sample {index+1}/{cluster_size} in Cluster {cluster_id} (Global Index {global_idx}) Results:")
            print(f"  Original Rouge-L: {rouge_score:.3f}")
            print(f"  Rephrased Rouge-L: {rephrased_rouge_score:.3f}")
            print(f"  Generated: {generated_text[:100]}...")
            print(f"  Rephrased: {generated_rephrased_text[:100]}...")

            if config.record:
                wandb.log({
                    "data_idx": global_idx,
                    "cluster_id": cluster_id,
                    "cluster_local_idx": index,
                    "rouge_score": rouge_score,
                    "rephrased_rouge_score": rephrased_rouge_score,
                    "generated_text": generated_text,
                    "rephrased_text": generated_rephrased_text
                })
        
        cluster_rouge_scores = rouge_scores[-cluster_size:]
        cluster_rephrased_scores = rephrased_rouge_scores[-cluster_size:]
        avg_cluster_rouge = sum(cluster_rouge_scores) / len(cluster_rouge_scores)
        avg_cluster_rephrased = sum(cluster_rephrased_scores) / len(cluster_rephrased_scores)
        
        print(f"\nCluster {cluster_id} Summary:")
        print(f"  Average Original Rouge-L: {avg_cluster_rouge:.3f}")
        print(f"  Average Rephrased Rouge-L: {avg_cluster_rephrased:.3f}")
        
        if config.record:
            wandb.log({
                "cluster_id": cluster_id,
                "cluster_avg_rouge": avg_cluster_rouge,
                "cluster_avg_rephrased_rouge": avg_cluster_rephrased,
                "cluster_size": cluster_size
            })
        weights_store="./Stored_weights"
        if config.save_weights_dir is not None:
            weights_base_dir = config.save_weights_dir
            weights_store = os.path.join(weights_store, weights_base_dir)
            os.makedirs(weights_store, exist_ok=True)
            
            # Create directory structure: save_weights_dir/adapter_{data_idx}/
            adapter_dir = os.path.join(weights_store, f"adapter_{cluster_idx}")
            os.makedirs(adapter_dir, exist_ok=True)
            
            # Save only the intervention-specific weights
            for module_key, intervention in reft_model.interventions.items():
                if hasattr(intervention, 'state_dict'):
                    adapter_weights = intervention.state_dict()
                    intervention_path = os.path.join(adapter_dir, f"intervention_{module_key}.pt")
                    torch.save(adapter_weights, intervention_path)
                    print(f"  Saved intervention weights to: {intervention_path}")

    

    print("\n" + "="*50)
    print("Cluster-based batch training completed!")

    # Calculate and print overall statistics
    original_scores = [r["rouge_score"] for r in all_results]
    rephrased_scores = [r["rephrased_rouge_score"] for r in all_results]

    print(f"Overall Statistics:")
    print(f"  Total clusters processed: {len(cluster_info)}")
    print(f"  Total data points trained: {len(all_results)}")
    print(f"  Average Original Rouge-L: {sum(original_scores)/len(original_scores):.3f}")
    print(f"  Average Rephrased Rouge-L: {sum(rephrased_scores)/len(rephrased_scores):.3f}")
    
    if config.record:
        wandb.log({
            "total_clusters": len(cluster_info),
            "total_data_points": len(all_results),
            "final_avg_original_rouge": sum(original_scores)/len(original_scores),
            "final_avg_rephrased_rouge": sum(rephrased_scores)/len(rephrased_scores),
        })

    print(f"Saving results to reft_results/reft_results_cluster_batched.jsonl...")
    os.makedirs(f"reft_results/{config.output_dir}", exist_ok=True)
    with open(f"reft_results/{config.output_dir}/results.jsonl", "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")
    
    print(f"Saving cluster information to {config.cluster_indices_path}...")
  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--record", type=bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="reft_adv_batch")
    parser.add_argument("--dataset", type=str, default="unke_v3")
    parser.add_argument("--noise_std", type=float, default=0.001)
    parser.add_argument("--drop_out", type=float, default=0.05)
    parser.add_argument("--adv_train_method", type=str, default="Adv")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--lambda_consistency", type=float, default=0.001)
    parser.add_argument("--adv_epsilon", type=float, default=0.001)
    parser.add_argument("--adv_steps", type=int, default=3)
    parser.add_argument("--adv_norm", type=str, default="l2")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="reft_results")
    parser.add_argument("--cluster_method", type=str, default="kmeans")
    parser.add_argument("--cluster_indices_path", type=str, default="./cluster_index/unke/unke_v3_3_hac_maxsize8.json")
    parser.add_argument("--save_weights_dir", type=str, default=None)
    
    args = parser.parse_args()
    config = get_config(args)
    Reft_train(config)