from ast import Try
import copy, json, random, re
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_line, theme_minimal
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({'font.size': 20, 'font.family': 'Sans'})
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
import numpy as np
import wandb
from REFT_module import LoreftIntervention_Implicit, LoreftIntervention_Explicit, LoreftIntervention_Adv_Explicit
from REFT_trainer import ReftTrainerImplicit, ReftTrainerAdv
from utils import reinit_intervention_weights
import argparse
from itertools import islice
import wandb
from utils import load_intervention_weights_loreft
from src.dataset.unke import UnkeForDirectOpt
from src.dataset.anyedit import AnyEditForDirectOpt
import argparse
from sentence_transformers import SentenceTransformer, util


bert_score_model = SentenceTransformer("all-MiniLM-L6-v2")

@dataclass
class ReftHyperparameters:
    num_samples: int = None  # Will be set to full dataset size
    rank: int = 8  # low_rank_dimension
    target_layer: int = 15
    batch_size: int = 1
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    record: bool = False
    wandb_project: str = "reft_single_datapoint"
    dataset: str = "unke"
    noise_std: float = 0.005  # Bottleneck noise standard deviation for hidden representations
    drop_out: float = 0.05
    adv_train_method: str = "Adv_Explicit"
    epochs: int = 1000
    learning_rate: float = 2e-3
    adv_epsilon: float = 0.001
    test_rephrase: bool = True
    adv_steps: int = 3
    adv_norm: str = "l2"
    lambda_consistency: float = 0.01
    output_dir: str = "reft_results_single"
    save_weights_dir: str = "adapter_weights"
    activation_path: str = "./activation/unke/llama_3_8b_layer15_no_answer_last_original.pt"
    original_query_activation_path: str = "./activation/unke/llama_3_8b_layer15_no_answer_last_original.pt"
    rephrased_query_activation_path: str = "./activation/unke/llama_3_8b_layer15_no_answer_last_rephrased.pt"
    

def load_activation_embeddings(activation_path):
    """Load and preprocess activation embeddings from .pt file"""
    activations = torch.load(activation_path, map_location='cpu')
    
    # Handle list of tensors case
    if isinstance(activations, list):
        activations = torch.stack(activations)
    
    # Squeeze if necessary (shape [N, 1, D] -> [N, D])
    if len(activations.shape) == 3 and activations.shape[1] == 1:
        activations = activations.squeeze(1)
    
    return activations

def compute_similarity_with_activations(query_embedding, activation_embeddings):
    """
    Compute cosine similarity between query embedding and activation embeddings.
    Returns index of most similar activation.
    """
    # Normalize embeddings for cosine similarity
    query_norm = F.normalize(query_embedding.unsqueeze(0), p=2, dim=1)
    activation_norm = F.normalize(activation_embeddings, p=2, dim=1)
    
    # Compute similarities
    similarities = torch.mm(query_norm, activation_norm.T).squeeze(0)
    
    # Return index of highest similarity
    best_idx = similarities.argmax().item()
    best_score = similarities[best_idx].item()
    
    return best_idx, best_score

def load_reft_adapter(reft_model, adapter_weights_dir, adapter_idx, device, config):
    """Load specific adapter weights into the ReFT model"""
    adapter_path = os.path.join(adapter_weights_dir, f"adapter_{adapter_idx}")

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    if "qwen" in config.model_name.lower():
        component = f"model.layers[{config.target_layer}].output"
    else:
        component = "block_output"

    reft_config = ReftConfig(representations={
        "layer": config.target_layer, "component": component,
        "intervention": LoreftIntervention(
        embed_dim=reft_model.model.config.hidden_size,
        low_rank_dimension=config.rank)})

    load_intervention_weights_loreft(reft_model, adapter_path, device, reft_config)

    return reft_model

def get_config(args):
    config = ReftHyperparameters()
    config.target_layer = args.target_layer
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
    config.save_weights_dir = args.save_weights_dir
    config.target_layer = args.target_layer
    config.activation_path = args.activation_path
    config.original_query_activation_path = args.original_query_activation_path
    config.rephrased_query_activation_path = args.rephrased_query_activation_path
    config.model_name = args.model_name
    config.attn_impl = args.attn_impl
    return config


scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


def Reft_train(config):
    weights_store="./Stored_weights"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name, model_max_length=512, 
        padding_side="right", use_fast=True
        
        )

    if config.dataset == "unke":
        edit_dataset = UnkeForDirectOpt().get_dataset()
    elif config.dataset == "unke_v3":
        edit_dataset = UnkeForDirectOpt().get_dataset_v3()
    elif config.dataset == "anyedit":
        edit_dataset = AnyEditForDirectOpt().get_dataset()

    # Set num_samples to full dataset size for batch training
    if args.num_samples is not None:
        config.num_samples = args.num_samples
    else:
        config.num_samples = len(edit_dataset)
    sample_indices = range(config.num_samples)
    print(f"Training on dataset with {config.num_samples} samples")

    data_examples = edit_dataset.select(sample_indices)


    edit_questions = [sample["question"] for sample in data_examples]
    edit_answers = [sample["answer"] for sample in data_examples]
    edit_rephrase_questions = [sample["para_question"] for sample in data_examples]


    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.bfloat16, device_map=device,attn_implementation=config.attn_impl,low_cpu_mem_usage=True,)


    if config.record:
        wandb.init(project=config.wandb_project)



    tokenizer.pad_token = tokenizer.unk_token

    # get reft model
    if "qwen" in config.model_name.lower():
        component = f"model.layers[{config.target_layer}].output"
    else:
        component = "block_output"

    if config.adv_train_method == "Explicit":
        print("Using Explicit ReFT")
        act_fn = None
        reft_config = ReftConfig(representations={
            "layer": config.target_layer, "component": component,
            "intervention": LoreftIntervention_Explicit(
                embed_dim=model.config.hidden_size,
                low_rank_dimension=config.rank,
                act_fn=act_fn,
                dropout_rate=config.drop_out,
                init_noise_std=config.noise_std
            )})
    elif config.adv_train_method == "Implicit":
        print("Using Implicit ReFT")
        reft_config = ReftConfig(representations={
            "layer": config.target_layer, "component": component,
            "intervention": LoreftIntervention_Implicit(
                embed_dim=model.config.hidden_size,
                low_rank_dimension=config.rank,
                dropout_rate=config.drop_out,
                init_noise_std=config.noise_std
            )})
    elif config.adv_train_method == "Adv_Explicit":
        print("Using Adversarial Explicit ReFT")
        reft_config = ReftConfig(representations={
            "layer": config.target_layer, "component": component,
            "intervention": LoreftIntervention_Adv_Explicit(
                embed_dim=model.config.hidden_size,
                low_rank_dimension=config.rank,
                dropout_rate=config.drop_out,
                adv_epsilon=config.adv_epsilon,
                adv_steps=config.adv_steps,
                adv_norm=config.adv_norm,
                init_noise_std=config.noise_std
            )})
    elif config.adv_train_method == "Vanilla":
        reft_config = ReftConfig(representations={
            "layer": config.target_layer, "component": component,
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

    # get training data to train our intervention to remember the following sequence

    batch_triggers = [tokenizer.apply_chat_template([ {"role": "user", "content": f"{q}"}], tokenize=False) for q in edit_questions]
    # batch_triggers = edit_questions
    batch_sequences = [f"{a}" for a in edit_answers]
    
    if config.dataset in ("unke", "unke_v3", "anyedit"):
        batch_rephrase_questions = [tokenizer.apply_chat_template([{"role": "user", "content": f"{q}"}], tokenize=False) for q in edit_rephrase_questions]
    else:
        batch_rephrase_questions = []
        for i in range(len(edit_rephrase_questions)):
            
            single_data=[tokenizer.apply_chat_template([{"role": "user", "content": f"{q}"}], tokenize=False) for q in edit_rephrase_questions[i]]
            batch_rephrase_questions.append(single_data)
    data_indices = list(range(len(batch_triggers)))



    batch_triggers = [batch_triggers[i] for i in data_indices]
    batch_sequences = [batch_sequences[i] for i in data_indices]
    batch_rephrase_questions = [batch_rephrase_questions[i] for i in data_indices]


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
        logging_steps=200,
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
            "dataset": config.dataset,
            "target_layer": config.target_layer

        })
        if config.adv_train_method == "Adv_Explicit":
            wandb.config.update({
                "adv_epsilon": config.adv_epsilon,
                "adv_steps": config.adv_steps,
                "adv_norm": config.adv_norm
            })

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
    else:
        trainer = ReftTrainerForCausalLM(
            model=reft_model, tokenizer=tokenizer,
            args=training_args, **data_module)
        print("Using Vanilla ReFT")

    # Batch training: process each data point individually with REFT module reinitialization
    print("Starting batch training on all data points...")
    print("=" * 50)

    all_results = []
    weights_base_dir = config.save_weights_dir
    weights_store = os.path.join(weights_store, weights_base_dir)
    os.makedirs(weights_store, exist_ok=True)

    for data_idx in range(len(batch_triggers)):
        print(f"\n{'='*20} Training on Data Point {data_idx + 1}/{len(batch_triggers)} {'='*20}")
        
        # Reinitialize REFT module before training on each data point
        print(f"Reinitializing REFT module for data point {data_idx + 1}...")
        reft_model = reinit_intervention_weights(reft_model)
        
        # Prepare single data point for training
        single_trigger = [batch_triggers[data_idx]]
        single_sequence = [batch_sequences[data_idx]]
        
        # Create data module for single data point
        single_data_module = make_last_position_supervised_data_module(
            tokenizer, model, single_trigger, single_sequence)
        
        # Create trainer for this data point
        if config.adv_train_method == "Adv_Explicit" or config.adv_train_method == "Explicit": 
            trainer = ReftTrainerAdv(
                lambda_consistency=config.lambda_consistency,
                model=reft_model, tokenizer=tokenizer,
                args=training_args, **single_data_module)
        elif config.adv_train_method == "Implicit":
            trainer = ReftTrainerImplicit(
                lambda_consistency=config.lambda_consistency,
                model=reft_model, tokenizer=tokenizer,
                args=training_args, **single_data_module)
        else:
            trainer = ReftTrainerForCausalLM(
                model=reft_model, tokenizer=tokenizer,
                args=training_args, **single_data_module)
        
        # Train on single data point
        print(f"Training REFT model on data point {data_idx + 1}...")
        _ = trainer.train()
        
        # Evaluate on the trained data point
        print(f"Evaluating on trained data point {data_idx + 1}...")
        
        
        if config.save_weights_dir is not None:
            adapter_dir = os.path.join(weights_store, f"adapter_{data_idx}")
            os.makedirs(adapter_dir, exist_ok=True)
            
            # Save only the intervention-specific weights
            for module_key, intervention in reft_model.interventions.items():
                if hasattr(intervention, 'state_dict'):
                    adapter_weights = intervention.state_dict()
                    intervention_path = os.path.join(adapter_dir, f"intervention_{module_key}.pt")
                    torch.save(adapter_weights, intervention_path)
                    print(f"  Saved intervention weights to: {intervention_path}")
    del reft_model
    torch.cuda.empty_cache()


def Reft_test(config):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    weights_store="./Stored_weights"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config.dataset == "unke":
        edit_dataset = UnkeForDirectOpt().get_dataset_v3()
    elif config.dataset == "unke_v3":
        edit_dataset = UnkeForDirectOpt().get_dataset_v3()
    elif config.dataset == "anyedit":
        edit_dataset = AnyEditForDirectOpt().get_dataset()
    if config.num_samples is None:
        config.num_samples = len(edit_dataset)

    data_samples = edit_dataset.select(range(config.num_samples))
    data_questions = [sample["question"] for sample in data_samples]
    data_answers = [sample["answer"] for sample in data_samples]
  
    data_rephrase_questions = [sample["para_question"] for sample in data_samples]
        
    # Load activation embeddings for similarity comparison
    print(f"Loading activation embeddings from {config.activation_path}...")
    activation_embeddings = load_activation_embeddings(config.activation_path)[:config.num_samples].to(device)
    print(f"Loaded {activation_embeddings.shape[0]} activation embeddings with dimension {activation_embeddings.shape[1]}")
    
    # Load query embeddings
    print(f"Loading original query embeddings from {config.original_query_activation_path}...")
    original_query_embeddings = load_activation_embeddings(config.original_query_activation_path)[:config.num_samples].to(device)
    print(f"Loaded {original_query_embeddings.shape[0]} original query embeddings")
    
    rephrased_query_embeddings = load_activation_embeddings(config.rephrased_query_activation_path)[:config.num_samples].to(device)
      
    # Validate that we have enough query embeddings for the number of samples
    if original_query_embeddings.shape[0] < config.num_samples:
        raise ValueError(f"Not enough original query embeddings: {original_query_embeddings.shape[0]} < {config.num_samples}")
    
    if config.test_rephrase and rephrased_query_embeddings.shape[0] < config.num_samples:
        raise ValueError(f"Not enough rephrased query embeddings: {rephrased_query_embeddings.shape[0]} < {config.num_samples}")
        

    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name, device_map=device)



    # get tokenizer
    model_max_length = 2048
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name, model_max_length=model_max_length, 
        padding_side="right", use_fast=True)
    tokenizer.pad_token = tokenizer.unk_token

    # Initialize ReFT model structure (without loading specific weights yet)
    print("Initializing ReFT model structure...")
    if "qwen" in config.model_name.lower():
        component = f"model.layers[{config.target_layer}].output"
    else:
        component = "block_output"

    reft_config = ReftConfig(representations={
        "layer": config.target_layer, "component": component,
        "intervention": LoreftIntervention(
        embed_dim=model.config.hidden_size,
        low_rank_dimension=config.rank)})

    reft_model = get_reft_model(model, reft_config)
    reft_model.print_trainable_parameters()
    print("ReFT model initialization completed!")

    batch_triggers = [tokenizer.apply_chat_template([{"role": "user", "content": f"{q}"}], tokenize=False) for q in data_questions[:config.num_samples]]
    batch_sequences = [f"{a}" for a in data_answers]

    batch_rephrase_triggers = [tokenizer.apply_chat_template([{"role": "user", "content": f"{q}"}], tokenize=False) for q in data_rephrase_questions[:config.num_samples]]
    batch_rephrase_sequences = [f"{a}" for a in data_answers]

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))


    results = []
    for data_idx, item in enumerate(range(len(batch_triggers))):
        print("==="*30)
        
        query_embedding = original_query_embeddings[item]
        best_idx, best_score = compute_similarity_with_activations(query_embedding, activation_embeddings)
       
        print(f"Selected adapter {best_idx} (similarity: {best_score:.4f})")
        print(config)
        reft_model = load_reft_adapter(reft_model, os.path.join(weights_store, config.save_weights_dir), best_idx, device, config)
        
        test_prompt_tokens = tokenizer(batch_triggers[item], return_tensors="pt").to(device)
        base_unit_location = test_prompt_tokens["input_ids"].shape[-1] - 1
        
        _, steered_response = reft_model.generate(
            test_prompt_tokens, 
            unit_locations={"sources->base": (None, [[[base_unit_location]]])},
            intervene_on_prompt=True, 
            max_new_tokens=512, 
            do_sample=False, 
            eos_token_id=tokenizer.eos_token_id, 
            early_stopping=True
        )
        generated_text = tokenizer.decode(steered_response[0][len(test_prompt_tokens["input_ids"][0]):], skip_special_tokens=True)
        print("\nOriginal query generated output:")
        print(generated_text)
        
        score = scorer.score(batch_sequences[item], generated_text)
        rouge_score = score["rougeL"].recall
        ref_embedding = bert_score_model.encode(batch_sequences[item], convert_to_tensor=True)
        gen_embedding = bert_score_model.encode(generated_text, convert_to_tensor=True)
        bert_score = util.cos_sim(ref_embedding, gen_embedding).item()
        
        res_dict = {
            "question": batch_triggers[item], 
            "reference": batch_sequences[item], 
            "generated_text": generated_text, 
            "rouge_score": rouge_score,
            "selected_adapter_idx": best_idx,
            "similarity_score": best_score,
            "Bert_Score": bert_score,
            "Retrieved_adapter": best_idx
        }

        if config.test_rephrase:
            print(f"Rephrased query: '{data_rephrase_questions[item]}'")
            rephrase_query_embedding = rephrased_query_embeddings[item]
            rephrase_best_idx, rephrase_best_score = compute_similarity_with_activations(rephrase_query_embedding, activation_embeddings)
            
            adapter_path = os.path.join(weights_store, config.save_weights_dir)
            print(f"Selected adapter {rephrase_best_idx} for rephrase (similarity: {rephrase_best_score:.4f})")
            reft_model = load_reft_adapter(reft_model, adapter_path, rephrase_best_idx, device, config)
            
            test_prompt_tokens_rephrase = tokenizer(batch_rephrase_triggers[item], return_tensors="pt").to(device)
            base_unit_location_rephrase = test_prompt_tokens_rephrase["input_ids"].shape[-1] - 1
            _, steered_response_rephrase = reft_model.generate(
                test_prompt_tokens_rephrase, 
                unit_locations={"sources->base": (None, [[[base_unit_location_rephrase]]])},
                intervene_on_prompt=True, 
                max_new_tokens=512,     
                do_sample=False, 
                eos_token_id=tokenizer.eos_token_id, 
                early_stopping=True
            )
            generated_text_rephrase = tokenizer.decode(steered_response_rephrase[0][len(test_prompt_tokens_rephrase["input_ids"][0]):], skip_special_tokens=True)
            print("\nRephrased query generated output:")
            print(generated_text_rephrase)
            
            score_rephrase = scorer.score(batch_rephrase_sequences[item], generated_text_rephrase)
            rouge_score_rephrase = score_rephrase["rougeL"].recall
            ref_embedding_rephrase = bert_score_model.encode(batch_rephrase_sequences[item], convert_to_tensor=True)
            gen_embedding_rephrase = bert_score_model.encode(generated_text_rephrase, convert_to_tensor=True)
            bert_score_rephrase = util.cos_sim(ref_embedding_rephrase, gen_embedding_rephrase).item()

            res_dict["rephrase_generated_text"] = generated_text_rephrase
            res_dict["rephrase_rouge_score"] = rouge_score_rephrase
            res_dict["rephrase_selected_adapter_idx"] = rephrase_best_idx
            res_dict["rephrase_similarity_score"] = rephrase_best_score
            res_dict["rephrase_Bert_Score"] = bert_score_rephrase
            res_dict["rephrase_Retrieved_adapter"] = rephrase_best_idx
            
            aligned=True if data_idx == rephrase_best_idx else False
            res_dict["aligned"] = aligned

        results.append(res_dict)
        if config.record:
            wandb.log({
                "data_idx": data_idx,
                "rouge_score": rouge_score,
                "rephrased_rouge_score": rouge_score_rephrase,
                "bert_score": bert_score,
                "rephrased_bert_score": bert_score_rephrase
            })
    
    

    os.makedirs(f"results/{config.dataset}", exist_ok=True)

    with open(f"results/{config.dataset}/{config.save_weights_dir}_{config.num_samples}.jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    print(f"Results saved to results/{config.save_weights_dir}.jsonl")
    return results
    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--record", type=bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="reft_single_datapoint")
    parser.add_argument("--dataset", type=str, default="unke_v3")
    parser.add_argument("--noise_std", type=float, default=0.02)
    parser.add_argument("--drop_out", type=float, default=0.05)
    parser.add_argument("--adv_train_method", type=str, default="Adv_Explicit")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--lambda_consistency", type=float, default=0.001)
    parser.add_argument("--adv_epsilon", type=float, default=0.001)
    parser.add_argument("--adv_steps", type=int, default=3)
    parser.add_argument("--adv_norm", type=str, default="l2")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="reft_results_single")
    parser.add_argument("--save_weights_dir", type=str, default=None)
    parser.add_argument("--target_layer", type=int, default=15)
    parser.add_argument("--test_rephrase", type=bool, default=True)
    parser.add_argument("--activation_path", type=str, default="./activation/unke/llama_3_8b_layer15_no_answer_last_original.pt")
    parser.add_argument("--original_query_activation_path", type=str, default="./activation/unke/llama_3_8b_layer15_no_answer_last_original.pt")
    parser.add_argument("--rephrased_query_activation_path", type=str, default="./activation/unke/llama_3_8b_layer15_no_answer_last_rephrased.pt")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--attn_impl", type=str, default="flash_attention_2", help="flash_attention_2 (default) or sdpa/eager")
    args = parser.parse_args()
    config = get_config(args)
    Reft_train(config)
    Reft_test(config)