from ast import Try
import copy, json, random, re
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import pandas as pd
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_line, theme_minimal
from matplotlib.ticker import MaxNLocator
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
from src.dataset.unke import UnkeForDirectOpt
from src.dataset.anyedit import AnyEditForDirectOpt
import wandb
from REFT_module import LoreftIntervention_Implicit, LoreftIntervention_Explicit, LoreftIntervention_Adv_Explicit
from REFT_trainer import ReftTrainerImplicit, ReftTrainerAdv
from utils import reinit_intervention_weights
import argparse
from itertools import islice

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
    adv_steps: int = 3
    adv_norm: str = "l2"
    lambda_consistency: float = 0.01
    output_dir: str = "reft_results_single"
    save_weights_dir: str = "adapter_weights"  # Directory to save each adapter's weights



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
    config.model_name = args.model_name
    config.attn_impl = args.attn_impl
    return config


scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


def Reft_train(config):

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
    config.num_samples = len(edit_dataset)
    sample_indices = range(len(edit_dataset))
    print(f"Training on full dataset with {config.num_samples} samples")

    data_examples = edit_dataset.select(sample_indices)
    # data_examples = ReplayDataset(data_examples, tokenizer, sample_ratio=3)


    edit_questions = [sample["question"] for sample in data_examples]
    edit_answers = [sample["answer"] for sample in data_examples]
    edit_rephrase_questions = [sample["para_question"] for sample in data_examples]


    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.bfloat16, device_map=device,attn_implementation=config.attn_impl,low_cpu_mem_usage=True,)


    if config.record:
        wandb.init(project=config.wandb_project)



    tokenizer.pad_token = tokenizer.unk_token

    # get reft model
    act_fn = None
    if "llama" in config.model_name.lower():

        if config.adv_train_method == "Explicit":
            print("Using Explicit ReFT")
            
            reft_config = ReftConfig(representations={
                "layer": config.target_layer, "component": "block_output",
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
        elif config.adv_train_method == "Vanilla":
            reft_config = ReftConfig(representations={
                "layer": config.target_layer, "component": "block_output",
                "intervention": LoreftIntervention(
                    embed_dim=model.config.hidden_size,
                    low_rank_dimension=config.rank,
                    dropout=config.drop_out,
                )})
    elif "qwen" in config.model_name.lower():
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

    for data_idx in range(len(batch_triggers)):
        print(f"\n{'='*20} Training on Data Point {data_idx + 1}/{len(batch_triggers)} {'='*20}")
        
        # Reinitialize REFT module before training on each data point
        print(f"Reinitializing REFT module for data point {data_idx + 1}...")
        reft_model = reinit_intervention_weights(reft_model)
        
        # Prepare single data point for training
        single_trigger = [batch_triggers[data_idx]]
        single_sequence = [batch_sequences[data_idx]]
        single_rephrase = [batch_rephrase_questions[data_idx]]
        
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
        
        # Test on original question
        test_prompt_tokens = tokenizer(single_trigger[0], return_tensors="pt").to(device)
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
        
        score = scorer.score(single_sequence[0], generated_text)
        rouge_score = score["rougeL"].recall
        ref_embedding = bert_score_model.encode(single_sequence[0], convert_to_tensor=True)
        gen_embedding = bert_score_model.encode(generated_text, convert_to_tensor=True)
        bert_score = util.cos_sim(ref_embedding, gen_embedding).item()

        rephrased_rouge_scores = []
        bert_score_rephrases = []
        # Test on rephrased question
        
        single_rephrase = single_rephrase[0]
        
        rephrase_prompt_tokens = tokenizer(single_rephrase, return_tensors="pt").to(device)
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
        rephrased_score = scorer.score(single_sequence[0], generated_rephrased_text)
        rephrased_rouge_score = rephrased_score["rougeL"].recall
            
        ref_embedding_rephrase = bert_score_model.encode(single_sequence[0], convert_to_tensor=True)
        gen_embedding_rephrase = bert_score_model.encode(generated_rephrased_text, convert_to_tensor=True)
        bert_score_rephrase = util.cos_sim(ref_embedding_rephrase, gen_embedding_rephrase).item()
        bert_score_rephrases.append(bert_score_rephrase)
        rephrased_rouge_scores.append(rephrased_rouge_score)
            
        print(f"Rephrased question: {single_rephrase[0]}")
        print(f"Rephrased text: {generated_rephrased_text}")
        print(f"Rephrased Rouge-L score: {rephrased_rouge_score:.3f}")
        print(f"Rephrased Bert Score: {bert_score_rephrase:.3f}")
        print("-" * 50)


        
        # Store results
        result = {
            "data_idx": data_idx,
            "question": single_trigger[0], 
            "reference": single_sequence[0], 
            "generated_text": generated_text, 
            "rouge_score": rouge_score, 
            "bert_score": bert_score,
            "rephrased_question": single_rephrase[0], 
            "rephrased_text": generated_rephrased_text, 
            "rephrased_rouge_score": rephrased_rouge_score,
            "rephrased_bert_score": bert_score_rephrase
        }
        all_results.append(result)

       
        
        # Log to wandb if recording
        if config.record:
            wandb.log({
                "data_point": data_idx + 1,
                "rouge_score": rouge_score,
                "rephrased_rouge_score": rephrased_rouge_score,
                "bert_score": bert_score,
                "rephrased_bert_score": bert_score_rephrase
            })
        
        print(f"Data Point {data_idx + 1} Results:")
        print(f"  Original Rouge-L: {rouge_score:.3f}")
        print(f"  Rephrased Rouge-L: {rephrased_rouge_score:.3f}")
        print(f"  Original Bert Score: {bert_score:.3f}")
        print(f"  Rephrased Bert Score: {bert_score_rephrase:.3f}")
        print(f"  Generated: {generated_text[:100]}...")
        print(f"  Rephrased: {generated_rephrased_text[:100]}...")
        
        # Save adapter weights for this data point
        weights_store="./Stored_weights"
        if config.save_weights_dir is not None:
            weights_base_dir = config.save_weights_dir
            weights_store = os.path.join(weights_store, weights_base_dir)
            os.makedirs(weights_store, exist_ok=True)
            
            # Create directory structure: save_weights_dir/adapter_{data_idx}/
            adapter_dir = os.path.join(weights_store, f"adapter_{data_idx}")
            os.makedirs(adapter_dir, exist_ok=True)
            
            # Save only the intervention-specific weights
            for module_key, intervention in reft_model.interventions.items():
                if hasattr(intervention, 'state_dict'):
                    adapter_weights = intervention.state_dict()
                    intervention_path = os.path.join(adapter_dir, f"intervention_{module_key}.pt")
                    torch.save(adapter_weights, intervention_path)
                    print(f"  Saved intervention weights to: {intervention_path}")

    print("\n" + "="*50)
    print("Batch training completed!")

    # Calculate and print overall statistics
    original_scores = [r["rouge_score"] for r in all_results]
    rephrased_scores = [r["rephrased_rouge_score"] for r in all_results]

    print(f"Overall Statistics:")
    print(f"  Average Original Rouge-L: {sum(original_scores)/len(original_scores):.3f}")
    print(f"  Average Rephrased Rouge-L: {sum(rephrased_scores)/len(rephrased_scores):.3f}")
    print(f"  Total data points trained: {len(all_results)}")
    original_bert_scores = [r["bert_score"] for r in all_results]
    rephrased_bert_scores = [r["rephrased_bert_score"] for r in all_results]
    print(f"  Average Original Bert Score: {sum(original_bert_scores)/len(original_bert_scores):.3f}")
    print(f"  Average Rephrased Bert Score: {sum(rephrased_bert_scores)/len(rephrased_bert_scores):.3f}")

    if config.record:
        wandb.log({
            "total_data_points": len(all_results),
            "final_avg_original_rouge": sum(original_scores)/len(original_scores),
            "final_avg_rephrased_rouge": sum(rephrased_scores)/len(rephrased_scores),
            "final_avg_original_bert_score": sum(original_bert_scores)/len(original_bert_scores),
            "final_avg_rephrased_bert_score": sum(rephrased_bert_scores)/len(rephrased_bert_scores),
        })

    # Save all results
    print(f"Saving results to reft_results/{config.output_dir}/results.jsonl...")
    os.makedirs(f"reft_results/{config.output_dir}", exist_ok=True)
    with open(f"reft_results/{config.output_dir}/{config.dataset}_results.jsonl", "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")
    print(f"Results saved to reft_results/{config.output_dir}/results.jsonl")

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
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--attn_impl", type=str, default="flash_attention_2", help="flash_attention_2 (default) or sdpa/eager")
    args = parser.parse_args()
    config = get_config(args)
    Reft_train(config)