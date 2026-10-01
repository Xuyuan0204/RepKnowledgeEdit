from ast import Try
import copy, json, random, re
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import pandas as pd
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
from src.dataset.unke import UnkeForDirectOpt
import wandb
from REFT_module import LoreftIntervention_Implicit, LoreftIntervention_Explicit, LoreftIntervention_Adv_Explicit
from REFT_trainer import ReftTrainerImplicit, ReftTrainerAdv
from utils import reinit_intervention_weights
import argparse

@dataclass
class ReftHyperparameters:
    num_samples: int = 10  # Will be set to full dataset size
    rank: int = 4  # low_rank_dimension
    target_layer: int = 15
    batch_size: int = 10
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    record: bool = False
    wandb_project: str = "reft_batch"
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


def get_config(args):
    config = ReftHyperparameters()
    config.rank = args.rank
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
    elif config.dataset == "tofu":
        edit_data = json.load(open("datasets/tofu/tofu_last_400_edit_data.json"))
        edit_dataset = Dataset.from_dict(edit_data)



    print(f"Training on full dataset with {config.num_samples} samples")

    data_examples = edit_dataset

    data_examples = data_examples.select(range(300))
    edit_questions = [sample["question"] for sample in data_examples]
    edit_answers = [sample["answer"] for sample in data_examples]
    edit_rephrase_questions = [sample["para_question"] for sample in data_examples]

    class TrainingDataEvaluationCallback(TrainerCallback):
        def __init__(self, reft_model, tokenizer, batch_triggers, batch_sequences, batch_rephrase_questions, eval_steps=500):
            self.reft_model = reft_model
            self.tokenizer = tokenizer
            self.batch_triggers = batch_triggers
            self.batch_sequences = batch_sequences
            self.batch_rephrase_questions = batch_rephrase_questions
            self.eval_steps = eval_steps
            self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % self.eval_steps == 0 and state.global_step > 0:
                print(f"\n--- Evaluating at step {state.global_step} ---")
                self.evaluate_on_training_data(state.global_step)
        
        def evaluate_on_training_data(self, step):
            self.reft_model.eval()
            total_rouge = 0.0
            total_rephrased_rouge = 0.0
            
            #sample 20 data points
            sample_indices = random.sample(range(len(self.batch_triggers)), config.num_samples)
        
            results = []
            with torch.no_grad():
                for i, (trigger, target) in enumerate(zip(self.batch_triggers, self.batch_sequences)):
                    if i not in sample_indices:
                        continue
                    test_prompt_tokens = self.tokenizer(trigger, return_tensors="pt").to(device)
                    
                    base_unit_location = test_prompt_tokens["input_ids"].shape[-1] - 1
                    
                    _, steered_responses = self.reft_model.generate(
                        test_prompt_tokens, 
                        unit_locations={"sources->base": (None, [[[base_unit_location]]])},
                        intervene_on_prompt=True, 
                        max_new_tokens=512,  # Same as final test
                        do_sample=False, 
                        eos_token_id=self.tokenizer.eos_token_id, 
                        early_stopping=True,
                    )

                    

                    generated_text = self.tokenizer.decode(steered_responses[0][len(test_prompt_tokens["input_ids"][0]):], skip_special_tokens=True)

                    score = self.scorer.score(target, generated_text)
                    rouge_score = score["rougeL"].recall
                    total_rouge += rouge_score
                    
                    

                    rephrase_prompt_tokens = self.tokenizer(self.batch_rephrase_questions[i], return_tensors="pt").to(device)
                    base_unit_location = rephrase_prompt_tokens["input_ids"].shape[-1] - 1
                    _, steered_response = self.reft_model.generate(
                        rephrase_prompt_tokens, 
                        unit_locations={"sources->base": (None, [[[base_unit_location]]])},
                        intervene_on_prompt=True, 
                        max_new_tokens=512, 
                        do_sample=False, 
                        eos_token_id=self.tokenizer.eos_token_id, 
                        early_stopping=True
                    )
                    generated_rephrased_text = self.tokenizer.decode(steered_response[0][len(rephrase_prompt_tokens["input_ids"][0]):], skip_special_tokens=True)
                    rephrased_rouge_score = self.scorer.score(target, generated_rephrased_text)
                    rephrased_rouge_score = rephrased_rouge_score["rougeL"].recall
                    total_rephrased_rouge += rephrased_rouge_score

                    print("==="*50)
                    print(f"Step {step} - Item {i}")
                    print(f"Generated text: {generated_text}")
                    print(f"Rouge-L score: {rouge_score:.3f}")
                    print(f"Rephrased question: {self.batch_rephrase_questions[i]}")
                    print(f"Rephrased text: {generated_rephrased_text}")
                    print(f"Rephrased Rouge-L score: {rephrased_rouge_score:.3f}")
                    print("-" * 50)
                    results.append({"question": trigger, "reference": target, "generated_text": generated_text, "rouge_score": rouge_score, "rephrased_text": generated_rephrased_text, "rephrased_rouge_score": rephrased_rouge_score})
                    
                    print("-" * 50)

            avg_rouge = total_rouge / len(sample_indices)
            avg_rephrased_rouge = total_rephrased_rouge / len(sample_indices)
            if config.record:
                wandb.log({"rouge_score": avg_rouge, "rephrased_rouge_score": avg_rephrased_rouge, "step": step})
            print(f"Step {step} - Average Rouge-L: {avg_rouge:.3f}")
            print(f"Step {step} - Average Rephrased Rouge-L: {avg_rephrased_rouge:.3f}")
            
            
            self.reft_model.train()


  

    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.bfloat16, device_map=device,attn_implementation="flash_attention_2",low_cpu_mem_usage=True,)


    if config.record:
        wandb.init(project=config.wandb_project)


    tokenizer.pad_token = tokenizer.unk_token

    # get reft model
    if config.adv_train_method == "Explicit":
        print("Using Explicit ReFT")
        act_fn = None
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

        print("Using Vanilla ReFT")
    else:
        raise ValueError(f"Invalid adv_train_method: {config.adv_train_method}")



    reft_model = get_reft_model(model, reft_config)
    reft_model.print_trainable_parameters()

    batch_triggers = [tokenizer.apply_chat_template([ {"role": "user", "content": f"{q}"}], tokenize=False) for q in edit_questions]
    batch_sequences = [f"{a}" for a in edit_answers]
    batch_rephrase_questions = [tokenizer.apply_chat_template([{"role": "user", "content": f"{q}"}], tokenize=False) for q in edit_rephrase_questions]
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

    print("Starting batch training on all data points...")
    print("=" * 50)

    all_results = []
    rouge_scores = []
    rephrased_rouge_scores = []
    for data_idx in range(0,len(batch_triggers),config.num_samples):
        print(f"\n{'='*20} Training on Data Point {data_idx + 1}/{len(batch_triggers)} {'='*20}")
        
        print(f"Reinitializing REFT module for data point {data_idx + 1}...")
        reft_model = reinit_intervention_weights(reft_model)
        
        trigger = batch_triggers[data_idx:data_idx+config.num_samples]
        sequence = batch_sequences[data_idx:data_idx+config.num_samples]
        rephrase = batch_rephrase_questions[data_idx:data_idx+config.num_samples]
        
        single_data_module = make_last_position_supervised_data_module(
            tokenizer, model, trigger, sequence)
        
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
        
        print(f"Training REFT model on data point {data_idx + 1}...")
        _ = trainer.train()
        
        print(f"Evaluating on trained data point {data_idx + 1}...")
        
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
            
            result = {
                "data_idx": index,
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
        
            print(f"Data Point {index} Results:")
            print(f"  Original Rouge-L: {rouge_score:.3f}")
            print(f"  Rephrased Rouge-L: {rephrased_rouge_score:.3f}")
            print(f"  Generated: {generated_text[:100]}...")
            print(f"  Rephrased: {generated_rephrased_text[:100]}...")

            if config.record:
                wandb.log({
                    "data_idx": index,
                    "rouge_score": rouge_score,
                    "rephrased_rouge_score": rephrased_rouge_score,
                    "generated_text": generated_text,
                    "rephrased_text": generated_rephrased_text
                })
    

    print("\n" + "="*50)
    print("Batch training completed!")

    original_scores = [r["rouge_score"] for r in all_results]
    rephrased_scores = [r["rephrased_rouge_score"] for r in all_results]

    print(f"Overall Statistics:")
    print(f"  Average Original Rouge-L: {sum(original_scores)/len(original_scores):.3f}")
    print(f"  Average Rephrased Rouge-L: {sum(rephrased_scores)/len(rephrased_scores):.3f}")
    print(f"  Total data points trained: {len(all_results)}")
    if config.record:
        wandb.log({
            "original_scores": sum(original_scores)/len(original_scores),
            "rephrased_scores": sum(rephrased_scores)/len(rephrased_scores),
            "total_data_points": len(all_results),
        })

    # Save all results
    print(f"Saving results to reft_results/reft_results_batched.jsonl...")
    os.makedirs("reft_results", exist_ok=True)
    with open("reft_results/reft_results_batched.jsonl", "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--record", type=bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="reft_adv_batch")
    parser.add_argument("--dataset", type=str, default="unke")
    parser.add_argument("--noise_std", type=float, default=0.02)
    parser.add_argument("--drop_out", type=float, default=0.05)
    parser.add_argument("--adv_train_method", type=str, default="Adv")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--lambda_consistency", type=float, default=0.001)
    parser.add_argument("--adv_epsilon", type=float, default=0.001)
    parser.add_argument("--adv_steps", type=int, default=3)
    parser.add_argument("--adv_norm", type=str, default="l2")
    parser.add_argument("--rank", type=int, default=4)
    args = parser.parse_args()
    config = get_config(args)
    Reft_train(config)