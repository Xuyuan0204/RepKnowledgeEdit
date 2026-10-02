from ast import Try
from asyncio import Condition
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
import wandb
from utils import load_intervention_weights_consreft, load_intervention_weights_loreft
from src.dataset.unke import UnkeForDirectOpt
import argparse

@dataclass
class ReftHyperparameters:
    num_samples: int = 10
    rank: int = 4  # low_rank_dimension
    target_layer: int = 13
    batch_size: int = 4
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model_path: str = "./reft_model_loreft/Llama-3.1-8B-Instruct/intervenable_model"  # Path to the trained intervention model
    dataset: str = "unke"
    test_rephrase: bool = True

    

def evaluate_rep(config):
    config = ReftHyperparameters()

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config.dataset in ("unke", "unke_v3"):
        edit_dataset = UnkeForDirectOpt().get_dataset_v3()

    data_samples = edit_dataset.select(range(config.num_samples))
    data_questions = [sample["question"] for sample in data_samples]
    data_answers = [sample["answer"] for sample in data_samples]
  
    if config.test_rephrase:
        data_rephrase_questions = [sample["para_question"] for sample in data_samples]
        
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name, device_map=device)
    model_max_length = 2048
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name, model_max_length=model_max_length, 
        padding_side="right", use_fast=True)
    tokenizer.pad_token = tokenizer.unk_token

    print(f"Loading pre-trained ReFT model from {config.model_path}...")
    reft_config = ReftConfig(representations={
        "layer": config.target_layer, "component": "block_output",
        "intervention": LoreftIntervention(
        embed_dim=model.config.hidden_size,
        low_rank_dimension=config.rank)})

    reft_model = get_reft_model(model, reft_config)
    if reft_config.to_dict()['intervention_types'][0].__name__ == 'LoreftIntervention':
        load_intervention_weights_loreft(reft_model, config.model_path, device,reft_config)
    else:
        load_intervention_weights_consreft(reft_model, config.model_path, device)

    reft_model.print_trainable_parameters()
    print("ReFT model loading completed!")

    batch_triggers = [tokenizer.apply_chat_template([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": f"{q}"}], tokenize=False) for q in data_questions]
    batch_sequences = [f"{a}" for a in data_answers]

    if config.test_rephrase:
        batch_rephrase_triggers = [tokenizer.apply_chat_template([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": f"{q}"}], tokenize=False) for q in data_rephrase_questions]
        batch_rephrase_sequences = [f"{a}" for a in data_answers]

    data_module  = make_last_position_supervised_data_module(
        tokenizer, model, batch_triggers, batch_sequences)

    if config.test_rephrase:
        data_module_rephrase = make_last_position_supervised_data_module(
            tokenizer, model, batch_rephrase_triggers, batch_rephrase_sequences)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))


    results = []
    for item in range(len(batch_triggers)):
        
        
        test_prompt_tokens = tokenizer(batch_triggers[item], return_tensors="pt").to(device)

        base_unit_location = test_prompt_tokens["input_ids"].shape[-1] - 1

        print("==="*30)
        print(f"Input prompt: '{batch_triggers[item]}'")
        
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
        print("\nGenerated output:")
        print(generated_text)
        
        score = scorer.score(batch_sequences[item], generated_text)
        rouge_score = score["rougeL"].recall
        
        res_dict = {"question": batch_triggers[item], "reference": batch_sequences[item], "generated_text": generated_text, "rouge_score": rouge_score}

        if config.test_rephrase:
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
            score_rephrase = scorer.score(batch_rephrase_sequences[item], generated_text_rephrase)
         
            rouge_score_rephrase = score_rephrase["rougeL"].recall
            res_dict["rephrase_generated_text"] = generated_text_rephrase
            res_dict["rephrase_rouge_score"] = rouge_score_rephrase

        results.append(res_dict)

        print(f"Rouge-L score: {rouge_score:.3f}")
        if config.test_rephrase:
            print(f"Rouge-L score for paraphrase: {rouge_score_rephrase:.3f}")

    with open(f"reft_results_{config.dataset}.jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="unke_v3")
    parser.add_argument("--model_path", type=str, default="./reft_model/intervenable_model")
    parser.add_argument("--target_layer", type=int, default=15)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=40)
    parser.add_argument("--test_rephrase", type=bool, default=False)
    args = parser.parse_args()

    config = ReftHyperparameters()
    config.dataset = args.dataset
    config.model_path = args.model_path
    config.target_layer = args.target_layer
    config.rank = args.rank
    config.num_samples = args.num_samples
    config.model_name = args.model_name

    evaluate_rep(config)