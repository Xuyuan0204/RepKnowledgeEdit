from ast import Try
from asyncio import Condition
from operator import index
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
import torch.nn.functional as F
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
import torch.nn.functional as F
import wandb
from utils import load_intervention_weights_loreft
from src.dataset.unke import UnkeForDirectOpt
from src.dataset.anyedit import AnyEditForDirectOpt
import argparse
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

bert_score_model = SentenceTransformer("all-MiniLM-L6-v2")

@dataclass
class ReftHyperparameters:
    num_samples: int = 10
    rank: int = 4  # low_rank_dimension
    target_layer: int = 15,
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    dataset: str = "unke"
    test_rephrase: bool = True
    activation_path: str = "./activation/unke/llama_3_8b_layer15_no_answer_last_original.pt"
    adapter_weights_dir: str = "./single_adapter_weights_rank4_implicit"
    original_query_activation_path: str = "./activation/unke/llama_3_8b_layer15_no_answer_last_original.pt"
    rephrased_query_activation_path: str = "./activation/unke/llama_3_8b_layer15_no_answer_last_rephrased.pt"
    save_path: str = "vanilla_unke"
    cluster_indices_path: str = "./cluster_index/unke/unke_v3_3_hac_maxsize8.json"


def load_activation_embeddings(activation_path):
    activations = torch.load(activation_path, map_location='cpu')
    
    if isinstance(activations, list):
        activations = torch.stack(activations)
    
    if len(activations.shape) == 3 and activations.shape[1] == 1:
        activations = activations.squeeze(1)
    
    return activations

def compute_similarity_with_activations(query_embedding, activation_embeddings):
    
    similarities = F.cosine_similarity(query_embedding, activation_embeddings, dim=1)
    
    best_idx = similarities.argmax().item()
    best_score = similarities[best_idx].item()

    return best_idx, best_score

def load_reft_adapter(reft_model, adapter_weights_dir, adapter_idx, device, config):
    adapter_path = os.path.join(adapter_weights_dir, f"adapter_{adapter_idx}")
    
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")
    
    reft_config = ReftConfig(representations={
        "layer": config.target_layer, "component": "block_output",
        "intervention": LoreftIntervention(
        embed_dim=reft_model.model.config.hidden_size,
        low_rank_dimension=config.rank)})
    
    load_intervention_weights_loreft(reft_model, adapter_path, device, reft_config)
    
    return reft_model

def evaluate_rep(config):
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.dataset in ("unke", "unke_v3"):
        edit_dataset = UnkeForDirectOpt().get_dataset_v3()
    elif config.dataset == "anyedit":
        edit_dataset = AnyEditForDirectOpt().get_dataset()
    if config.num_samples is None:
        config.num_samples = len(edit_dataset)

    data_samples = edit_dataset.select(range(config.num_samples))
    data_questions = [sample["question"] for sample in data_samples]
    data_answers = [sample["answer"] for sample in data_samples]
  
    if config.test_rephrase:
        data_rephrase_questions = [sample["para_question"] for sample in data_samples]
        
    
    activation_embeddings = load_activation_embeddings(config.activation_path)[:config.num_samples].to(device)
   
    with open(config.cluster_indices_path, 'r') as f:
        cluster_data = json.load(f)
    print(f"Loaded {len(cluster_data)} cluster indices")
    original_query_embeddings = load_activation_embeddings(config.original_query_activation_path)

    index_embeddings = []
    
    data_idx_to_cluster_id = {}
    for cluster_id, indices in tqdm(cluster_data.items()):
        for data_idx in indices:
            data_idx_to_cluster_id[data_idx] = int(cluster_id)
    
    print(f"Built mapping for {len(data_idx_to_cluster_id)} data indices across {len(cluster_data)} clusters")
    
  



    if config.test_rephrase:
        print(f"Loading rephrased query embeddings from {config.rephrased_query_activation_path}...")
        rephrased_query_embeddings = load_activation_embeddings(config.rephrased_query_activation_path)[:config.num_samples].to(device)
        print(f"Loaded {rephrased_query_embeddings.shape[0]} rephrased query embeddings")
        
    if original_query_embeddings.shape[0] < config.num_samples:
        raise ValueError(f"Not enough original query embeddings: {original_query_embeddings.shape[0]} < {config.num_samples}")
    
    if config.test_rephrase and rephrased_query_embeddings.shape[0] < config.num_samples:
        raise ValueError(f"Not enough rephrased query embeddings: {rephrased_query_embeddings.shape[0]} < {config.num_samples}")
        

    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name, device_map=device)

    original_query_embeddings = original_query_embeddings.to(device)
    rephrased_query_embeddings = rephrased_query_embeddings.to(device)
    activation_embeddings = activation_embeddings.to(device)

    model_max_length = 2048
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name, model_max_length=model_max_length, 
        padding_side="right", use_fast=True)
    tokenizer.pad_token = tokenizer.unk_token

    reft_config = ReftConfig(representations={
        "layer": config.target_layer, "component": "block_output",
        "intervention": LoreftIntervention(
        embed_dim=model.config.hidden_size,
        low_rank_dimension=config.rank)})

    reft_model = get_reft_model(model, reft_config)
    reft_model.print_trainable_parameters()
    print("ReFT model initialization completed!")

    batch_triggers = [tokenizer.apply_chat_template([{"role": "user", "content": f"{q}"}], tokenize=False) for q in data_questions[:config.num_samples]]
    batch_sequences = [f"{a}" for a in data_answers]

    if config.test_rephrase:
        batch_rephrase_triggers = [tokenizer.apply_chat_template([{"role": "user", "content": f"{q}"}], tokenize=False) for q in data_rephrase_questions[:config.num_samples]]
        batch_rephrase_sequences = [f"{a}" for a in data_answers]

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    

    results = []
    for data_idx, item in enumerate(range(len(batch_triggers))):
        print("==="*30)
        query_embedding = original_query_embeddings[item]
        best_idx, best_score = compute_similarity_with_activations(query_embedding.to(device), activation_embeddings)
        best_idx = data_idx_to_cluster_id[best_idx]
        print(f"Selected adapter {best_idx} (similarity: {best_score:.4f})")
        reft_model = load_reft_adapter(reft_model, config.adapter_weights_dir, best_idx, device, config)
        
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
            rephrase_query_embedding = rephrased_query_embeddings[item]
            rephrase_best_idx, rephrase_best_score = compute_similarity_with_activations(rephrase_query_embedding, activation_embeddings)
            rephrase_best_idx = data_idx_to_cluster_id[rephrase_best_idx]
            print(f"Selected adapter {rephrase_best_idx} for rephrase (similarity: {rephrase_best_score:.4f})")
            reft_model = load_reft_adapter(reft_model, config.adapter_weights_dir, rephrase_best_idx, device, config)
            
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
            
            aligned=True if data_idx_to_cluster_id[data_idx] == rephrase_best_idx else False
            res_dict["aligned"] = aligned

        results.append(res_dict)

        print(f"Rouge-L score: {rouge_score:.3f}")
        if config.test_rephrase:
            print(f"Rouge-L score for paraphrase: {rouge_score_rephrase:.3f}")

    os.makedirs(f"results/{config.dataset}", exist_ok=True)

    with open(f"results/{config.dataset}/{config.save_path}_{config.num_samples}_cluster.jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    print(f"Results saved to results/{config.save_path}_{config.num_samples}_cluster.jsonl")

if __name__ == "__main__":
    import sys
    from utils import yaml_defaults_from_argv

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="YAML config; its values become defaults, any CLI flag overrides")
    parser.add_argument("--dataset", type=str, default="unke_v3")
    parser.add_argument("--target_layer", type=int, default=15)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--test_rephrase", type=bool, default=True)
    parser.add_argument("--activation_path", type=str, default="./activation/unke/llama_3_8b_layer15_no_answer_last_original.pt")
    parser.add_argument("--adapter_weights_dir", type=str, default="./Stored_weights/cluster_unke_explicit_v3")
    parser.add_argument("--original_query_activation_path", type=str, default="./activation/unke/llama_3_8b_layer15_no_answer_last_original.pt")
    parser.add_argument("--rephrased_query_activation_path", type=str, default="./activation/unke/llama_3_8b_layer15_no_answer_last_rephrased.pt")
    parser.add_argument("--save_path", type=str, default="vanilla_unke")
    parser.add_argument("--cluster_indices_path", type=str, default="./cluster_index/unke/unke_v3_3_hac_maxsize8.json")

    _cfg = yaml_defaults_from_argv(sys.argv)
    if _cfg:
        _valid = {a.dest for a in parser._actions}
        parser.set_defaults(**{k: v for k, v in _cfg.items() if k in _valid})

    args = parser.parse_args()

    config = ReftHyperparameters()
    config.dataset = args.dataset
    config.target_layer = args.target_layer
    config.rank = args.rank
    config.num_samples = args.num_samples
    config.model_name = args.model_name
    config.test_rephrase = args.test_rephrase
    config.activation_path = args.activation_path
    config.adapter_weights_dir = args.adapter_weights_dir 
    config.original_query_activation_path = args.original_query_activation_path
    config.rephrased_query_activation_path = args.rephrased_query_activation_path
    config.save_path = args.save_path
    config.cluster_indices_path = args.cluster_indices_path
    print(config)
    evaluate_rep(config)