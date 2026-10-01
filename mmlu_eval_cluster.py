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
import wandb
from utils import load_intervention_weights_consreft, load_intervention_weights_loreft
from src.dataset.unke import UnkeForDirectOpt
from src.dataset.anyedit import AnyEditForDirectOpt
import argparse
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm





def map_answer_to_choice(answer):
    if answer == "A":
        return 0
    elif answer == "B":
        return 1
    elif answer == "C":
        return 2
    elif answer == "D":
        return 3
    else:
        return None

def extract_answer_choice(response):
    """Extract A, B, C, or D from model response"""
    import re
    response = response.strip().upper()
    
    patterns = [
        r'\b([ABCD])\b',  # Single letter
        r'ANSWER[:\s]*([ABCD])',  # "Answer: A" format
        r'([ABCD])\.',  # "A." format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    
    for char in response:
        if char in 'ABCD':
            return char
    
    return None

@dataclass
class ReftHyperparameters:
    num_samples: int = 10
    rank: int = 4  # low_rank_dimension
    target_layer: int = 15,
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    dataset: str = "mmlu"
    test_rephrase: bool = True
    activation_path: str = "./activation/unke/llama_3_8b_layer15_no_answer_last_original.pt"
    adapter_weights_dir: str = "./Stored_weights/unke_explicit_llama_cluster"
    original_query_activation_path: str = "./activation/unke/llama_3_8b_layer15_no_answer_last_original.pt"
    rephrased_query_activation_path: str = "./activation/unke/llama_3_8b_layer15_no_answer_last_rephrased.pt"
    save_path: str = "vanilla_unke"
    similarity_threshold: float = 0.9  # threshold for using intervention vs vanilla model
    cluster_indices_path: str = "./cluster_index/unke/unke_v3_3_hac_maxsize8.json"


def mmlu_question(que, choices, tokenizer):
    """Format MMLU question with multiple choice options for batch processing"""
    fulltext = []
    for i in range(len(que)):
        question_text = f"{que[i]}\n\nA. {choices[i][0]}\nB. {choices[i][1]}\nC. {choices[i][2]}\nD. {choices[i][3]}\n\nAnswer:"
        formatted_text = tokenizer.apply_chat_template([
            {"role": "system", "content": "You are a helpful assistant. Answer the multiple choice question by directly selecting A, B, C, or D."},
            {"role": "user", "content": question_text}
        ], tokenize=False, add_generation_prompt=True)
        fulltext.append(formatted_text)
    return fulltext
    


def load_activation_embeddings(activation_path):
    """Load and preprocess activation embeddings from .pt file"""
    activations = torch.load(activation_path, map_location='cpu')
    
    if isinstance(activations, list):
        activations = torch.stack(activations)
    
    if len(activations.shape) == 3 and activations.shape[1] == 1:
        activations = activations.squeeze(1)
    
    return activations

def compute_similarity_with_activations(query_embedding, activation_embeddings):
   
    query_norm = F.normalize(query_embedding.unsqueeze(0), p=2, dim=1)
    activation_norm = F.normalize(activation_embeddings, p=2, dim=1)
    
    similarities = torch.mm(query_norm, activation_norm.T).squeeze(0)
    
    best_idx = similarities.argmax().item()
    best_score = similarities[best_idx].item()
    
    return best_idx, best_score

def load_reft_adapter(reft_model, adapter_weights_dir, adapter_idx, device, config):
    """Load specific adapter weights into the ReFT model"""
    adapter_path = os.path.join(adapter_weights_dir, f"adapter_{adapter_idx}")
    
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")
    
    # Load intervention weights - reuse existing loading function
    reft_config = ReftConfig(representations={
        "layer": config.target_layer, "component": "block_output",
        "intervention": LoreftIntervention(
        embed_dim=reft_model.model.config.hidden_size,
        low_rank_dimension=config.rank)})
    
    load_intervention_weights_loreft(reft_model, adapter_path, device, reft_config)
    
    return reft_model

def evaluate_rep(config):
    

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config.dataset == "mmlu":
        edit_dataset = UnkeForDirectOpt().get_dataset_mmlu()

    if config.num_samples is not None:
        edit_dataset = edit_dataset.select(range(config.num_samples))
    data_questions = [sample["mmlu_questions"] for sample in edit_dataset]
    data_answers = [sample["mmlu_answer"] for sample in edit_dataset]
    data_choices = [sample["mmlu_choices"] for sample in edit_dataset]

    

    model_max_length =256
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name, model_max_length=model_max_length, 
        padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    mmlu_questions = mmlu_question(data_questions, data_choices, tokenizer)

    
    activation_embeddings = load_activation_embeddings(config.activation_path).to(device)
    
    original_query_embeddings = load_activation_embeddings("./activation/mmlu/llama_3_8b_layer15_no_answer_last_original.pt").to(device)
   

    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name, device_map=device)

    reft_config = ReftConfig(representations={
        "layer": config.target_layer, "component": "block_output",
        "intervention": LoreftIntervention(
        embed_dim=model.config.hidden_size,
        low_rank_dimension=config.rank)})

    reft_model = get_reft_model(model, reft_config)
    reft_model.print_trainable_parameters()
    print("ReFT model initialization completed!")

    batch_triggers = [ q for q in mmlu_questions ]
    
    with open(config.cluster_indices_path, 'r') as f:
        cluster_data = json.load(f)
    print(f"Loaded {len(cluster_data)} cluster indices")

    data_idx_to_cluster_id = {}
    for cluster_id, indices in tqdm(cluster_data.items()):
        for data_idx in indices:
            data_idx_to_cluster_id[data_idx] = int(cluster_id)
    
    print(f"Built mapping for {len(data_idx_to_cluster_id)} data indices across {len(cluster_data)} clusters")
    
  
    results = []
    acc = 0
    for data_idx, item in tqdm(enumerate(range(len(batch_triggers)))):
        print("==="*30)
        print(f"Processing item {item+1}/{len(batch_triggers)}")
        
        print(f"Original query: '{data_questions[item]}'")
        query_embedding = original_query_embeddings[item]
        best_idx, best_score = compute_similarity_with_activations(query_embedding, activation_embeddings)
        best_idx = data_idx_to_cluster_id[best_idx]
        print(f"Selected adapter {best_idx} (similarity: {best_score:.4f})")
        
        test_prompt_tokens = tokenizer(batch_triggers[item], return_tensors="pt").to(device)
        
        if best_score >= config.similarity_threshold:
            print(f"Using ReFT intervention (score {best_score:.4f} >= threshold {config.similarity_threshold})")
            reft_model = load_reft_adapter(reft_model, config.adapter_weights_dir, best_idx, device, config)
            
            base_unit_location = test_prompt_tokens["input_ids"].shape[-1] - 1
            
            _, steered_response = reft_model.generate(
                test_prompt_tokens, 
                unit_locations={"sources->base": (None, [[[base_unit_location]]])},
                intervene_on_prompt=True, 
                max_new_tokens=32, 
                do_sample=False, 
                eos_token_id=tokenizer.eos_token_id, 
                early_stopping=True,
                temperature=0.001
            )
            generated_text = tokenizer.decode(steered_response[0][len(test_prompt_tokens["input_ids"][0]):], skip_special_tokens=True)
            used_intervention = True
        else:
            print(f"Using vanilla model (score {best_score:.4f} < threshold {config.similarity_threshold})")
            # Use the base model without intervention
            with torch.no_grad():
                vanilla_response = model.generate(
                    test_prompt_tokens["input_ids"],
                    attention_mask=test_prompt_tokens["attention_mask"],
                    max_new_tokens=32,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    early_stopping=True,
                )
            generated_text = tokenizer.decode(vanilla_response[0][len(test_prompt_tokens["input_ids"][0]):], skip_special_tokens=True)
            used_intervention = False
        answer_choice = map_answer_to_choice(extract_answer_choice(generated_text))
        
        groudtruth_answers=data_answers[item]        
        data_results = {
            "question": data_questions[item],
            "answer_choice": answer_choice,
            "groudtruth_answers": groudtruth_answers,
            "generated_text": generated_text,
            "correct": True if answer_choice == groudtruth_answers else False,
            "similarity_score": best_score,
            "used_intervention": used_intervention,
            "selected_adapter": best_idx if used_intervention else None
        }
        if answer_choice == groudtruth_answers:
            acc += 1
        results.append(data_results)
        
    print(f"Accuracy: {acc/len(batch_triggers)}")
    if not os.path.exists("./results/mmlu_results"):
        os.makedirs("./results/mmlu_results")
    with open(f"./results/mmlu_results/life_long_cluster.jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mmlu")
    parser.add_argument("--target_layer", type=int, default=15)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--test_rephrase", type=bool, default=True)
    parser.add_argument("--activation_path", type=str, default="./activation/unke/llama_3_8b_layer15_no_answer_last_original.pt")
    parser.add_argument("--adapter_weights_dir", type=str, default="./Stored_weights/unke_explicit_llama_cluster")
    parser.add_argument("--original_query_activation_path", type=str, default="./activation/unke_v3/llama_3_8b_layer15_no_answer_last_original.pt")
    parser.add_argument("--rephrased_query_activation_path", type=str, default="./activation/unke_v3/llama_3_8b_layer15_no_answer_last_rephrased.pt")
    parser.add_argument("--save_path", type=str, default="vanilla_unke")
    parser.add_argument("--similarity_threshold", type=float, default=0.9)
    parser.add_argument("--cluster_indices_path", type=str, default="./cluster_index/unke/unke_v3_3_hac_maxsize8.json")


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
    config.similarity_threshold = args.similarity_threshold
    config.cluster_indices_path = args.cluster_indices_path
    print(config)
    evaluate_rep(config)