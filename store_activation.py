import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BertForSequenceClassification, BertTokenizer
import argparse
import random
import numpy as np
import pandas as pd
import torch
import copy
import json
import yaml
from datasets import load_dataset, Dataset, concatenate_datasets
from peft import get_peft_model, LoraConfig
from torch.utils.data import DataLoader
import evaluate
from tqdm import tqdm
from transformers import AutoModel,AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch import Tensor
import os
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS
from src.dataset.unke import UnkeForDirectOpt
from src.dataset.wiki import WikiForDirectOpt
from src.dataset.akew import CounterFactForDirectOpt
from k_means_constrained import KMeansConstrained
from src.dataset.anyedit import AnyEditForDirectOpt
import matplotlib.pyplot as plt
import numpy as np


Target_layer=15


def preprocess_only_question(example,tokenizer,dataset_name, data_src="original"):
    results = {
        "input_ids": [],
        "attention_mask": [],
        "label": [],
    }
    if dataset_name == "counterfact":
      
        for i in range(len(example["question"])):
            if data_src == "original":
                question = example["question"][i]
            
            
            # Format as chat conversation using the tokenizer's chat template
                messages = [
                    {"role": "user", "content": question}
                ]
                
                
                if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
                    # Get just the user message part with generation prompt
                    formatted_text = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True  # This adds the assistant prompt
                    )
                else:
                    # Fallback to simple format if no chat template available
                    formatted_text = f"User: {question}"
                
                # Tokenize the formatted question
                tokenized = tokenizer(
                    formatted_text,
                    padding=False,
                    truncation=True,
                )
                results["input_ids"].append(tokenized.input_ids)
                results["attention_mask"].append(tokenized.attention_mask)
                results["label"].append(1)
            elif data_src == "rephrased":
               
                question = example["para_question"][i][0]
                
                messages = [
                            {"role": "user", "content": question}
                        ]
                        
                        # Apply chat template
                if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
                            # Get just the user message part with generation prompt
                    formatted_text = tokenizer.apply_chat_template(
                                messages, 
                                tokenize=False, 
                                add_generation_prompt=True  # This adds the assistant prompt
                            )
                else:
                            # Fallback to simple format if no chat template available
                    formatted_text = f"User: {question}"
                        
                tokenized = tokenizer(
                            formatted_text,
                            padding=False,
                            truncation=True,
                        )
                results["input_ids"].append(tokenized.input_ids)
                results["attention_mask"].append(tokenized.attention_mask)
                results["label"].append(1)
        return results
    
    elif dataset_name == "unke" or dataset_name == "wiki" or dataset_name == "anyedit" or dataset_name == "unke_v3":
       
        for i in range(len(example["question"])):

            if data_src == "original":
                question = example["question"][i]
            elif data_src == "rephrased":
            
                question = example["para_question"][i]
            
            messages = [
                {"role": "user", "content": question}
            ]
            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
                formatted_text = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True  # This adds the assistant prompt
                )
            else:
                # Fallback to simple format if no chat template available
                formatted_text = f"User: {question}\nAssistant:"
            tokenized = tokenizer(
                formatted_text,
                padding=False,
                truncation=True,
            )
            results["input_ids"].append(tokenized.input_ids)
            results["attention_mask"].append(tokenized.attention_mask)
            results["label"].append(1)
        return results
    return results

def preprocess_answer(example,tokenizer,dataset_name, data_src="original"):
    results = {
        "input_ids": [],
        "attention_mask": [],
        "label": [],
    }
    if dataset_name == "unke" or dataset_name == "wiki" or dataset_name == "anyedit":
      

        for i in range(len(example["question"])):
            if data_src == "original":
                question = example["question"][i]
            elif data_src == "rephrased":
                question = example["para_question"][i]
            messages = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": example["answer"][i]}
                ]
            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
                    formatted_text = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True  # This adds the assistant prompt
                    )
                
            tokenized = tokenizer(
                    formatted_text,
                    padding=False,
                    truncation=True,
                )
            results["input_ids"].append(tokenized.input_ids)
            results["attention_mask"].append(tokenized.attention_mask)
            results["label"].append(1)
        return results
        
    return results



class GetHookedValue:
    def __init__(self, model_name, device='cpu',checkpoint_value=None,tokenizer=None):
        self.model_name = model_name
        self.device = device
        self.activations = {}
        self.handles = []
    

        # Load the tokenizer and model
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer=tokenizer

        # Set model to evaluation mode
        self.store_feature_list = []
        self.layer_output=[]
        self.model.eval()
        self.register_hooks()

    def hook_fn(self, module, input, output):
        # Store the input activations right before the down_proj layer
        # input[0] contains the activations going into the down_proj
        res = input[0].detach().cpu()
        self.activations['layer_'+str(Target_layer)] = res

    def register_hooks(self):
        # Register hook specifically on layer Target_layer
        target_module = self.model.model.layers[Target_layer]
        handle = target_module.register_forward_hook(self.hook_fn)
        self.handles.append(handle)

    def remove_hooks(self):
        # Remove all hooks to avoid memory leaks
        for handle in self.handles:
            handle.remove()

    def get_activations(self, inputs, cal_type="mean"):
        # Clear activations dictionary
        self.activations = {}

        

        
        inputs["input_ids"] =   torch.tensor(inputs["input_ids"]).unsqueeze(0).to(self.model.device)
        inputs["attention_mask"] = torch.tensor(inputs["attention_mask"]).unsqueeze(0).to(self.model.device)
        if  "label" in inputs:
            del inputs["label"]
        # Run the model (no gradient computation needed)
        with torch.no_grad():
            self.model(**inputs)
        

    
        if 'layer_'+str(Target_layer) in self.activations:
            if cal_type=="mean":
                self.activations['layer_'+str(Target_layer)] = torch.mean(self.activations['layer_'+str(Target_layer)], dim=1)
            elif cal_type=="last":
                self.activations['layer_'+str(Target_layer)] = self.activations['layer_'+str(Target_layer)][:, -1, :]
            
        return self.activations
        

    def inference(self, text_data, cal_type="mean"):

        for data in tqdm(text_data):
            
            res = self.get_activations(data, cal_type=cal_type)
              
            self.layer_output.append(res[f'layer_{Target_layer}'])
        
        return self.layer_output    
    


def store_activations(model_name="meta-llama/Llama-3.1-8B-Instruct", dataset_name="unke", only_question=False, data_src="original"):


    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if dataset_name == "unke":
        edit_dataset = UnkeForDirectOpt().get_dataset()
    elif dataset_name == "unke_v3":
        edit_dataset = UnkeForDirectOpt().get_dataset_v3()
    elif dataset_name == "wiki":
        edit_dataset = WikiForDirectOpt().get_dataset()
    elif dataset_name == "counterfact":
        edit_dataset = CounterFactForDirectOpt().get_dataset()
    elif dataset_name == "anyedit":
        edit_dataset = AnyEditForDirectOpt().get_dataset()

        
    # edit_dataset=edit_dataset.select(range(500))
    
    if only_question is True:
        dataset=edit_dataset.map(preprocess_only_question,batched=True,remove_columns=edit_dataset.column_names,fn_kwargs={'tokenizer': tokenizer,'dataset_name':dataset_name, 'data_src':data_src})

    else:
        dataset=edit_dataset.map(preprocess_answer,batched=True,remove_columns=edit_dataset.column_names,fn_kwargs={'tokenizer': tokenizer,'dataset_name':dataset_name, 'data_src':data_src})
    

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


    Hook=GetHookedValue(model_name,device=device,tokenizer=tokenizer) 

    res_inference=Hook.inference(dataset, cal_type="last")

    stored_folder=f"activation/{dataset_name}/"
    if not os.path.exists(stored_folder):
        os.makedirs(stored_folder)

    model_prefix = "qwen_2_5_7b" if "qwen" in model_name.lower() else "llama_3_8b"
    stored_path=stored_folder+f"{model_prefix}_layer{Target_layer}_no_answer_last_{data_src}.pt"
    print("Length of res_inference",len(res_inference))
    torch.save(res_inference,stored_path)
    

if __name__ == "__main__":
    # store_activations(dataset_name="anyedit", only_question=True, data_src="rephrased")
    # store_activations(dataset_name="anyedit", only_question=True, data_src="original")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="unke_v3")
    parser.add_argument("--only_question", type=bool, default=True)
    parser.add_argument("--data_src", type=str, default="original")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--target_layer", type=int, default=15)
    args = parser.parse_args()
    Target_layer = args.target_layer
    store_activations(model_name=args.model_name, dataset_name=args.dataset_name, only_question=args.only_question, data_src=args.data_src)