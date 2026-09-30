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


Target_layer=18


def preprocess_only_question(example,tokenizer,dataset_name, data_src="original"):
    results = {
        "input_ids": [],
        "attention_mask": [],
        "label": [],
    }
    if dataset_name == "mmlu":
      
        for i in range(len(example["mmlu_questions"])):
           
            question = example["mmlu_questions"][i]
            
            
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
    


def store_activations(dataset_name="unke", data_src="original"):


    
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if dataset_name == "unke":
        edit_dataset = UnkeForDirectOpt().get_dataset()
    elif dataset_name == "unke_v3":
        edit_dataset = UnkeForDirectOpt().get_dataset_v3()
    elif dataset_name == "wiki":
        edit_dataset = WikiForDirectOpt().get_dataset()
    elif dataset_name == "mmlu":
        edit_dataset = UnkeForDirectOpt().get_dataset_mmlu()
    elif dataset_name == "counterfact":
        edit_dataset = CounterFactForDirectOpt().get_dataset()
    elif dataset_name == "anyedit":
        edit_dataset = AnyEditForDirectOpt().get_dataset()

    
    dataset=edit_dataset.map(preprocess_only_question,batched=True,remove_columns=edit_dataset.column_names,fn_kwargs={'tokenizer': tokenizer,'dataset_name':dataset_name, 'data_src':data_src})

    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


    Hook=GetHookedValue(model_name,device=device,tokenizer=tokenizer) 

    res_inference=Hook.inference(dataset, cal_type="last")

    stored_folder=f"activation/{dataset_name}/"
    if not os.path.exists(stored_folder):
        os.makedirs(stored_folder)

    stored_path=stored_folder+f"qwen_2_5_7b_layer{Target_layer}_no_answer_last_{data_src}.pt"

    torch.save(res_inference,stored_path)
    

if __name__ == "__main__":

    store_activations(dataset_name="mmlu",)