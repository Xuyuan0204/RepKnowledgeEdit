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
from src.dataset.anyedit import AnyEditForDirectOpt

from k_means_constrained import KMeansConstrained

import matplotlib.pyplot as plt
import numpy as np


Target_layer=15


def preprocess_only_question(example,tokenizer,dataset_name):
    results = {
        "input_ids": [],
        "attention_mask": [],
        "label": [],
    }
    if dataset_name == "counterfact":
      
        for i in range(len(example["question"])):
            
            question = example["question"][i]
            
            
            messages = [
                    {"role": "user", "content": question}
                ]
                
                
            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
                    # Get just the user message part with generation prompt
                formatted_text = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True  
                    )
            else:
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
    
    elif dataset_name == "unke" or dataset_name == "unke_v3" or dataset_name == "wiki" or dataset_name == "anyedit":
        for i in range(len(example["question"])):
            question = example["question"][i]
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




class GetHookedValue:
    def __init__(self, model_name, device='cpu',checkpoint_value=None,tokenizer=None):
        self.model_name = model_name
        self.device = device
        self.activations = {}
        self.handles = []
    
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer=tokenizer

        self.store_feature_list = []
        self.layer_output=[]
        self.model.eval()
        self.register_hooks()

    def hook_fn(self, module, input, output):
        res = input[0].detach().cpu()
        self.activations['layer_'+str(Target_layer)] = res

    def register_hooks(self):
        # Register hook specifically on layer Target_layer
        target_module = self.model.model.layers[Target_layer]
        handle = target_module.register_forward_hook(self.hook_fn)
        self.handles.append(handle)

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()

    def get_activations(self, inputs, cal_type="mean"):
        
        self.activations = {}

        

        
        inputs["input_ids"] =   torch.tensor(inputs["input_ids"]).unsqueeze(0).to(self.model.device)
        inputs["attention_mask"] = torch.tensor(inputs["attention_mask"]).unsqueeze(0).to(self.model.device)
        if  "label" in inputs:
            del inputs["label"]
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
    


def cluster_activations(dataset_name="wiki",method="cluster", only_question=False, model_name="meta-llama/Llama-3.1-8B-Instruct"):


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
        dataset=edit_dataset.map(preprocess_only_question,batched=True,remove_columns=edit_dataset.column_names,fn_kwargs={'tokenizer': tokenizer,'dataset_name':dataset_name})

    else:
        dataset=edit_dataset.map(preprocess_answer,batched=True,remove_columns=edit_dataset.column_names,fn_kwargs={'tokenizer': tokenizer,'dataset_name':dataset_name})



    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Hook=GetHookedValue(model_name,device=device,tokenizer=tokenizer) 

    res_inference=Hook.inference(dataset, cal_type="last")

    stored_folder=f"output/{dataset_name}/"
    if not os.path.exists(stored_folder):
        os.makedirs(stored_folder)

    model_prefix = "qwen_2_5_7b" if "qwen" in model_name.lower() else "llama_3_8b"
    stored_path=stored_folder+f"{model_prefix}_layer{Target_layer}_no_answer_last.pt"

    torch.save(res_inference,stored_path)

    # Canonical cluster-index location shared with train_cluster.py / test_cluster_rep.py
    dataset_dir = dataset_name.replace("_v3", "")
    cluster_out_folder = f"cluster_index/{dataset_dir}/"
    if not os.path.exists(cluster_out_folder):
        os.makedirs(cluster_out_folder)
    if method=="cluster":

        print("Applying KMeans clustering...")


        activations_array = torch.cat(res_inference, dim=0).numpy()  # Shape: [num_samples, hidden_size]
        cluster_size=8
        cluster_num=int(np.ceil(len(edit_dataset)/cluster_size))
        if cluster_num> len(edit_dataset)/cluster_size:
            size_min= cluster_size-1

        cluster_num=16
        kmeans = KMeans(
            n_clusters=cluster_num,
            init='k-means++',
            n_init=10,
            max_iter=1000,
            random_state=0
        )

        cluster_labels = kmeans.fit_predict(activations_array)

        cluster_indices = {}
        for idx, cluster_id in enumerate(cluster_labels):
            cluster_id = int(cluster_id)  # Convert to int for JSON serialization
            if cluster_id not in cluster_indices:
                cluster_indices[cluster_id] = []
            cluster_indices[cluster_id].append(idx)

        cluster_indices_path = cluster_out_folder + f"{dataset_name}_3_kmeans_k{cluster_num}.json"
        with open(cluster_indices_path, 'w') as f:
            json.dump(cluster_indices, f, indent=2)

        print(f"Clustering completed. Cluster indices saved to: {cluster_indices_path}")
        print(f"Cluster distribution:")
        for cluster_id, indices in cluster_indices.items():
            print(f"  Cluster {cluster_id}: {len(indices)} samples")
    
    elif method=="hac":
        print("Applying Hierarchical Agglomerative Clustering with complete linkage...")
        
        activations_array = torch.cat(res_inference, dim=0).numpy()  # Shape: [num_samples, hidden_size]
        
        tau = 0.9
        max_cluster_size =8 # Upper bound on cluster size
        distance_threshold = 1 - tau  # cut when max pairwise distance <= 1 - tau
        
        def split_large_cluster(cluster_indices, cluster_activations, max_size, base_tau=0.90, min_tau=0.10):
            """
            Recursively split a cluster that exceeds max_size by applying progressively higher tau values (stricter thresholds).
            """
            if len(cluster_indices) <= max_size:
                return [cluster_indices]
            
            current_tau = base_tau
            tau_step = 0.02
            max_tau = 0.99 # Maximum tau to try before giving up
            
            while current_tau <= max_tau:
                current_tau += tau_step  # Increase tau for stricter clustering
                current_distance_threshold = 1 - current_tau
               
                sub_clust = AgglomerativeClustering(
                    n_clusters=None,
                    linkage="complete",
                    metric="cosine",
                    distance_threshold=current_distance_threshold
                )
                sub_labels = sub_clust.fit_predict(cluster_activations)
                
                sub_clusters = {}
                for i, sub_label in enumerate(sub_labels):
                    if sub_label not in sub_clusters:
                        sub_clusters[sub_label] = []
                    sub_clusters[sub_label].append(cluster_indices[i])
                
                if len(sub_clusters) > 1:
                    final_clusters = []
                    for sub_cluster_indices in sub_clusters.values():
                        if len(sub_cluster_indices) <= max_size:
                            final_clusters.append(sub_cluster_indices)
                        else:
                            sub_activations = cluster_activations[[cluster_indices.index(idx) for idx in sub_cluster_indices]]
                            final_clusters.extend(split_large_cluster(sub_cluster_indices, sub_activations, max_size, current_tau, min_tau))
                    return final_clusters
            force_clust = AgglomerativeClustering(
                n_clusters=None,
                linkage="complete", 
                metric="cosine",
                distance_threshold=1 - 0.99  # Very strict threshold
            )
            force_labels = force_clust.fit_predict(cluster_activations)
            
            micro_clusters = {}
            for i, label in enumerate(force_labels):
                if label not in micro_clusters:
                    micro_clusters[label] = []
                micro_clusters[label].append(cluster_indices[i])
            
            micro_cluster_list = list(micro_clusters.values())
            final_clusters = []
            current_group = []
            
            for micro_cluster in micro_cluster_list:
                if len(current_group) + len(micro_cluster) <= max_size:
                    current_group.extend(micro_cluster)
                else:
                    if current_group:
                        final_clusters.append(current_group)
                    current_group = micro_cluster[:]
            
            if current_group:
                final_clusters.append(current_group)
            
            return final_clusters
        
        # Apply initial AgglomerativeClustering
        clust = AgglomerativeClustering(
            n_clusters=None,
            linkage="complete",
            metric="cosine",            # distance = 1 - cosine_similarity
            distance_threshold=distance_threshold  # cut when max pairwise distance <= 1 - tau
        )
        cluster_labels = clust.fit_predict(activations_array)
        
        initial_cluster_indices = {}
        for idx, cluster_id in enumerate(cluster_labels):
            cluster_id = int(cluster_id)  
            if cluster_id not in initial_cluster_indices:
                initial_cluster_indices[cluster_id] = []
            initial_cluster_indices[cluster_id].append(idx)
        
        print(f"Initial clustering created {len(initial_cluster_indices)} clusters")
        
        final_cluster_indices = {}
        cluster_counter = 0
        large_clusters_count = 0
        
        for cluster_id, indices in initial_cluster_indices.items():
            if len(indices) > max_cluster_size:
                large_clusters_count += 1
                print(f"Splitting cluster {cluster_id} (size: {len(indices)}) into smaller clusters...")
                
                cluster_activations = activations_array[indices]
                
                split_clusters = split_large_cluster(indices, cluster_activations, max_cluster_size, tau)
                
                for split_cluster in split_clusters:
                    final_cluster_indices[cluster_counter] = split_cluster
                    cluster_counter += 1
            else:
                
                final_cluster_indices[cluster_counter] = indices
                cluster_counter += 1
        
        cluster_indices_path = cluster_out_folder + f"{dataset_name}_3_hac_maxsize{max_cluster_size}.json"
        with open(cluster_indices_path, 'w') as f:
            json.dump(final_cluster_indices, f, indent=2)

        print(f"HAC clustering completed. Cluster indices saved to: {cluster_indices_path}")
        print(f"Total clusters: {len(final_cluster_indices)} (split {large_clusters_count} large clusters)")
        print(f"Cluster distribution:")
        cluster_sizes = []
        for cluster_id, indices in final_cluster_indices.items():
            size = len(indices)
            cluster_sizes.append(size)
            print(f"  Cluster {cluster_id}: {size} samples")
        
        print(f"Cluster size statistics: min={min(cluster_sizes)}, max={max(cluster_sizes)}, mean={np.mean(cluster_sizes):.1f}")
        if max(cluster_sizes) > max_cluster_size:
            print(f"Warning: Some clusters still exceed max size of {max_cluster_size}")
        else:
            print(f"✓ All clusters respect the maximum size limit of {max_cluster_size}")
    
    
    elif method=="hash":
        return 

if __name__ == "__main__":

    cluster_activations(dataset_name="unke_v3", method="hac", only_question=True,
                        model_name="meta-llama/Llama-3.1-8B-Instruct")