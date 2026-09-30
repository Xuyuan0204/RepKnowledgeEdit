import csv
import random
from collections import defaultdict
from pathlib import Path
import copy
import json
import torch
from datasets import load_dataset, Dataset
import transformers


random.seed(42)


def dict_to(d, device):
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.to(device)
        elif isinstance(v, dict):
            new_dict[k] = dict_to(v, device)
        else:
            new_dict[k] = v

    return new_dict


class UnkeForDirectOpt:
    def __init__(self):
        self.dataset = defaultdict()
        self.dataset = self.get_dataset()

    def get_dataset(self):
        raw_dataset = json.load(open("datasets/UnKE/final_data_v2.json"))

        edit_dict = {"question": [], "para_question": [], "answer": []}
        for i in range(len(raw_dataset)):
            for k in edit_dict:
                edit_dict[k].append(raw_dataset[i][k])
        edit_dataset = Dataset.from_dict(edit_dict)

        return edit_dataset

    def get_dataset_v3(self):
        raw_dataset = json.load(open("datasets/UnKE/final_data_v3.json"))

        edit_dict = {"question": [], "para_question": [], "answer": []}
        for i in range(len(raw_dataset)):
            for k in edit_dict:
                edit_dict[k].append(raw_dataset[i][k])
        edit_dataset = Dataset.from_dict(edit_dict)

        return edit_dataset

    def get_dataset_mmlu(self):
        raw_dataset = []
        with open("datasets/UnKE/mmlu_shot.jsonl", "r") as f:
            for line in f:
                raw_dataset.append(json.loads(line.strip()))

        edit_dict = {"mmlu_questions": [], "mmlu_answer": [], "mmlu_choices": []}
        for item in raw_dataset:
            edit_dict["mmlu_questions"].append(item["question"])
            edit_dict["mmlu_answer"].append(item["answer"])
            edit_dict["mmlu_choices"].append(item["choices"])
        edit_dataset = Dataset.from_dict(edit_dict)

        return edit_dataset

    # def __preprocess__(self, tokenizer):

    #     def preprocess_dataset(examples):
    #         results = {
    #             "input_ids": [],
    #             "attention_mask": [],
    #             "label": [],
    #             "question_length": [],
    #         }

    #         for i in range(len(examples["question"])):
    #             # Use the main question as the prompt
    #             prompt = examples["question"][i]
    #             # Create full text with question + answer
                
                
    #             if 
    #             tokenized = tokenizer(
    #                 full_text,
    #                 truncation=True,
    #                 padding="max_length",
    #                 max_length=512,
    #                 add_special_tokens=True,
    #             )
                
    #             # Calculate number of tokens in the question (to mask in labels)
    #             num_question_token = len(tokenizer.tokenize(prompt, add_special_tokens=True))
    #             num_padding_token = len(tokenized.attention_mask) - sum(tokenized.attention_mask)
                
    #             if tokenizer.padding_side == 'left':
    #                 num_label_mask = num_question_token + num_padding_token
    #             else:
    #                 num_label_mask = num_question_token
                
    #             # Create labels with question portion masked (-100)
    #             label = copy.deepcopy(tokenized.input_ids)
    #             for j in range(num_label_mask):
    #                 label[j] = -100

    #             results["input_ids"].append(torch.tensor(tokenized.input_ids))
    #             results["attention_mask"].append(torch.tensor(tokenized.attention_mask))
    #             results["label"].append(torch.tensor(label))
    #             results["question_length"].append(torch.tensor(num_question_token))

    #         return results

    #     # Apply preprocessing to the dataset
    #     processed_dataset = self.dataset.map(
    #         preprocess_dataset, 
    #         batched=True, 
    #         remove_columns=self.dataset.column_names
    #     )
        
    #     processed_dataset.set_format(
    #         type="torch",
    #         columns=[
    #             "input_ids",
    #             "attention_mask", 
    #             "label",
    #             "question_length",
    #         ],
    #     )
        
    #     self.dataset = processed_dataset

    # def build_traindataset(self, tokenizer):
    #     self.__preprocess__(tokenizer)
    #     return self.dataset