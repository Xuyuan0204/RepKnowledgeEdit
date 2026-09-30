import random
from collections import defaultdict
import json
from datasets import Dataset

random.seed(42)


class WikiForDirectOpt:
    def __init__(self):
        self.dataset = defaultdict()
        self.dataset = self.get_dataset()

    def get_dataset(self):
        raw_dataset = json.load(open("datasets/wiki/wiki_edit_data.json"))

        edit_dict = {"question": [], "para_question": [], "answer": []}
        for i in range(len(raw_dataset)):
            for k in edit_dict:
                edit_dict[k].append(raw_dataset[i][k])
        edit_dataset = Dataset.from_dict(edit_dict)

        return edit_dataset
