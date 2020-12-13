import glob
import json
import operator
import os

import torch
from tqdm import tqdm

PT_DIRS = "/disk1/sajad/datasets/sci/pubmed-dataset/bert-files-450/512-seqAllen-labels-whole-sectioned/"
JSON_DIRS = "/disk1/sajad/datasets/sci/pubmed-dataset/jsons-450/plain/"

def check_path_existense(dir):
    if os.path.exists(dir):
        return
    os.makedirs(dir)

for se in ["test"]:
    count = set()
    oracles = {}
    golds = {}
    avg_sents_len = {}
    inst_count = 0
    for j, f in tqdm(enumerate(glob.glob(PT_DIRS + se + '*.pt')), total=len(glob.glob(PT_DIRS + se + '*.pt'))):
        instances = torch.load(f)
        for inst_idx, instance in enumerate(instances):
            sentences = instance['src_txt']
            sent_labels = instance['sent_labels']
            rg_scores = instance['src_sent_labels']
            gold_summary = instance['tgt_txt'].replace('<q>','')
            paper_id = instance['paper_id'].split('___')[0]
            count.add(paper_id)
        print(len(instances))
        inst_count += len(instances)

    print(len(count))
    print(inst_count)


# for se in ["val", "train", "test"]:
#     count = 0
#     oracles = {}
#     golds = {}
#     avg_sents_len = {}
#     for j, f in tqdm(enumerate(glob.glob(JSON_DIRS + se + '*.json')), total=len(glob.glob(JSON_DIRS + se + '*.json'))):
#         instances = json.load(open(f))
#         count += len(instances)
#     print(count)
