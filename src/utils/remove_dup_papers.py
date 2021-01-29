import collections
import glob

import torch
from tqdm import tqdm

papers_id = set()
for f in tqdm(glob.glob("/disk1/sajad/datasets/sci/arxivL/bert-files/2500-segmented-intro1536-15/*.pt"), total=len(glob.glob("/disk1/sajad/datasets/sci/arxivL/bert-files/2500-segmented-intro1536-15/*.pt"))):
    instances = torch.load(f)
    new_sent_labels = collections.defaultdict(dict)
    papers_src = collections.defaultdict(dict)
    papers_rgs = collections.defaultdict(dict)
    papers_sent_sections = collections.defaultdict(dict)
    papers_tgts = collections.defaultdict(dict)
    sent_sect_labels_whole = collections.defaultdict(dict)
    papers_bert_count = list()
    new_instances = []
    for inst_idx, instance in enumerate(instances):
        paper_id = '___'.join(instance['paper_id'].split('___')[0:2])
        if paper_id not in papers_id:
            new_instances.append(instance)
            papers_id.add(paper_id)
        else:
            print('dup')
            continue

    torch.save(new_instances, f)