import glob

import torch
from tqdm import tqdm

paper_src = {}
paper_tgt = {}
DATASET='pubmedL'
for s in ["test"]:
    for file in tqdm(glob.glob(f"/disk1/sajad/datasets/sci/{DATASET}/bert-files/2048-segmented-intro2048-20-introConc/{s}*.pt")):
        instances = torch.load(file)
        for instance in instances:
            src_txt = instance['src_txt']
            tgt_txt = instance['tgt_txt']

            paper_id = instance['paper_id'].split('___')[0]

            if paper_id not in paper_src.keys():
                paper_src[paper_id] = ' '.join(src_txt)
            else:
                paper_src[paper_id] += ' '
                paper_src[paper_id] += ' '.join(src_txt)

            if paper_id not in paper_tgt.keys():
                paper_tgt[paper_id] = tgt_txt


for p_id, txt in paper_tgt.items():
    with open(f'/disk1/sajad/datasets/sci/{DATASET}/text/test/' + p_id + '.target', mode='a') as F:
        F.write(txt)

for p_id, txt in paper_src.items():
    with open(f'/disk1/sajad/datasets/sci/{DATASET}/text/test/' + p_id + '.src', mode='a') as F:
        F.write(txt)

