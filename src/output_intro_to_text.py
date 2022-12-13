import glob
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.rouge_score import evaluate_rouge_avg, evaluate_rouge

intro_kws = set()
for f in glob.glob("csv_files/arxiv_sects_info.csv"):
    try:
        df = pd.read_csv(f)
    except:
        continue
    for d in df['intro']:
        intro_kws.add(d)
intro_kws = list(intro_kws)

text_files = {
    'train': [],
    'val': [],
    'test': []
}

# for set in ["val", "test", "train"]:
for set in ["test"]:
    for f in tqdm(glob.glob("/disk1/sajad/datasets/sci/arxivL/splits-with-sections-introConc/" + set + '/*.json'), total=len(glob.glob("/disk1/sajad/datasets/sci/arxivL/splits-with-sections-introConc/" + set + '/*.json'))):
        with open(f, mode='r') as fR:
            intro_str = ''
            ent = json.load(fR)

            sections = []
            for s in ent['sentences']:
                if s[1] not in sections:
                    sections.append(s[1])
                if s[1].lower() in intro_kws:
                    sent = ' '.join([s for s in s[0] if s!='-' and s!='_'])
                    intro_str += sent
                    intro_str += ' '
            if len(intro_str) == 0:
                first_section = ent['sentences'][0][1].lower()
                for s in ent['sentences']:
                    if s[1].lower() == first_section:
                        sent = ' '.join([s for s in s[0] if s != '-' and s != '_'])
                        intro_str += sent
                        intro_str += ' '
            gold_str = ' '.join([' '.join([t for t in e if t != '-' and t != '_']) for e in ent['gold'] ])
            paper_id = ent['id']
            text_files[set].append(
                {
                    'paper_id': paper_id,
                    'intro': intro_str,
                    'summary': gold_str,
                }
            )

def _mp_rg(f):
    return evaluate_rouge([f['intro']], [f['summary']])

for set, files in text_files.items():
    # if not os.path.exists(f'/disk1/sajad/datasets/sci/arxivL/intro_summary/{set}/'):
    #     os.makedirs(f'/disk1/sajad/datasets/sci/arxivL/intro_summary/{set}/')
    wr_dir = f'/disk1/sajad/datasets/sci/arxivL/intro_summary/'
    # with open(f'{wr_dir}/{set}.json', mode='w') as fW:
    #     for f in files:
    #         json.dump(f, fW)
    #         fW.write('\n')

    if set == "test":
        scores = {
            'r1': [],
            'r2': [],
            'rl': [],
        }


        pool = Pool(9)
        for cal in tqdm(pool.imap_unordered(_mp_rg, files), total=len(files)):
            scores['r1'].append(cal[0])
            scores['r2'].append(cal[1])
            scores['rl'].append(cal[2])

        print(
            f'{np.mean(np.asarray(scores["r1"]))} \n'
            f'{np.mean(np.asarray(scores["r2"]))} \n'
            f'{np.mean(np.asarray(scores["rl"]))} \n'
        )

        # print(evaluate_rouge_avg([f['intro'] for f in files], [f['summary'] for f in files]))