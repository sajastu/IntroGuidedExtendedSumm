import argparse
import collections
import glob
import json
import numpy as np
import os
import pickle
import statistics
from multiprocessing.pool import Pool

import torch
from tqdm import tqdm

from utils.rouge_score import evaluate_rouge, evaluate_rouge_avg

parser = argparse.ArgumentParser()
parser.add_argument("-pt_dirs_src", default='')
parser.add_argument("-set", default='')

args = parser.parse_args()

# if args.set == 'train':
#     os._exit(0)

PT_DIRS = args.pt_dirs_src


def check_path_existense(dir):
    if os.path.exists(dir):
        return
    os.makedirs(dir)


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

def Diff(li1, li2):
    return (list(list(set(li1)-set(li2)) + list(set(li2)-set(li1))))

def cal_rg_single(params):
    oracle, gold = params

    r1, r2, rl = evaluate_rouge([oracle], [gold])

    return r1,r2,rl

for se in [args.set]:

    oracles = {}
    golds = {}
    avg_sents_len = {}
    sent_true_labels = collections.defaultdict(dict)
    debug_sent_lists = []
    paper_ids = set()
    for j, f in tqdm(enumerate(glob.glob(os.path.join(PT_DIRS, se + '*.pt'))), total=len(glob.glob(os.path.join(PT_DIRS, se + '*.pt')))):
        instances = torch.load(f)
        for inst_idx, instance in enumerate(instances):
            sentences = instance['src_txt']
            sent_labels = instance['sent_labels']

            rg_scores = instance['src_sent_rg']
            gold_summary = instance['tgt_txt'].replace('<q>', '')
            paper_id = instance['paper_id'].split('___')[0]
            if paper_id == "astro-ph9805315":
                print(instance['paper_id'])
            paper_ids.add(paper_id)
            sent_true_labels[f][instance['paper_id']] = sent_labels
            # import pdb;pdb.set_trace()
            new_labels = []
            instance_picked_up = 0

            if len(sent_labels) != len(sentences) or len(instance['src_txt']) != len(instance['clss']):
                print(instance['paper_id'].split('___')[0])
                # print(f)
                import pdb;pdb.set_trace()
                sent_labels = [0 for _ in range(len(sentences))]

            for j, s in enumerate(sentences):

                if sent_labels[j] == 1:
                    instance_picked_up += 1
                    if paper_id not in oracles:
                        oracles[paper_id] = s + ' '
                    else:
                        oracles[paper_id] += s
                        oracles[paper_id] += ' '
                # else:
            if paper_id not in avg_sents_len:
                avg_sents_len[paper_id] = instance_picked_up
                # sent_true_labels[paper_id] = sent_labels
            else:
                avg_sents_len[paper_id] += instance_picked_up
                # if paper_id=="PMC6426562.nxml":
                #     print(f)
                #     import pdb;pdb.set_trace()
                # sent_true_labels[paper_id] += sent_labels
                # if avg_sents_len[paper_id] > 10:
            golds[paper_id] = gold_summary
    # check_path_existense("sent_labels_files/pubmedL/")
    # pickle.dump(sent_true_labels, open("sent_labels_files/pubmedL/" + args.set + ".labels.p", "wb"))
    # import pdb;pdb.set_trace()
    for diff in Diff(oracles.keys(), golds.keys()):
        oracles[diff] = ''
        print(diff)
    oracles = dict(sorted(oracles.items()))
    golds = dict(sorted(golds.items()))
    avg_sents_len = dict(sorted(avg_sents_len.items()))

    paper_ids = list(paper_ids)

    with open(f"{se}_paper_ids.txt", mode='w') as FF:
        for p in paper_ids:
            FF.write(p)
            FF.write('\n')

    print('avg oracle sentence number: {}'.format(statistics.mean(avg_sents_len.values())))
    print('median oracle sentence number: {}'.format(statistics.median(avg_sents_len.values())))
    oracle_sets = []
    for o, g in zip(oracles.values(), golds.values()):
        oracle_sets.append((o, g))

    avg_rg = {'1': [], '2':[], 'l':[]}
    pool = Pool(24)
    for d in tqdm(pool.imap_unordered(cal_rg_single, oracle_sets), total=len(oracle_sets)):
        avg_rg['1'].append(d[0])
        avg_rg['2'].append(d[1])
        avg_rg['l'].append(d[2])

    # r1, r2, rl = evaluate_rouge_avg(oracles.values(), golds.values(), use_progress_bar=True)
    # r1, r2, rl = 1,1,1
    r1 = np.mean(avg_rg['1'])
    r2 = np.mean(avg_rg['2'])
    rl = np.mean(avg_rg['l'])
    print('r1: {}, r2: {}, rl: {}'.format(r1, r2, rl))

    if not is_non_zero_file(PT_DIRS + '/' + 'config.json'):
        config = collections.defaultdict(dict)
        for metric, score in zip(["RG-1", "RG-2", "RG-L"], [r1, r2, rl]):
            config[se][metric] = score

        config[se]["Avg oracle sentence length"] = statistics.mean(avg_sents_len.values())
        config[se]["Median oracle sentence length"] = statistics.median(avg_sents_len.values())
        with open(PT_DIRS + '/' + 'config.json', mode='w') as F:
            json.dump(config, F, indent=4)
    else:
        config = json.load(open(PT_DIRS + '/' + 'config.json'))
        config_all = collections.defaultdict(dict)
        for key, val in config.items():
            for k, v in val.items():
                config_all[key][k] = v

        for metric, score in zip(["RG-1", "RG-2", "RG-L"], [r1, r2, rl]):
            config_all[se][metric] = score
        config_all[se]["Avg oracle sentence length"] = statistics.mean(avg_sents_len.values())
        config_all[se]["Median oracle sentence length"] = statistics.median(avg_sents_len.values())
        with open(PT_DIRS + '/' + 'config.json', mode='w') as F:
            json.dump(config_all, F, indent=4)
