import argparse
import glob
import json
from multiprocessing.pool import Pool

import torch
from tqdm import tqdm

# PT_DIRS = "/disk1/sajad/datasets/sci/arxiv-long/v1/bert-files/512-sectioned-seqAllen-real/"

parser = argparse.ArgumentParser()
parser.add_argument("-read_from", default='')
parser.add_argument("-dataset", default='')
parser.add_argument("-set", default='')
parser.add_argument("-single_json_base", default='')
parser.add_argument("-selective", default='')

args = parser.parse_args()

PT_DIRS = args.read_from
dataset = args.dataset
set = args.set
single_json_base = args.single_json_base
selective_set = args.selective


def _multi_write_from_json_files(params):

    f, set, write_file = params

    paper = json.load(open(f))
    sentences = [s[-2].lower() for s in paper['sentences']]

    if len(sentences) > 3:
        inst = {"set": set,
                "dataset": dataset,
                "abstract_id": str(paper['filename']),
                "sentences": sentences
                }

        with open(write_file, mode='a') as f:
            json.dump(inst, f)
            f.write('\n')


if len(single_json_base) == 0:
    for se in [set]:
        whole_instances = []
        for f in tqdm(glob.glob(PT_DIRS + se + '*.pt'), total=len(glob.glob(PT_DIRS + se + '*.pt'))):
            instances = torch.load(f)
            for instance in instances:
                whole_instances.append(instance)
        c = 0
        saved_insts = []
        for instance in whole_instances:
            paper_id = instance['paper_id']
            sentences = instance['src_txt']
            inst = {'set': se, 'dataset': dataset, 'abstract_id': paper_id, 'sentences': sentences,
                    "labels": ["background", "objective", "result", "result"], "confs": [0.7895, 0.4211, 0.6316, 0.7895]}
            saved_insts.append(inst)

            with open(dataset + '.' + se + '.jsonl', mode='a') as f:
                json.dump(inst, f)
                f.write('\n')


elif len(single_json_base) > 0:
    files = []
    for se in [set]:
        whole_papers = {}

        if len(selective_set) > 0:
            papers = []
            with open(selective_set) as F:
                for l in F:
                    papers.append(l.replace('.pdf', '.json').strip())

        for f in tqdm(glob.glob(args.single_json_base + '/' + set + '/*.json'), total=len(glob.glob(args.single_json_base + '/' + set + '/*.json'))):
            if len(selective_set) > 0 and f.split('/')[-1] in papers:
                files.append((f, set, dataset + '.' + se + '.jsonl'))
            elif len(selective_set) == 0:
                files.append((f, set, dataset + '.' + se + '.jsonl'))


        for f in tqdm(files, total=len(files)):
            _multi_write_from_json_files(f)
            # import pdb;pdb.set_trace()


