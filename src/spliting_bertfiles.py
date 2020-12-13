import argparse
import glob
import json
import operator
import os

import torch
from tqdm import tqdm

import statistics

parser = argparse.ArgumentParser()
parser.add_argument("-pt_dirs_src", default='')
# parser.add_argument("-write_to", default='')

args = parser.parse_args()

PT_DIRS = args.pt_dirs_src
PT_DIRS_CHUNK = PT_DIRS[:-1] + '-chunked/'

def check_path_existense(dir):
    if os.path.exists(dir):
        return
    os.makedirs(dir)

CHUNK_SIZE_TRAIN = 3000
CHUNK_SIZE_VALID = 2000

size_dict = {'train':CHUNK_SIZE_TRAIN, 'val':CHUNK_SIZE_VALID, 'test':CHUNK_SIZE_VALID}

for se in ["train", "val", "test"]:

    oracles = {}
    golds = {}
    avg_sents_len = {}
    whole_instances = []

    written_files = glob.glob(PT_DIRS + '/' + se + '*.pt')

    idxs = [int(w.split('/')[-1].split('.')[1]) for w in written_files]

    written_files = zip(written_files, idxs)

    written_files = sorted(dict(written_files).items(), key=lambda x:x[1])
    written_files = [w[0] for w in written_files]

    for j, f in tqdm(enumerate(written_files), total=len(written_files)):
        instances = torch.load(f)
        for inst_idx, instance in enumerate(instances):
            whole_instances.append(instance)

    total_chunks = (len(whole_instances) // size_dict[se]) + 1
    print('Total chunks: {}'.format(total_chunks))
    check_path_existense(PT_DIRS_CHUNK)
    for i in range(total_chunks):
        try:
            torch.save(whole_instances[i*size_dict[se]: (i+1)*size_dict[se]], PT_DIRS_CHUNK + se + '.' + str(i) + '.pt')
            print('Saved file: {}'.format(PT_DIRS_CHUNK + se + '.' + str(i) + '.pt'))
        except:
            torch.save(whole_instances[i*size_dict[se]:], PT_DIRS_CHUNK + se + '.' + str(i) + '.pt')
            print('Saved file: {}'.format(PT_DIRS_CHUNK + se + '.' + str(i) + '.pt'))

