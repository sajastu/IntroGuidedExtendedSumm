import argparse
import collections
import glob
import os
from multiprocessing.pool import Pool

import torch
from tqdm import tqdm

from utils.rouge_utils import greedy_selection, greedy_selection_section_based_intro_conc, \
    greedy_selection_section_based_intro_conc_logical

parser = argparse.ArgumentParser()
parser.add_argument("-pt_dirs_src", default='')
parser.add_argument("-write_to", default='')
parser.add_argument("-set", default='')
parser.add_argument("-n_sents", type=int, required=True)
parser.add_argument("-greedy", type=bool, default=False)

args = parser.parse_args()

PT_DIRS = args.pt_dirs_src
PT_DIRS_DEST = args.write_to
ST = args.set
SENT_NUM = args.n_sents
IS_GREEDY = args.greedy

import spacy

nlp = spacy.load("en_core_sci_lg")
nlp.disable_pipes('ner', 'tagger', 'parser')
nlp.add_pipe(nlp.create_pipe('sentencizer'))


def tokenize_sent(text_src):
    doc_src = nlp(text_src)
    toks = []
    for tok in doc_src:
        if len(tok.text.strip()) > 0 and '\\u' not in tok.text.strip():
            toks.append(tok.text)
    return toks


def modify_labels_multi_(params):
    f, PT_DIRS_DEST = params
    # if os.path.exists(PT_DIRS_DEST + f.split('/')[-1]):
    #     return
    paper_count = set()
    instances = torch.load(f)
    new_sent_labels = collections.defaultdict(dict)
    for inst_idx, instance in enumerate(instances):
        sentences = instance['src_txt']
        tgt_sents = instance['tgt_txt'].split('<q>')
        rg_scores = instance['src_sent_labels']
        paper_id = instance['paper_id'].split('___')[0]
        instance_id = instance['paper_id']
        papers_src[paper_id][instance_id] = sentences
        papers_rgs[paper_id][instance_id] = rg_scores
        papers_tgts[paper_id] = tgt_sents

    for idx, paper_id in enumerate(papers_src):
        sections_sents = papers_src[paper_id]
        sections_sents_rg = papers_rgs[paper_id]
        all_sents = []
        all_sents_rg = []

        for sect_id, sect_sents in sections_sents.items():
            all_sents.extend(sect_sents)

        for sect_id, sect_sents in sections_sents_rg.items():
            all_sents_rg.extend(sect_sents)

        all_sents = [(i, sent) for i, sent in enumerate(all_sents)]

        zip_sents = zip(all_sents, all_sents_rg)

        if not IS_GREEDY:
            zip_sents = sorted(list(zip_sents), key=lambda x: x[1], reverse=True)
            top_sents_labels = [z[0][0] for z in zip_sents[:SENT_NUM]]

        else:
            top_sents_labels = greedy_selection([tokenize_sent(z[0][1]) for z in zip_sents],
                                                [tokenize_sent(s) for s in papers_tgts[paper_id]], SENT_NUM)

        sent_labels = [0] * len(all_sents)
        for l in top_sents_labels:
            sent_labels[l] = 1

        # new_sent_labels[paper_id] = sent_labels
        already_picked = 0
        for j, (sect_id, sect_sents) in enumerate(sections_sents.items()):
            try:
                should_picked = len(sect_sents)
                new_sent_labels[paper_id][sect_id] = sent_labels[already_picked:already_picked + should_picked]
                already_picked += should_picked
            except:
                pass

    new_instances_to_save = []
    for inst_idx, instance in enumerate(instances):
        instance_id = instance['paper_id']
        paper_id = instance['paper_id'].split('___')[0]

        paper_labels = new_sent_labels[paper_id]
        if len(paper_labels) > 0:
            paper_count.add(paper_id)
            sent_labels = new_sent_labels[paper_id][instance_id]
            instance['sent_labels'] = sent_labels

            new_instances_to_save.append(instance)

    check_path_existense(PT_DIRS_DEST)
    torch.save(new_instances_to_save, PT_DIRS_DEST + f.split('/')[-1])
    print('Saved file: {} with instances : {} and papers: {}'.format(PT_DIRS_DEST + f.split('/')[-1],
                                                                     len(new_instances_to_save), len(paper_count)))
    return len(paper_count)

paper_ids = set()

def modify_labels_multi_section_based(params):
    f, PT_DIRS_DEST = params
    print(f"Processing: {f}")
    # if os.path.exists(PT_DIRS_DEST + f.split('/')[-1]):
    #     return
    paper_count = set()
    instances = torch.load(f)
    new_sent_labels = collections.defaultdict(dict)
    papers_src = collections.defaultdict(dict)
    papers_rgs = collections.defaultdict(dict)
    papers_sent_sections = collections.defaultdict(dict)
    papers_tgts = collections.defaultdict(dict)
    sent_sect_labels_whole = collections.defaultdict(dict)
    papers_bert_count = list()
    for inst_idx, instance in enumerate(instances):
        sentences = instance['src_txt']
        tgt_sents = instance['tgt_txt'].split('<q>')
        rg_scores = instance['src_sent_rg']
        sent_sections_txt = instance['sent_sections_txt']
        sent_sect_labels = instance['sent_sect_labels']
        paper_id = instance['paper_id'].split('___')[0]
        paper_ids.add(paper_id)

        papers_bert_count.append(str(inst_idx) + ' ' + str(instance['paper_id']))
        papers_src[paper_id][instance['paper_id']] = sentences
        # if instance['paper_id']=='hep-ex0111032___measurement of the @xmath and @xmath decays___0':
        #     print(paper_id)
        # import pdb;pdb.set_trace()

        papers_rgs[paper_id][instance['paper_id']] = rg_scores
        papers_sent_sections[paper_id][instance['paper_id']] = sent_sections_txt
        sent_sect_labels_whole[paper_id][instance['paper_id']] = sent_sect_labels
        papers_tgts[paper_id] = tgt_sents

    # with open('papers.txt', mode='w') as F:
    #     for s in papers_bert_count:
    #         F.write(str(s) + '\n')

    sect_percentage = {'abstract': 0, 'intro': 0, 'method':0, 'exp': 0, 'res': 0, 'conc': 0}
    sect_percentage_seqIDS = {'0': 0, '1': 0, '2':0, '3': 0, '4': 0}
    for idx, paper_id in enumerate(papers_src):
        segment_sents = papers_src[paper_id]
        segment_sents_section = papers_sent_sections[paper_id]
        sections_lens = {}


        for seg_idx, seg_sents_section in segment_sents_section.items():
            for sect in seg_sents_section:
                if sect not in sections_lens.keys():
                    sections_lens[sect] = 1
                else:
                    sections_lens[sect] +=1

        # for seg_idx, seg_sents in segment_sents.items():
        #     if seg_idx.split('___')[1] not in sections_lens.keys():
        #         sections_lens[seg_idx.split('___')[1]] = len(seg_sents)
        #     else:
        #         sections_lens[seg_idx.split('___')[1]] += len(seg_sents)
        sections_lens_vals = list(sections_lens.values())
        sections_sents_rg = papers_rgs[paper_id]
        sections_sent_sections = papers_sent_sections[paper_id]
        all_sents = []
        all_sents_rg = []
        all_sections_text = []

        for sect_id, seg_sents in segment_sents.items():
            all_sents.extend(seg_sents)

        # if paper_id=='1403.2376':
        #     import pdb;pdb.set_trace()

        if len(all_sents) < SENT_NUM:
            # del new_sent_labels[paper_id]
            continue

        for sect_id, seg_sents in sections_sents_rg.items():
            all_sents_rg.extend(seg_sents)

        for sect_id, seg_sents in sections_sent_sections.items():
            all_sections_text.extend(seg_sents)

        all_sections_text = list(dict.fromkeys(all_sections_text))

        all_sents = [(i, sent) for i, sent in enumerate(all_sents)]

        zip_sents = zip(all_sents, all_sents_rg)

        if not IS_GREEDY:
            zip_sents = sorted(list(zip_sents), key=lambda x: x[1], reverse=True)
            top_sents_labels = [z[0][0] for z in zip_sents[:SENT_NUM]]

        else:

            # top_sents_labels, _, dist_percentage = greedy_selection_section_based_intro_conc(
            #     paper_id,
            #     [tokenize_sent(z[0][1]) for z in zip_sents], [tokenize_sent(s) for s in papers_tgts[paper_id]],
            #     sections_lens,
            #     all_sections_text,
            #     SENT_NUM)

            # try:
            top_sents_labels, _, dist_percentage, dist_sect_ID_percentage = greedy_selection_section_based_intro_conc_logical(
                paper_id,
                [tokenize_sent(z[0][1]) for z in zip_sents], [tokenize_sent(s) for s in papers_tgts[paper_id]],
                sections_lens_vals,
                all_sections_text,
                SENT_NUM,
                doc_section_list=sum(sent_sect_labels_whole[paper_id].values(), []) )
            # except:
            #     import pdb;pdb.set_trace()

            for k,val in dist_percentage.items():
                sect_percentage[k]+=val

            for k,val in dist_sect_ID_percentage.items():
                sect_percentage_seqIDS[k] += val

        sent_labels = [0] * len(all_sents)
        for l in top_sents_labels:
            sent_labels[l] = 1

        already_picked = 0
        for j, (sect_id, seg_sents) in enumerate(segment_sents.items()):
            try:
                should_picked = len(seg_sents)
                new_sent_labels[paper_id][sect_id] = sent_labels[already_picked:already_picked + should_picked]
                already_picked += should_picked

            except:
                pass


    new_instances_to_save = []
    for inst_idx, instance in enumerate(instances):
        paper_id = instance['paper_id'].split('___')[0]

        paper_labels = new_sent_labels[paper_id]
        if paper_id=="astro-ph9805315":
            print(instance['paper_id'])
        # if sum([sum(s) for s in paper_labels.values()]) > 15:
        #     print('heree')
        sum([sum(labels) for seg, labels in paper_labels.items()])
        sum([len(labels) for seg, labels in paper_labels.items()])

        # if paper_id=='1403.2376':
        #     import pdb;pdb.set_trace()

        # assert sum([len(labels) for seg, labels in paper_labels.items()]) == sum([len(sents) for seg, sents in papers_src[paper_id].items()]), \
        # f"paper_id: {paper_id} and file: {f}"

        if sum([sum(labels) for seg, labels in paper_labels.items()]) > 0:
            # and sum(
                # [len(sents) for seg, sents in papers_src[paper_id].items()]) > SENT_NUM:
            paper_count.add(paper_id)
            sent_labels = new_sent_labels[paper_id][instance['paper_id']]
            # if sum(sent_labels) > 15:
            #     print(paper_id)
            instance['sent_labels'] = sent_labels
            # if paper_id == 'hep-ex0111032' and instance['paper_id']=='hep-ex0111032___measurement of the @xmath and @xmath decays___0':
            #     import pdb;
            #     pdb.set_trace()
            new_instances_to_save.append(instance)

    check_path_existense(PT_DIRS_DEST)
    torch.save(new_instances_to_save, os.path.join(PT_DIRS_DEST, f.split('/')[-1]))

    print('Saved file: {} with instances : {} and papers: {}'.format(os.path.join(PT_DIRS_DEST, f.split('/')[-1]),
                                                                     len(new_instances_to_save), len(paper_count)))
    return len(paper_count), sect_percentage, sect_percentage_seqIDS


def check_path_existense(dir):
    if os.path.exists(dir):
        return
    os.makedirs(dir)


# for se in ["train", "test", "val"]:
for se in [ST]:
    total_papers = 0

    golds = {}
    avg_sents_len = {}
    a_lst = []
    tmp_list = []
    explored_ids = []
    whole_sect_percentage = {'abstract':0, 'intro':0, 'method':0, 'exp':0, 'res':0, 'conc':0}
    whole_sect_percentage_seqIDS = {'0':0, '1':0, '2':0, '3':0, '4':0}
    print(
        'There are {} bert files that should be loaded...'.format(len(glob.glob(os.path.join(PT_DIRS, se+'*.pt')))))
    for j, f in enumerate(glob.glob(os.path.join(PT_DIRS, se+'*.pt'))):
        #################################
        ########## DEBUG #############
        #################################
        tmp_list.append((f, PT_DIRS_DEST))
        # modify_labels_multi_section_based((f, PT_DIRS_DEST))

        #################################
        ########## DEBUG #############
        #################################

    pool = Pool(24)
    for d in tqdm(pool.imap_unordered(modify_labels_multi_section_based, tmp_list), total=len(tmp_list)):
        total_papers += d[0]
        for k,v in d[1].items():
            whole_sect_percentage[k] += v
        for k,v in d[2].items():
            whole_sect_percentage_seqIDS[k] += v

    print('---------\n Done with {} set with total papers of {}'.format(se, total_papers))

    print(
        '-Distributions: abstract: {:4.4f}, introduction: {:4.4f}, method: {:4.4}, experimental:{:4.4f}, results: {:4.4f}, conc:{:4.4f}'.
            format(
            whole_sect_percentage['abstract'] / total_papers,
            whole_sect_percentage['intro'] / total_papers,
            whole_sect_percentage['method'] / total_papers,
            whole_sect_percentage['exp'] / total_papers,
            whole_sect_percentage['res'] / total_papers,
            whole_sect_percentage['conc'] / total_papers,
        )
    )

    print(
        '-Distributions SeqIDS: [0]: {:4.4f}, [1]: {:4.4f}, [2]: {:4.4}, [3]:{:4.4f}, [4]: {:4.4f}'.
            format(
            whole_sect_percentage_seqIDS['0'] / total_papers,
            whole_sect_percentage_seqIDS['1'] / total_papers,
            whole_sect_percentage_seqIDS['2'] / total_papers,
            whole_sect_percentage_seqIDS['3'] / total_papers,
            whole_sect_percentage_seqIDS['4'] / total_papers,
        )
    )