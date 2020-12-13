import argparse
import csv
import json
import math
import os
import pickle
import traceback
from collections import Counter
from datetime import datetime
from multiprocessing.pool import Pool
from operator import itemgetter

import numpy as np
import torch
from sentence_transformers import util
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.rouge_score import evaluate_rouge, evaluate_rouge_avg
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()


def visualize_heatmap(data, p_id, y_labels, x_labels, oracle_idxs, summary_idx, sect_lens, bs_selected_ids):
    x_labels_tmp = []

    for x in x_labels:
        if x in oracle_idxs and x in bs_selected_ids and x in summary_idx:
            x_labels_tmp.append("[(*" +str(x) + "*)]")
        elif x in oracle_idxs and x in bs_selected_ids:
            x_labels_tmp.append("(*" +str(x) + "*)")
        elif x in oracle_idxs and x in summary_idx:
            x_labels_tmp.append("[*" +str(x) + "*]")
        elif x in oracle_idxs :
            x_labels_tmp.append("*" +str(x) + "*")
        elif x in summary_idx :
            x_labels_tmp.append("[" +str(x) + "]")
        elif x in bs_selected_ids :
            x_labels_tmp.append("(" +str(x) + ")")
        else:
            x_labels_tmp.append(str(x))

    plt.figure(figsize=(len(x_labels) / 1, 16))
    data = np.array(data)

    data_tmp = list()
    # import pdb;pdb.set_trace()
    for j, d in enumerate(data):
        dtemp = []
        for idx, dd in enumerate(d):
            if j % 2 == 1:
                if idx in [summary_idx[(j-1)//2]]:
                    dtemp.append("%.2f**"%dd)
                else:
                    dtemp.append("%.2f"%dd)
            else:
                dtemp.append("%.2f"%dd)

        data_tmp.append(dtemp)

    mask = np.zeros_like(data)
    mask[data<0.0009] = True
    data_tmp = np.array(data_tmp)
    ax = sns.heatmap(data, linewidths=.5, cmap="YlGnBu", annot=data_tmp, fmt="", xticklabels=x_labels_tmp,yticklabels=y_labels, mask=mask)
    # ax = sns.heatmap(data, linewidths=.5, cmap="YlGnBu", annot=data_tmp, fmt="", xticklabels=x_labels_tmp,yticklabels=y_labels)

    sectLens = list(sect_lens.values())


    vLines = [sum(sectLens[:idx+1]) for idx,len in enumerate(sectLens)]
    hLines = [y_idx for y_idx, label in enumerate(y_labels) if y_idx%2==0]

    ax.vlines(vLines, *ax.get_ylim())
    ax.hlines(hLines, *ax.get_xlim())

    plt.savefig("mmr_figs/"+ p_id + '.pdf')



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_path_existence(dir):
    if os.path.exists(dir):
        return
    else:
        os.makedirs(dir)


def _report_rouge(predictions, references):
    r1, r2, rl = evaluate_rouge_avg(predictions, references)

    print("Metric\tScore\t95% CI")
    print("ROUGE-1\t{:.4f}\t({:.4f},{:.4f})".format(r1, 0, 0))
    print("ROUGE-2\t{:.4f}\t({:.4f},{:.4f})".format(r2, 0, 0))
    print("ROUGE-L\t{:.4f}\t({:.4f},{:.4f})".format(rl, 0, 0))
    return r1, r2, rl


def get_cosine_sim(embedding_a, embedding_b):
    try:
        embedding_a = torch.from_numpy(embedding_a)
    except:
        embedding_a = embedding_a

    try:
        embedding_b = torch.from_numpy(embedding_b)
    except:
        embedding_b = embedding_b

    cosine_scores = util.pytorch_cos_sim(embedding_a, embedding_b)

    if cosine_scores < 0:
        return 0
    else:
        return cosine_scores.item()


def get_cosine_sim_from_txt(sent_a, sent_b, model):
    embeddings1 = model.encode(sent_a, convert_to_tensor=True)
    embeddings2 = model.encode(sent_b, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

    if cosine_scores < 0:
        return 0
    else:
        return cosine_scores.item()


def cal_mmr_unified(source_sents, partial_summary, partial_summary_idx, sentence_encodings, partial_summary_sects,
                    sent_scores, sent_sects_whole_true, section_contributions, sent_sect_len, co1, co2, co3, cos, beta=0.1):
    current_mmrs = []
    for idx, (sent, score, contr) in enumerate(zip(source_sents, sent_scores, section_contributions)):
        if idx in partial_summary_idx:
            current_mmrs.append(0)
            continue
        else:
            # current_mmrs.append(1.0)
            current_mmrs.append(score * contr * sent_sect_len[idx] / len(source_sents))
    return current_mmrs, []


# def cal_mmr(source_sents, partial_summary, partial_summary_idx, sentence_encodings, partial_summary_sects,
#             sent_scores, sent_sects_whole_true, section_contributions, co1, co2, co3, cos, beta=0.1):
#     current_mmrs = []
#     deductions = []
#     # MMR formula:
#
#     ## MMR = argmax (\alpha Sim(si, D) - \beta max SimSent(si, sj) - \theta max SimSect(sj, sj))
#     for idx, sent in enumerate(source_sents):
#         cosine_vals = []
#         if idx in partial_summary_idx:
#             current_mmrs.append(-1000)
#             continue
#
#         section_contrib = section_contributions[idx]
#         sent_txt = sent
#         sent_section = sent_sects_whole_true[idx]
#
#         ######################
#         ## calculate first term
#         ######################
#
#         first_subterm1 = sent_scores[idx]
#         # if cos:
#         #     first_subterm2 = get_cosine_sim(sentence_encodings, idx)
#         #     first_term = (.95 * first_subterm1) + (.05 * first_subterm2)
#         # else:
#         #     first_term = first_subterm1
#
#         first_term = first_subterm1
#
#         ######################
#         # calculate second term
#         ######################
#         if co2 > 0:
#             max_rg_score = 0
#             for sent in partial_summary:
#                 rg_score = evaluate_rouge([sent], [sent_txt], type='p')[2]
#                 if rg_score > max_rg_score:
#                     max_rg_score = rg_score
#             second_term = max_rg_score
#
#             # for summary_idx, sent in enumerate(partial_summary):
#             #     cosine_vals.append(get_cosine_sim_from_txt(sent, sent_txt, model))
#             # cosine_vals.append(get_cosine_sim(sentence_encodings[partial_summary_idx[summary_idx]], sentence_encodings[idx]))
#             # max_cos = max(cosine_vals)
#             # second_term = max_cos
#
#         else:
#             second_term = 0
#
#         ######################
#         # calculate third term
#         ######################
#
#         partial_summary_sects_counter = {}
#         for sect in partial_summary_sects:
#             if sect not in partial_summary_sects_counter:
#                 partial_summary_sects_counter[sect] = 1
#             else:
#                 partial_summary_sects_counter[sect] += 1
#
#         for sect in partial_summary_sects_counter:
#             if sect == sent_section:
#                 partial_summary_sects_counter[sect] = ((partial_summary_sects_counter[sect] + 1) / 10) * (1/section_contrib) * beta
#                 # partial_summary_sects_counter[sect] = ((partial_summary_sects_counter[sect] + 1) / len(partial_summary_idx))
#             else:
#                 partial_summary_sects_counter[sect] = ((partial_summary_sects_counter[sect]) / 10) * (1/section_contrib) * beta
#
#         third_term = max(partial_summary_sects_counter.values())
#         # print(co1, co2, co3)
#         mmr_sent = co1 * first_term - co2 * second_term - co3 * third_term
#         # mmr_sent = co1 * first_term - co3 * third_term
#         # mmr_sent = co1 * first_term - co3 * third_term
#         current_mmrs.append(mmr_sent)
#         # deductions.append((co2 * second_term,co3 * third_term))
#         deductions.append((co3 * third_term))
#         # deductions.append((co2 * second_term))
#     return current_mmrs, deductions

def cal_mmr(source_sents, partial_summary, partial_summary_idx, sentence_encodings, partial_summary_sects,
            sent_scores, sent_sects_whole_true, section_contributions, co1, co2, co3, cos, beta=0.1):
    current_mmrs = []
    deductions = []
    # MMR formula:

    ## MMR = argmax (\alpha Sim(si, D) - \beta max SimSent(si, sj) - \theta max SimSect(sj, sj))
    for idx, sent in enumerate(source_sents):
        cosine_vals = []
        if idx in partial_summary_idx:
            current_mmrs.append(-1000)
            continue

        section_contrib = section_contributions[idx]
        sent_txt = sent
        sent_section = sent_sects_whole_true[idx]

        ######################
        ## calculate first term
        ######################

        first_subterm1 = sent_scores[idx]
        # if cos:
        #     first_subterm2 = get_cosine_sim(sentence_encodings, idx)
        #     first_term = (.95 * first_subterm1) + (.05 * first_subterm2)
        # else:
        #     first_term = first_subterm1

        first_term = first_subterm1

        ######################
        # calculate second term
        ######################
        if co2 > 0:
            max_rg_score = 0
            for sent in partial_summary:
                rg_score = evaluate_rouge([sent], [sent_txt], type='p')[2]
                if rg_score > max_rg_score:
                    max_rg_score = rg_score
            second_term = max_rg_score

            # for summary_idx, sent in enumerate(partial_summary):
            #     cosine_vals.append(get_cosine_sim_from_txt(sent, sent_txt, model))
            # cosine_vals.append(get_cosine_sim(sentence_encodings[partial_summary_idx[summary_idx]], sentence_encodings[idx]))
            # max_cos = max(cosine_vals)
            # second_term = max_cos

        else:
            second_term = 0

        ######################
        # calculate third term
        ######################

        partial_summary_sects_counter = {}
        for sect in partial_summary_sects:
            if sect not in partial_summary_sects_counter:
                partial_summary_sects_counter[sect] = 1
            else:
                partial_summary_sects_counter[sect] += 1

        # try:
        #     third_term = ((partial_summary_sects_counter[sent_section] + 1) / 10) * (1/section_contrib) * beta
        #     third_term = ((partial_summary_sects_counter[sent_section]) / len(partial_summary_idx)) * (sent_scores[idx] * sent_sectwise_rg[idx])
        # except:
        #     third_term = 0
        for sect in partial_summary_sects_counter:
            if sect == sent_section:
                partial_summary_sects_counter[sect] = ((partial_summary_sects_counter[sect] + 1) / 10) * (1/section_contrib) * beta
                # partial_summary_sects_counter[sect] = ((partial_summary_sects_counter[sect] + 1) / len(partial_summary_idx))
            else:
                partial_summary_sects_counter[sect] = ((partial_summary_sects_counter[sect]) / 10) * (1/section_contrib) * beta
                # partial_summary_sects_counter[sect] = (partial_summary_sects_counter[sect] / len(partial_summary_idx))

        third_term = max(partial_summary_sects_counter.values())
        # print(co1, co2, co3)
        mmr_sent = co1 * first_term - co2 * second_term - co3 * third_term
        # mmr_sent = co1 * first_term - co3 * third_term
        # mmr_sent = co1 * first_term - co3 * third_term
        current_mmrs.append(mmr_sent)
        # deductions.append((co2 * second_term,co3 * third_term))
        deductions.append((co3 * third_term))
        # deductions.append((co2 * second_term))
    return current_mmrs, deductions

def intersection(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def _get_precision_(sent_true_labels, summary_idx):

    oracle_cout = sum(sent_true_labels)
    if oracle_cout==0:
        return 0

    # oracle_cout = oracle_cout if oracle_cout > 0 else 1
    pos = 0
    neg = 0
    for idx in summary_idx:
        if sent_true_labels[idx] == 0:
            neg+=1
        else:
            pos+=1

    if pos ==0:
        return 0
    prec= pos / len(summary_idx)

    # recall --how many relevants are retrieved?
    recall = pos / int(oracle_cout)

    try:
        F = (2*prec*recall) / (prec+recall)
        return F

    except Exception:
        traceback.print_exc()
        os._exit(2)



def _multi_mmr(params):
    p_id, sent_scores, paper_srcs, paper_tgt, sent_sects_whole_true, source_sent_encodings, sent_sects_whole_true, section_textual, sent_true_labels, sent_sectwise_rg, co1, co2, co3, cos, PRED_LEN, bs_selected_ids = params

    section_textual = np.array(section_textual)
    sent_sectwise_rg = np.array(sent_sectwise_rg)
    sent_true_labels = np.array(sent_true_labels)
    sent_scores = np.array(sent_scores)
    paper_srcs = np.array(paper_srcs)
    sent_sects_whole_true = np.array(sent_sects_whole_true)


    # keep the eligible ids by checking conditions on the sentences
    keep_ids = [idx for idx, s in enumerate(paper_srcs) if len(
        paper_srcs[idx].replace('.', '').replace(',', '').replace('(', '').replace(')', '').replace('-',
                                                                                                    '').replace(
            ':', '').replace(';', '').replace('*', '').split()) > 5 and len(
        paper_srcs[idx].replace('.', '').replace(',', '').replace('(', '').replace(')', '').replace('-',
                                                                                                    '').replace(
            ':', '').replace(';', '').replace('*', '').split()) < 100]

    # filter out ids based on the scores (top50 scores) --> should update keep ids

    source_sents = paper_srcs[keep_ids]
    sent_scores = sent_scores[keep_ids]
    sent_sects_whole_true = sent_sects_whole_true[keep_ids]
    section_textual = section_textual[keep_ids]
    sent_sectwise_rg = sent_sectwise_rg[keep_ids]
    source_sent_encodings = source_sent_encodings[keep_ids]
    sent_true_labels = sent_true_labels[keep_ids]
    paper_srcs_orc = paper_srcs[keep_ids]



    src_with_sections = [(s, sec) for s, sec in zip(source_sents, section_textual)]
    sect_lens = {}
    cur_sect = ''
    cur_sect_len = 0
    try:
        for idx, sent in enumerate(src_with_sections):


            if idx == len(src_with_sections)-1 and len(sect_lens) == 0:
                sect_lens[cur_sect] = cur_sect_len

            if idx==0:
                cur_sect=sent[1]
                cur_sect_len+=1

            if idx>0 and sent[1]==cur_sect:
                cur_sect_len+=1
                if idx==len(src_with_sections)-1:
                    sect_lens[cur_sect] = cur_sect_len
                    break

            if sent[1] != cur_sect:
                sect_lens[cur_sect] = cur_sect_len
                cur_sect=sent[1]
                cur_sect_len = 1
                if idx==len(src_with_sections)-1:
                    sect_lens[cur_sect] = cur_sect_len
                    break


        sent_sect_len = [sect_lens[sec] for sec in section_textual]
    except Exception:
        # import pdb;pdb.set_trace()
        traceback.print_exc()

    oracle_sects = [s for idx, s in enumerate(section_textual) if sent_true_labels[idx] == 1]

    sent_scores = np.asarray([s - 1.00 for s in sent_scores])

    # sent_scores = [s / np.max(sent_scores) for s in sent_scores]
    # sent_scores = np.array(sent_scores)

    top_score_ids = np.argsort(-sent_scores, 0)

    if len(sent_sects_whole_true) == 0:
        return

    pruned = False
    if pruned:
        # keep top 100 sents
        top_score_ids = top_score_ids[:100]
        top_score_ids = [sorted(top_score_ids[:100]).index(s) for s in top_score_ids]
        # only keep if it's above threshold
        # top_score_ids = [t for t in top_score_ids if sent_scores[t] > 0.01]

        sent_scores = sent_scores[top_score_ids]
        sent_sects_whole_true = sent_sects_whole_true[top_score_ids]
        section_textual = section_textual[top_score_ids]
        source_sent_encodings = source_sent_encodings[top_score_ids]
        source_sents = np.asarray(source_sents)
        source_sents = source_sents[top_score_ids]

    section_sent_contrib = [((s / sum(set(sent_sectwise_rg))) + 0.001) for s in sent_sectwise_rg]

    summary = []
    summary_sects = []


    score_dist = {}

    score_dist[str(0) + '-MRR'] = [0 for _ in range(len(source_sents))]
    score_dist[str(0) + '-mul'] = sent_scores

    # pick the first top-score sentence to start with...
    summary_sects += [section_textual[top_score_ids[0]]]
    summary += [source_sents[top_score_ids[0]]]
    sent_scores_model = sent_scores

    summary_idx = [top_score_ids[0]]

    # augment the summary with MMR until the pred length reach.
    MMR_score_dyn = {}
    MMR_score_dyn[top_score_ids[0]] = 0
    oracle_idx = [idx for idx, s in enumerate(source_sents) if sent_true_labels[idx] == 1]

    for summary_num in range(1, PRED_LEN):
        if len(source_sents) <= len(summary):
            break
        MMRs_score, deductions = cal_mmr(source_sents, summary, summary_idx, source_sent_encodings,
                                         summary_sects, sent_scores_model, section_textual, section_sent_contrib, co1,
                                         co2, co3, cos)


        # sl = list(sent_scores_model)
        # sl = [s[0] for s in sent_scores_model]
        # import pdb;pdb.set_trace()
        # sent_score_sclaed = np.array([s / np.max(sent_scores_model) for s in sent_scores_model])
        # MMRs_score, deductions = cal_mmr(source_sents, summary, summary_idx, source_sent_encodings,
        #                              summary_sects, sent_scores_model, section_textual, section_sent_contrib, co1,
        #                              co2, co3, cos)

        # MMRs_score, deductions = cal_mmr_unified(source_sents, summary, summary_idx, source_sent_encodings,
        #                              summary_sects, sent_scores_model, section_textual, section_sent_contrib, sent_sect_len, co1,
        #                              co2, co3, cos)

        # sl = list(sent_scores_model)
        sent_scores = np.multiply(sent_scores_model, np.array(MMRs_score))
        score_dist[str(summary_num) + '-MRR'] = MMRs_score
        score_dist[str(summary_num) + '-mul'] = sent_scores

        # sent_scores = np.array(MMRs_score)

        # autment summary with the updated sent scores
        top_score_ids = np.argsort(-sent_scores, 0)
        # if sent_scores[top_score_ids[0]] > 0:
            # break
        # else:
        summary_idx += [top_score_ids[0]]
        MMR_score_dyn[top_score_ids[0]] = MMRs_score[top_score_ids[0]]
        summary += [source_sents[top_score_ids[0]]]
        summary_sects += [section_textual[top_score_ids[0]]]

    visualize_heatmap(data=[np.array(d) for d in score_dist.values()], p_id=p_id, y_labels=list(score_dist.keys()),
                      x_labels=[i for i in range(len(source_sents))], oracle_idxs=oracle_idx,
                      summary_idx=summary_idx, sect_lens= sect_lens, bs_selected_ids=bs_selected_ids)

    summary = [s for s in sorted(zip(summary_idx, summary, summary_sects), key=lambda x: x[0])]
    prec = _get_precision_(sent_true_labels, summary_idx)
    summary_sects = [s[2] for s in sorted(zip(summary_idx, summary, summary_sects), key=lambda x: x[0])]

    pred_dict = {}
    for s in summary:
        pred_dict[s[0]] = [s[1], sent_scores_model[s[0]], MMR_score_dyn[s[0]], s[2]]
    pred_dict2 = {}
    for key,val in pred_dict.items():
        pred_dict2[str(key)] = val
    pred_dict = pred_dict2
    print(str(sorted(summary_idx)))

    # with open('mmr_outputs/sections_multi_{}.csv'.format(co3), 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(
    #         [str(p_id),
    #          # str(' '.join([s[0] for s in pred_dict])),
    #          json.dumps(pred_dict, indent=3),
    #          str(sorted(summary_idx)),
    #          str(sorted(oracle_idx)),
    #          str(summary_sects),
    #          str(oracle_sects),
    #          intersection(Counter(summary_sects), Counter(oracle_sects)),
    #          evaluate_rouge([' '.join([val[0] for s, val in pred_dict.items()])], [paper_tgt])[0],
    #          evaluate_rouge([' '.join([val[0] for s, val in pred_dict.items()])], [paper_tgt])[1],
    #          evaluate_rouge([' '.join([val[0] for s, val in pred_dict.items()])], [paper_tgt])[2],
    #          prec])
    return [val[0] for s, val in pred_dict.items()], paper_tgt, p_id


def _bertsum_baseline_1(params):
    # Set model in validating mode.
    def _get_ngrams(n, text):
        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i:i + n]))
        return ngram_set

    def _block_tri(c, p):
        tri_c = _get_ngrams(3, c.split())
        for s in p:
            tri_s = _get_ngrams(3, s.split())
            if len(tri_c.intersection(tri_s)) > 0:
                return True
        return False
    p_id, sent_scores, paper_srcs, paper_tgt, sent_sects_whole_true, source_sent_encodings, sent_sects_whole_true, section_textual, sent_true_labels, sent_sectwise_rg, co1, co2, co3, cos, PRED_LENGTH = params


    section_textual = np.array(section_textual)
    sent_sectwise_rg = np.array(sent_sectwise_rg)
    sent_true_labels = np.array(sent_true_labels)
    sent_scores = np.array(sent_scores)
    paper_srcs = np.array(paper_srcs)
    sent_sects_whole_true = np.array(sent_sects_whole_true)

    try:
        keep_ids = [idx for idx, s in enumerate(paper_srcs) if len(
            paper_srcs[idx].replace('.', '').replace(',', '').replace('(', '').replace(')', '').replace('-',
                                                                                                        '').replace(
                ':', '').replace(';', '').replace('*', '').split()) > 5 and len(
            paper_srcs[idx].replace('.', '').replace(',', '').replace('(', '').replace(')', '').replace('-',
                                                                                                        '').replace(
                ':', '').replace(';', '').replace('*', '').split()) < 100]

        section_textual = section_textual[keep_ids]
        paper_srcs = paper_srcs[keep_ids]
        sent_scores = sent_scores[keep_ids]
        sent_sects_whole_true = sent_sects_whole_true[keep_ids]
        sent_true_labels = sent_true_labels[keep_ids]

        sent_scores = np.asarray([s - 1.00 for s in sent_scores])

        top_sent_indexes = np.argsort(-sent_scores, 0)

        oracle_sects = [s for idx, s in enumerate(section_textual) if sent_true_labels[idx] == 1]
        oracle_idx = [idx for idx, s in enumerate(paper_srcs) if sent_true_labels[idx] == 1]
        lenorc = len(oracle_sects)
        # import pdb;pdb.set_trace()
        _pred = []
        # import pdb;pdb.set_trace()
        for j in top_sent_indexes:
            if (j >= len(paper_srcs)):
                continue
            candidate = paper_srcs[j].strip()
            if True:
            # if (not _block_tri(candidate, [s[0] for s in _pred])):
                _pred.append((candidate, j, section_textual[j], sent_scores[j]))

            if (len(_pred) == PRED_LENGTH):
                break

        selected_idx = [s[1] for s in _pred]
        print(str(sorted(selected_idx)))
        print(str(sorted(oracle_idx)))
        sections = [s[2] for s in sorted(_pred,  key= lambda x : x[1])]
        prec = _get_precision_(sent_true_labels, selected_idx)

        pred_dict={}
        for s in sorted(_pred,  key= lambda x : x[1]):
            pred_dict[str(s[1])] = [s[0], s[3], s[2]]
        pretty_json = json.dumps(pred_dict, indent=2)
        # with open('mmr_outputs/sections_bertsum.csv', 'a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([str(p_id),
        #                      # str(' '.join([('[%d: %s (%4.4f) ––%s' % (s[1], s[0], s[3], s[2])) for s in sorted(_pred,  key= lambda x : x[1])])),
        #                      pretty_json,
        #                      str(sorted(selected_idx)),
        #                      str(sorted(oracle_idx)),
        #                      str(sections),
        #                      str(oracle_sects),
        #                      intersection(Counter(sections), Counter(oracle_sects)),
        #                      evaluate_rouge([' '.join([s[0] for s in sorted(_pred,  key= lambda x : x[1])])], [paper_tgt])[0],
        #                      evaluate_rouge([' '.join([s[0] for s in sorted(_pred,  key= lambda x : x[1])])], [paper_tgt])[1],
        #                      evaluate_rouge([' '.join([s[0] for s in sorted(_pred,  key= lambda x : x[1])])], [paper_tgt])[2],
        #                      prec]
        #                     )
        return [s[0] for s in sorted(_pred,  key= lambda x : x[1])], paper_tgt, p_id, sorted(selected_idx)
    except Exception:
        traceback.print_exc()
        print('here')
        import pdb;
        pdb.set_trace()
        # return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-co1", type=float)
    parser.add_argument("-co2", type=float)
    parser.add_argument("-co3", type=float)
    parser.add_argument("-cos", type=str2bool, nargs='?', const=True, default=False)

    args = parser.parse_args()

    # with open('mmr_outputs/sections_bertsum.csv', 'a', newline='') as file:
    with open('mmr_outputs/sections_multi_{}.csv'.format(args.co3), 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["paper_id", "our Pred", "our pred_idx", "oracle pred idx" "our Section", "Oracle sects", "our Overlap", "our RG-1", "our RG-2", "our RG-L", "our F"])
        # writer.writerow(["paper_id", "bs Pred", "bs pred_idx", "oracle pred idx" "bs Section", "Oracle sects", "bs Overlap", "bs RG-1", "bs RG-2", "bs RG-L", "bs F"])

    # saved_list = pickle.load(open("save_list_arxiv_main_test.p", "rb"))
    saved_list = pickle.load(open("save_list_arxiv_long_test_rouge.p", "rb"))
    # saved_list = pickle.load(open("save_list_pubmed_long_test_rouge.p", "rb"))
    # saved_list = pickle.load(open("save_list_arxiv_long_test_rouge.p", "rb"))
    preds = {}
    golds = {}
    preds1 = {}
    golds1 = {}
    a_lst = []
    for s, val in tqdm(saved_list.items(), total=len(saved_list)):
        if '1610.09653' in val[0]:
            val = val + (args.co1, args.co2, args.co3, args.cos, 10)
            d = _bertsum_baseline_1((val))
            preds[d[2]] = d[0]
            golds[d[2]] = d[1]
            val = val + (d[3],)
            d = _multi_mmr((val))
            preds1[d[2]] = d[0]
            golds1[d[2]] = d[1]

    print(f'Calculating RG scores for {len(preds)} papers...')
    r1, r2, rl = _report_rouge([' '.join(p) for p in preds.values()], golds.values())

    print(f'Calculating RG scores for {len(preds1)} papers...')
    r1, r2, rl = _report_rouge([' '.join(p) for p in preds1.values()], golds1.values())