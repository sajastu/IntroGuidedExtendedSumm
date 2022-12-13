import argparse
import csv
import json
import math
import os
import pickle
import traceback
from collections import Counter, OrderedDict
from multiprocessing.pool import Pool

import matplotlib.pyplot as plt
import numpy as np;
import torch
from sentence_transformers import util
from tqdm import tqdm

from utils.rouge_score import evaluate_rouge, evaluate_rouge_avg
from matplotlib.patches import Rectangle

np.random.seed(0)
import seaborn as sns;

sns.set_theme()


def visualize_heatmap(data, p_id, y_labels, x_labels, oracle_idxs, summary_idx, sect_lens, bs_selected_ids, filename):
    x_labels_tmp = []
    for x in x_labels:
        if x in oracle_idxs and x in summary_idx:
            x_labels_tmp.append("--*" +str(x) + "*--")
        elif x in oracle_idxs:
            x_labels_tmp.append("*"+str(x) + "*")

        elif x in summary_idx:
            x_labels_tmp.append("--" +str(x) + "--")
        else:
            x_labels_tmp.append(str(x))
    sns.set(font_scale=6.5)

    plt.figure(figsize=(200, 12))
    data = np.array(data)

    data_tmp = list()

    dtemp=[]
    x_labels_tmp_cons = []
    dataconstrained= []

    for idx, dd in enumerate(data):
        if dd >= 0.15 or idx in oracle_idxs:
            dtemp.append("%.2f"%dd)
            x_labels_tmp_cons.append(x_labels_tmp[idx])
            dataconstrained.append(data[idx])
        # else:
        #     import pdb;
        #     pdb.set_trace()
        #
        #     x_labels_tmp = x_labels_tmp[:idx] + x_labels_tmp[idx+1:]
        #
        #     data = np.concatenate((data[:idx], data[idx+1:]))

    data_tmp.append(dtemp)

    # mask = np.zeros_like(data)
    # mask[data<0.1] = True
    data_tmp = np.array(data_tmp)
    x_labels_tmp_cons = np.array(x_labels_tmp_cons)
    dataconstrained = np.array(dataconstrained)
    # ss = sns.color_palette("light:b", as_cmap=True)
    ss = sns.cubehelix_palette(as_cmap=True)
    ss = sns.color_palette("light:b", as_cmap=True)


    # ax = sns.heatmap(data[None, :], linewidths=.9, cmap="YlGnBu", annot=data_tmp, fmt="", xticklabels=x_labels_tmp,yticklabels=y_labels, mask=mask[None, :])
    ax = sns.heatmap(dataconstrained[None, :], linewidths=.5, cmap=ss, annot=False, fmt="",annot_kws={'rotation': 90}, xticklabels=[x.replace('--','').replace('--*','') for x in x_labels_tmp_cons],yticklabels=y_labels)
    # ax = sns.heatmap(data, linewidths=.5, cmap="YlGnBu", annot=data_tmp, fmt="", xticklabels=x_labels_tmp,yticklabels=y_labels)

    # sectLens = list(sect_lens.values())


    # vLines = [sum(sectLens[:idx+1]) for idx,len in enumerate(sectLens)]
    # hLines = [y_idx for y_idx, label in enumerate(y_labels) if y_idx%2==0]

    # ax.vlines(vLines, *ax.get_ylim())
    # ax.hlines(hLines, *ax.get_xlim())
    # ax = g.ax_heatmap

    for idx, x in enumerate(x_labels_tmp_cons):
        if "--" in x:
            ax.add_patch(Rectangle((idx, 0), 1, 1, edgecolor='black', lw=12))
    ax.set_xlabel("Source sentence number")
    plt.savefig("mmr_figs/"+ p_id + '-' + filename.split('-')[-1]+ '.pdf', bbox_inches='tight', format='pdf', pad_inches=1)



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
                    sent_scores, sent_sects_whole_true, section_contributions, sent_sect_len, co1, co2, co3, cos,
                    beta=0.1):
    current_mmrs = []
    for idx, (sent, score, contr) in enumerate(zip(source_sents, sent_scores, section_contributions)):
        if idx in partial_summary_idx:
            current_mmrs.append(0)
            continue
        else:
            # current_mmrs.append(1.0)
            current_mmrs.append(score * contr * sent_sect_len[idx] / len(source_sents))
    return current_mmrs, []


# def scale_between(scaledMin, scaledMax):


def cal_mmr(source_sents, partial_summary, partial_summary_idx, partial_summary_sects,
            sent_scores, sent_sects_whole_true, section_contributions, sent_sectwise_rg, section_textual, sect_lens,
            co1, co2, co3, cos, beta=0.1, sentence_encodings=None, PRED_LEN=0, sect_encodings=None):
    current_mmrs = []
    deductions = []
    # MMR formula:

    ## MMR = argmax (\alpha Sim(si, D) - \beta max SimSent(si, sj) - \theta max SimSect(sj, sj))
    for idx, sent in enumerate(source_sents):
        if idx in partial_summary_idx:
            current_mmrs.append(-1000)
            continue

        section_contrib = section_contributions[idx]
        # print(section_contrib)
        sent_txt = sent
        sent_section = sent_sects_whole_true[idx]
        # sent_section = section_textual[idx]

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

        # for sect in partial_summary_sects_counter:
        #     if sect == sent_section:
        #         # partial_summary_sects_counter[sect] = ((partial_summary_sects_counter[sect] + 1) / 15) * (1/section_contrib) * beta
        #         partial_summary_sects_counter[sect] = ((partial_summary_sects_counter[sect] + 1) / len(partial_summary))
        #     else:
        #         # partial_summary_sects_counter[sect] = ((partial_summary_sects_counter[sect]) / 15) * (1/section_contrib) * beta
        #         partial_summary_sects_counter[sect] = ((partial_summary_sects_counter[sect] / len(partial_summary)))
        #

        try:
            third_term = (partial_summary_sects_counter[sent_section]) / PRED_LEN
        except:
            third_term = 0


        # sect_cosine_sims = []
        # for sect in partial_summary_sects:
        #     sect_cosine_sims.append(util.pytorch_cos_sim(sect_encodings[sect], sentence_encodings[idx])[0][0].item())
        # third_term = max(sect_cosine_sims)

        mmr_sent = co1 * first_term - co2 * second_term - co3 * third_term
        current_mmrs.append(mmr_sent)
        deductions.append((co3 * third_term))
    return current_mmrs, deductions


def intersection(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0) ** 2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0) ** 2 for k in terms))
    return dotprod / (magA * magB)


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def _get_precision_(sent_true_labels, summary_idx):
    oracle_cout = sum(sent_true_labels)
    if oracle_cout == 0:
        return 0, 0, 0

    # oracle_cout = oracle_cout if oracle_cout > 0 else 1
    pos = 0
    neg = 0
    for idx in summary_idx:
        if sent_true_labels[idx] == 0:
            neg += 1
        else:
            pos += 1

    if pos == 0:
        return 0, 0, 0
    prec = pos / len(summary_idx)

    # recall --how many relevants are retrieved?
    recall = pos / int(oracle_cout)

    try:
        F = (2 * prec * recall) / (prec + recall)
        return F, prec, recall

    except Exception:
        traceback.print_exc()
        os._exit(2)


def modulate_sent_scores(sent_scores, sent_sectwise_rg, sent_sect_textual, sect_len):
    out_sent_scores = list()
    for idx, sent_score in enumerate(sent_scores):
        # out_sent_scores.append(sent_score * sent_sectwise_rg[idx] / sect_len[sent_sect_textual[idx]])
        out_sent_scores.append(sent_score * sent_sectwise_rg[idx])
    return out_sent_scores


def _get_sect_encodings(sent_sects_whole_true, source_sent_encodings):
    sect_encodings={}
    for sect, enc in zip(sent_sects_whole_true, source_sent_encodings):
        if sect not in sect_encodings.keys():
            sect_encodings[sect] = [enc]
        else:
            sect_encodings[sect].append(enc)

    for sect, encs in sect_encodings.items():
        sect_encodings[sect] = np.mean([e for e in np.array(encs)], axis=0)

    return sect_encodings


def _mmr(params):
    p_id, sent_scores, paper_srcs, paper_tgt, sent_sects_whole_true, sent_sects_whole_true, section_textual, sent_true_labels, sent_sectwise_rg, source_sent_encodings, co1, co2, co3, cos, PRED_LEN, filename = params
    section_textual = np.array(section_textual)
    sent_sectwise_rg = np.array(sent_sectwise_rg)
    sent_true_labels = np.array(sent_true_labels)
    sent_scores = np.array(sent_scores)
    paper_srcs = np.array(paper_srcs)
    sent_sects_whole_true = np.array(sent_sects_whole_true)
    source_sent_encodings = np.array(source_sent_encodings)


    # sect_encodings = _get_sect_encodings(sent_sects_whole_true, source_sent_encodings)
    # sect_encodings = _get_sect_encodings(section_textual, source_sent_encodings)

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

    # source_sent_encodings = source_sent_encodings[keep_ids]
    sect_encodings = None

    sent_true_labels = sent_true_labels[keep_ids]

    src_with_sections = [(s, sec) for s, sec in zip(source_sents, section_textual)]
    sect_lens = {}
    cur_sect = ''
    cur_sect_len = 0
    try:
        for idx, sent in enumerate(src_with_sections):

            if idx == len(src_with_sections) - 1 and len(sect_lens) == 0:
                sect_lens[cur_sect] = cur_sect_len

            if idx == 0:
                cur_sect = sent[1]
                cur_sect_len += 1

            if idx > 0 and sent[1] == cur_sect:
                cur_sect_len += 1
                if idx == len(src_with_sections) - 1:
                    sect_lens[cur_sect] = cur_sect_len
                    break

            if sent[1] != cur_sect:
                sect_lens[cur_sect] = cur_sect_len
                cur_sect = sent[1]
                cur_sect_len = 1
                if idx == len(src_with_sections) - 1:
                    sect_lens[cur_sect] = cur_sect_len
                    break

        sent_sect_len = [sect_lens[sec] for sec in section_textual]
    except Exception:
        # import pdb;pdb.set_trace()
        traceback.print_exc()

    # oracle_sects = [s for idx, s in enumerate(section_textual) if sent_true_labels[idx] == 1]
    oracle_sects = [s for idx, s in enumerate(sent_sects_whole_true) if sent_true_labels[idx] == 1]

    sent_scores = np.asarray([s - 1.00 for s in sent_scores])

    # sent_scores = [s / np.max(sent_scores) for s in sent_scores]
    # sent_scores = np.array(sent_scores)

    top_score_ids = np.argsort(-sent_scores, 0)

    if len(sent_sects_whole_true) == 0:
        return

    section_sent_contrib = [((s / sum(set(sent_sectwise_rg))) + 0.00001) for s in sent_sectwise_rg]
    summary = []
    summary_sects = []

    score_dist = {}

    score_dist[str(0) + '-MRR'] = [0 for _ in range(len(source_sents))]
    score_dist[str(0) + '-mul'] = sent_scores

    #####################
    #####################
    #####################

    section_mult = True
    if not section_mult:
        # import pdb;pdb.set_trace()
        sent_scores = modulate_sent_scores(sent_scores, sent_sectwise_rg, section_textual, sect_lens)
        # import pdb;pdb.set_trace()

        sent_scores = np.asarray([s for s in sent_scores])

        top_sent_indexes = np.argsort(-sent_scores, 0)

        oracle_sects = [s for idx, s in enumerate(section_textual) if sent_true_labels[idx] == 1]
        oracle_idx = [idx for idx, s in enumerate(source_sents) if sent_true_labels[idx] == 1]
        # import pdb;pdb.set_trace()
        _pred = []
        # import pdb;pdb.set_trace()
        for j in top_sent_indexes:
            if (j >= len(source_sents)):
                continue
            candidate = source_sents[j].strip()
            if True:
                # if (not _block_tri(candidate, [s[0] for s in _pred])):
                _pred.append((candidate, j, section_textual[j], sent_scores[j]))

            if (len(_pred) == PRED_LEN):
                break

        selected_idx = [s[1] for s in _pred]
        # print(str(sorted(selected_idx)))
        # print(str(sorted(oracle_idx)))
        sections = [s[2] for s in sorted(_pred, key=lambda x: x[1])]
        F, prec, rec = _get_precision_(sent_true_labels, selected_idx)

        pred_dict = {}
        for s in sorted(_pred, key=lambda x: x[1]):
            pred_dict[str(s[1])] = [s[0], s[3], str(s[2])]
        pretty_json = json.dumps(pred_dict, indent=2)

        r1 = evaluate_rouge([' '.join([s[0] for s in sorted(_pred, key=lambda x: x[1])])], [paper_tgt])[0]
        r2 = evaluate_rouge([' '.join([s[0] for s in sorted(_pred, key=lambda x: x[1])])], [paper_tgt])[1]
        rl = evaluate_rouge([' '.join([s[0] for s in sorted(_pred, key=lambda x: x[1])])], [paper_tgt])[2]

        if len(filename) > 0:
            with open('mmr_outputs/' + filename + '.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([str(p_id),
                                 pretty_json,
                                 str(sorted(selected_idx)),
                                 str(sorted(oracle_idx)),
                                 str(sections),
                                 str(oracle_sects),
                                 intersection(Counter(sections), Counter(oracle_sects)),
                                 r1,
                                 r2,
                                 rl,
                                 F]
                                )
        return [s[0] for s in sorted(_pred, key=lambda x: x[1])], paper_tgt, p_id, r1, r2, rl

    else:
        # pick the first top-score sentence to start with...
        # summary_sects += [section_textual[top_score_ids[0]]]
        summary_sects += [sent_sects_whole_true[top_score_ids[0]]]
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
            MMRs_score, deductions = cal_mmr(source_sents, summary, summary_idx,
                                             summary_sects, sent_scores_model,
                                             sent_sects_whole_true, section_sent_contrib,
                                             sent_sectwise_rg, section_textual, sect_lens,
                                             co1, co2, co3, cos, PRED_LEN=PRED_LEN, sect_encodings=sect_encodings, sentence_encodings=source_sent_encodings)

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
            # summary_sects += [section_textual[top_score_ids[0]]]
            summary_sects += [sent_sects_whole_true[top_score_ids[0]]]

        summary = [s for s in zip(summary_idx, summary, summary_sects)]
        F, p, r = _get_precision_(sent_true_labels, summary_idx)
        summary_sects = [s[2] for s in sorted(zip(summary_idx, summary, summary_sects), key=lambda x: x[0])]

        pred_dict = {}
        for s in summary:
            pred_dict[s[0]] = {"sentence": s[1], "sent_score": sent_scores_model[s[0]], "MMR":MMR_score_dyn[s[0]], "section": str(s[2]), "true_label": str(sent_true_labels[s[0]])}
        pred_dict2 = {}
        for key, val in pred_dict.items():
            pred_dict2[str(key)] = val
        pred_dict = pred_dict2
        r1 = evaluate_rouge([' '.join([val["sentence"] for s, val in sorted(pred_dict.items(), key= lambda x:int(x[0]))])], [paper_tgt])[0]
        r2 = evaluate_rouge([' '.join([val["sentence"] for s, val in sorted( pred_dict.items(), key= lambda x:int(x[0]))])], [paper_tgt])[1]
        rl = evaluate_rouge([' '.join([val["sentence"] for s, val in sorted( pred_dict.items() , key= lambda x:int(x[0]))])], [paper_tgt])[2]

        if len(filename) > 0:
            with open('mmr_outputs/' + filename + '.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(
                    [str(p_id),
                     # str(' '.join([s[0] for s in pred_dict])),
                     json.dumps(pred_dict, indent=2),
                     str(sorted(summary_idx)),
                     str(sorted(oracle_idx)),
                     str([str(s) for s in summary_sects]),
                     str([str(o) for o in oracle_sects]),
                     intersection(Counter(summary_sects), Counter(oracle_sects)),
                     r1,
                     r2,
                     rl,
                     F])
        return [val["sentence"] for s, val in pred_dict.items()], paper_tgt, p_id,r1, r2, rl

class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first encountered'

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)

def _base(params):

    p_id, sent_scores, paper_srcs, paper_tgt, sent_sects_whole_true, sent_sects_whole_true, section_textual, sent_true_labels, sent_sectwise_rg, source_sent_encodings, co1, co2, co3, cos, PRED_LEN, filename = params
    section_textual = np.array(section_textual)
    sent_sectwise_rg = np.array(sent_sectwise_rg)
    sent_true_labels = np.array(sent_true_labels)
    sent_scores = np.array(sent_scores)
    paper_srcs = np.array(paper_srcs)
    sent_sects_whole_true = np.array(sent_sects_whole_true)

    # sent_scores = np.asarray([s - 1.00 for s in sent_scores])

    # top_sent_indexes = np.argsort(-sent_scores, 0)

    # sl = list(sent_scores_model)
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

        # oracle_sects = [s for idx, s in enumerate(section_textual) if sent_true_labels[idx] == 1]
        oracle_sects = [s for idx, s in enumerate(sent_sects_whole_true) if sent_true_labels[idx] == 1]
        # import pdb;pdb.set_trace()
        _pred = []
        # import pdb;pdb.set_trace()
        for j in top_sent_indexes:
            if (j >= len(paper_srcs)):
                continue
            candidate = paper_srcs[j].strip()
            if True:
                # if (not _block_tri(candidate, [s[0] for s in _pred])):
                _pred.append((candidate, j, sent_sects_whole_true[j], sent_scores[j]))
            if (len(_pred) == PRED_LEN):
                break

        selected_idx = [s[1] for s in _pred]
        selected_idx = sorted(selected_idx)
        section_counter = OrderedCounter(section_textual)

        selected_idx_disply = []

        for s in selected_idx:
            selected_idx_disply.append(str(s) + ' [{}]'.format(section_textual[s]))


        sections = [s[2] for s in sorted(_pred, key=lambda x: x[1])]
        F, prec, rec = _get_precision_(sent_true_labels, selected_idx)
        # print(F)
        pred_dict = {}
        # for s in sorted(_pred, key=lambda x: x[1]):
        for s in _pred:
            pred_dict[str(s[1])] = {"sentence": s[0], "score":s[3], "section":str(s[2]), "true_label": str(sent_true_labels[s[1]]) }


        pretty_json = json.dumps(pred_dict, indent=2)
        r1 = \
        evaluate_rouge([' '.join([val["sentence"] for s, val in sorted(pred_dict.items(), key=lambda x: int(x[0]))])],
                       [paper_tgt])[0]
        r2 = \
        evaluate_rouge([' '.join([val["sentence"] for s, val in sorted(pred_dict.items(), key=lambda x: int(x[0]))])],
                       [paper_tgt])[1]
        rl = \
        evaluate_rouge([' '.join([val["sentence"] for s, val in sorted(pred_dict.items(), key=lambda x: int(x[0]))])],
                       [paper_tgt])[2]

        # oracle_idx = [str(idx) + ' (' + str(sent_scores[idx]) + ', ' + str(sent_sects_whole_true[idx]) + ')' for idx, s in enumerate(paper_srcs) if sent_true_labels[idx] == 1]

        oracle_idx = sorted([idx for idx, s in enumerate(paper_srcs) if sent_true_labels[idx] == 1])
        oracle_idx_display = []
        for o in oracle_idx:
            oracle_idx_display.append(str(o) + " [{}]".format(section_textual[o]))

        # pred = ' '.join([s for s in paper_srcs[sorted(selected_idx)]])
        # oracle = ' '.join([s for s in paper_srcs[sorted(oracle_idx)]])

        pred = ''
        oracle = ''

        for s in  sorted(selected_idx):
            pred += paper_srcs[s] + "[{}]".format(section_textual[s])
            pred += ' '

        for s in  sorted(oracle_idx):
            oracle += paper_srcs[s] + "[{}]".format(section_textual[s])
            oracle += ' '

        instance = {'p_id':p_id, 'oracle':oracle,  'pred':pred, 'gold':paper_tgt}

        # visualize_heatmap(data=np.array(sent_scores), p_id=p_id, y_labels=[],
        #                   x_labels=[i for i in range(len(paper_srcs))], oracle_idxs=oracle_idx,
        #                   summary_idx=sorted(selected_idx), sect_lens=[], bs_selected_ids=[], filename=filename)

        if len(filename) > 0:
            with open('anal_outputs/' + filename + '.samples.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([str(p_id),
                                 # str(' '.join([('[%d: %s (%4.4f) ––%s' % (s[1], s[0], s[3], s[2])) for s in sorted(_pred,  key= lambda x : x[1])])),
                                 pretty_json,
                                 str(selected_idx_disply),
                                 str(oracle_idx_display),
                                 str([str(s) for s in sections]),
                                 str(oracle_sects),
                                 # intersection(Counter(sections), Counter(oracle_sects)),
                                 0,
                                 r1,
                                 r2,
                                 rl,
                                 F,
                                 pred,
                                 oracle,
                                 paper_tgt]
                                )

        return [s[0] for s in sorted(_pred, key=lambda x: x[1])], paper_tgt, p_id, r1, r2, rl, instance.copy()
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
    parser.add_argument("-set", type=str)
    parser.add_argument("-pred_len", type=int)
    parser.add_argument("-method", type=str)
    parser.add_argument("-saved_list", type=str)
    parser.add_argument("-cos", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-end", type=str2bool, nargs='?', const=True, default=False)

    args = parser.parse_args()


    ########### PARAMS dataset-specific ###########
    ########### ########### ########### ###########

    PRED_LEN = args.pred_len

    # filename = ""

    filename = "results_" + args.saved_list.replace("/disk1/sajad/save_lists/", "")

    ########### ########### ########### ###########
    ########### ########### ########### ###########
    if len(filename) > 0:
        with open('anal_outputs/' + filename + '.samples.csv', 'w', newline='') as file:
            # with open('mmr_outputs/sections_multi_{}_arxivmain.csv'.format(args.co3), 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ["paper_id", "Pred", "pred_idx", "oracle pred idx", "section", "Oracle sects", "Overlap",
                 "RG-1", "RG-2", "RG-L", "F", "pred_txt", "oracle_txt", "paper_tgt"]
            )

    saved_list = pickle.load(open(args.saved_list, "rb"))

    preds = {}
    golds = {}
    preds1 = {}
    golds1 = {}
    a_lst = []

    sample_papers = []
    with open("arxivL-samples.txt") as F:
        for l in F:
            sample_papers.append(l.strip())
    for s, val in tqdm(saved_list.items(), total=len(saved_list)):
        if val[0] in sample_papers:
        #     if args.cos:
        #         val = val + (sentence_encodings[val[0]], args.co1, args.co2, args.co3, args.cos, PRED_LEN, filename)
        #     else:
        #         val = val + (None, args.co1, args.co2, args.co3, args.cos, PRED_LEN, filename)
        #     a_lst.append((val))

            val = val + (None, args.co1, args.co2, args.co3, args.cos, PRED_LEN, filename)
            a_lst.append((val))

        else:
            for s in sample_papers:
                if str(s) in str(val[0]):
                    val = val + (None, args.co1, args.co2, args.co3, args.cos, PRED_LEN, filename)
                    a_lst.append((val))


    #     d = _base(a_lst[-1])
        # d = _mmr((val))

    pool = Pool(24)
    preds = {}
    golds = {}
    sent_len = []
    instances = []

    rouge_scores = {"r1": [], "r2": [], "rl": []}

    for a in a_lst:
        _base(a)

    # for d in tqdm(pool.imap(eval(args.method), a_lst), total=len(a_lst)):
    #     if d is not None:
    #         p_id = d[2]
    #         preds[p_id] = d[0]
    #         golds[p_id] = d[1]
    #         rouge_scores["r1"].append(d[3])
    #         rouge_scores["r2"].append(d[4])
    #         rouge_scores["rl"].append(d[5])
    #         instance = d[-1]
    #         instances.append(instance)
    # pool.close()
    # pool.join()

    with open('txt_out/' + filename + '.json', mode='w') as F:
        for instance in instances:
            json.dump(instance, F)
            F.write('\n')

    setting = {"mmr": f"argmax [({args.co1} term1 - {args.co2} max term2 - {args.co3} max term3)]",
               "description": f"with cosine similarity? {args.cos}; .75, .25 (first term)"}

    print(f'Calculating RG scores for {len(preds)} papers...')
    r1 = np.mean(rouge_scores["r1"])
    r2 = np.mean(rouge_scores["r2"])
    rl = np.mean(rouge_scores["rl"])

    print('-R1: {:.4f} \n-R2: {:.4f}, \n-RL: {:.4f}'.format(r1, r2, rl))

    if not args.end:
        with open('scores.txt', mode='a') as F:
            F.write('{:.4f},{:.4f},{:.4f};'.format(r1, r2, rl))
    else:
        with open('scores.txt', mode='a') as F:
            F.write('{:.4f},{:.4f},{:.4f}'.format(r1, r2, rl) + "}")
        with open('scores.txt', mode='r') as F:
            print('for %s' % args.set)
            for l in F:
                print("={" + l)
        os.remove('scores.txt')


    # r1, r2, rl = _report_rouge([' '.join(p) for p in preds.values()], golds.values())

    # MODEL = 'bertsum_results'
    # timestamp = datetime.now().strftime("%Y_%m_%d-%I_%p")
    #
    # check_path_existence("{}/{:4.4f}_{:4.4f}_{:4.4f}__{}/".format(MODEL, r1, r2, rl, timestamp))
    # with open("{}/{:4.4f}_{:4.4f}_{:4.4f}__{}/info.json".format(MODEL, r1, r2, rl, timestamp), mode='w') as F:
    #     json.dump(setting, F, indent=4)
    #
    # can_path = '{}/{:4.4f}_{:4.4f}_{:4.4f}__{}/val-arxivl.source'.format(MODEL, r1, r2, rl, timestamp)
    # gold_path = '{}/{:4.4f}_{:4.4f}_{:4.4f}__{}/val-arxivl.target'.format(MODEL, r1, r2, rl, timestamp)
    # save_pred = open(can_path, 'w')
    # save_gold = open(gold_path, 'w')
    # for id, pred in preds.items():
    #     save_pred.write(' '.join(pred).strip().replace('<q>', ' ') + '\n')
    #     save_gold.write(golds[id].replace('<q>', ' ').strip() + '\n')
