import gc
import glob
import hashlib
import json
import os
import os.path
import pickle
import re
import sys
from os.path import join as pjoin

import pandas as pd
import torch
from multiprocess import Pool
from tqdm import tqdm

from others.logging import logger
from others.tokenization import BertTokenizer, LongformerTokenizerMine
from prepro.utils import _get_word_ngrams
from utils.rouge_score import evaluate_rouge
from datetime import datetime
from uuid import uuid4
import pdb

from utils.rouge_utils import cal_rouge

nyt_remove_words = ["photo", "graph", "chart", "map", "table", "drawing"]

INTRO_KWs_STR = "[introduction, introduction and motivation, motivation, motivations, basics and motivations, introduction., [sec:intro]introduction, *introduction*, i. introduction, [sec:level1]introduction, introduction and motivation, introduction[sec:intro], [intro]introduction, introduction and main results, introduction[intro], introduction and summary, [sec:introduction]introduction, overview, 1. introduction, [sec:intro] introduction, introduction[sec:introduction], introduction and results, introduction and background, [sec1]introduction, [introduction]introduction, introduction and statement of results, introduction[introduction], introduction and overview, introduction:, [intro] introduction, [sec:1]introduction, introduction and main result, introduction[sec1], [sec:level1] introduction, motivations, outline, introductory remarks, [sec1] introduction, introduction and motivations, 1.introduction, introduction and definitions, introduction and notation, introduction and statement of the results, i.introduction, introduction[s1], [sec:level1]introduction +,  introduction., introduction[s:intro], [i]introduction, [sec:int]introduction, introduction and observations, [introduction] introduction, [sec:1] introduction, **introduction**, [seci]introduction,, **introduction**, [seci]introduction, introduction and summary of results, introduction and outline, preliminary remarks, general introduction, [sec:intr]introduction, [s1]introduction, introduction[sec_intro], introduction and statement of main results, scientific motivation, [sec:sec1]introduction, *questions*, introduction and the model, intoduction, challenges, introduction[sec-intro], introduction and result, inroduction, [sec:intro]introduction +, introdution, 1 introduction, brief summary, motivation and introduction, [1]introduction, introduction and related work, [sec:one]introduction, [section1]introduction, [sect:intro]introduction]"

INTRO_KWs_STR = "[" + str(['' + kw.strip() + '' for kw in INTRO_KWs_STR[1:-1].split(',')]) + "]"
INTRO_KWs = eval(INTRO_KWs_STR)[0]

CONC_KWs_STR = "[conclusion, conclusions, conclusion and future work, conclusions and future work, conclusion & future work, extensions, future work, related work and discussion, discussion and related work, conclusion and future directions, summary and future work, limitations and future work, future directions, conclusion and outlook, conclusions and future directions, conclusions and discussion, discussion and future directions, conclusions and discussions, conclusion and future direction, conclusions and future research, conclusion and future works, future plans, concluding remarks, conclusions and outlook, summary and outlook, final remarks, outlook, conclusion and outlook, conclusions and future work, summary and discussions, conclusion and future work, conclusions and perspectives, summary and concluding remarks, future work, conclusions., discussion and outlook, discussion & conclusions, open problems, remarks, conclusions[sec:conclusions], conclusion and perspectives, summary and future work, conclusion., summary & conclusions, closing remarks, final comments, future prospects, open questions, *conclusions*, [sec:conclusions]conclusions, conclusions and summary, comments, conclusion[sec:conclusion], perspectives, [sec:conclusion]conclusion, conclusions and future directions, summary & discussion, conclusions and remarks, conclusions and prospects, discussions and summary, future directions, conclusions and final remarks, the future, concluding comments, conclusions and open problems, summary[sec:summary], conclusions and future prospects, summary and remarks, conclusions and further work, conclusions[conclusions], [sec:summary]summary, comments and conclusions, summary and future prospects, [conclusion]conclusion, conclusion and remarks, concluding remark, further remarks, prospects, conclusion and open problems, conclusion and summary, v. conclusions, iv. conclusions,  summary and conclusions, summary and prospects, conclusions:, conclusion[conclusion], summary and final remarks, summary and future directions, summary & conclusion, [summary]summary, iv. conclusion, further questions, conclusion and future directions,  concluding remarks, further work, [conclusions]conclusions, outlook and conclusions, v. conclusion, *summary*, concluding remarks and open problems, conclusions and future works, future, [sec:conclusions] conclusions, [sec:concl]conclusions, remarks and conclusions, concluding remarks., conclusion and future works, summary., 4. conclusions, discussion and open problems, summary and comments, final remarks and conclusions, summary and conclusions., [sec:conc]conclusions, summary[summary], conclusions and open questions, [sec:conclusion]conclusions, further directions, conclusions and implications, conclusions & outlook, review, [sec:level1]conclusion, future developments, [sec:conc] conclusions, conclusions[sec:concl], conclusions and future perspectives, summary, conclusions and outlook, conclusions & discussion, [conclusions] conclusions, future research, concluding remarks and outlook, conclusions and future research, conclusion & outlook, discussion and future directions, conclusions[sec:conc], summary & outlook, vi. conclusions, future plans, [sec:summary] summary, conclusions and comments, conclusion and further work, conclusion and open questions, conclusions & future work, 5. conclusions, [sec:conclusion] conclusion, *concluding remarks*, iv. summary, conclusions[conc], conclusion:, [concl]conclusions, summary and perspective, conclusions[sec:conclusion], [sec:level1]conclusions, open issues, [sec:conc]conclusion, [sec:concl]conclusion, [sec:sum]summary, summary of the results, implications and conclusions, conclusions[conclusion], some remarks, conclusions[concl], conclusion and future research, conclusion remarks, vi. conclusion, perspective, conclusions and future developments, [conc]conclusion, general remarks, summary and conclusions[sec:summary], summary and open questions, 4. conclusion, conclusion and future prospects, concluding remarks and perspectives, remarks and questions, remarks and questions, [conclusion] conclusion, summary and implications, conclusive remarks, comments and conclusions, summary of conclusions, [conclusion]conclusions, conclusion and perspective, conclusion[sec:conc], [sec:summary]summary and conclusions, [sec:level1]summary, [sec:con]conclusion, [sec:level4]conclusion, conclusions and outlook., [summary]summary and conclusions, conclusion[sec:concl], 5. conclusion, [conc]conclusions, outlook and conclusion, remarks and conclusion,  summary and conclusion, conlusions, conclusion and final remarks, v. summary, future outlook, future improvements, summary and open problems, conclusion[concl], summary]"

CONC_KWs_STR = "[" + str(['' + kw.strip() + '' for kw in CONC_KWs_STR[1:-1].split(',')]) + "]"
CONC_KWs = eval(CONC_KWs_STR)[0]

ABSTRACT_KWs_STR = "[abstract, 0 abstract]"
ABSTRACT_KWs_STR = "[" + str(['' + kw.strip() + '' for kw in ABSTRACT_KWs_STR[1:-1].split(',')]) + "]"
ABS_KWs = eval(ABSTRACT_KWs_STR)[0]



def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


# Tokenization classes

class LongformerData():
    def __init__(self, args=None):
        if args is not None:
            self.args = args
        self.CHUNK_LIMIT = 2500

        self.tokenizer = LongformerTokenizerMine.from_pretrained('longformer-based-uncased', do_lower_case=True)

        self.sep_token = '</s>'
        self.cls_token = '<s>'
        self.pad_token = '<pad>'
        self.tgt_bos = 'madeupword0000'
        self.tgt_eos = 'madeupword0001'
        self.tgt_sent_split = 'madeupword0002'

        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def cal_token_len(self, src):
        idxs = [i for i, s in enumerate(src) if (len(s[0]) > self.args.min_src_ntokens_per_sent)]
        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        src = src[:self.args.max_src_nsents]
        src_txt = [' '.join(sent[0]) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]

        return len(src_subtokens)

    def cal_token_len_prep(self, src):
        # idxs = [i for i, s in enumerate(src)]
        # src = [src[i] for i in idxs]
        src_txt = [sent[0] for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]

        return len(src_subtokens)


    def _encode_intro(self, src_sents_tokens, _intro_labels, chunk_size=2498):
        idxs = [i for i, s in enumerate(src_sents_tokens) if (len(s[1]) > self.args.min_src_ntokens_per_sent) and (len(s[1]) < self.args.max_src_ntokens_per_sent)]
        src_sents_tokens = [src_sents_tokens[i] for i in idxs]
        _intro_labels = [_intro_labels[i] for i in idxs]

        src_txt = [' '.join(sent[1]).strip() for sent in src_sents_tokens]
        # section_heading_txt = [s for s in section_heading_txt]
        sample_till_sent = 0

        lenS = 0
        for idx, sent in enumerate(src_txt):
            if lenS < chunk_size:
                sent_subtokens = self.tokenizer.tokenize(sent)
                lenS += len(sent_subtokens)
                sample_till_sent +=1
            else:
                lenS -= len(src_txt[idx-1])
                sample_till_sent -= 1
                break

        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt[:sample_till_sent])
        intro_labels = _intro_labels[:sample_till_sent]
        # text = self.cls_token + ' '.join(src_txt) + self.cls_token
        src_subtokens = self.tokenizer.tokenize(text)

        src_subtokens = [self.cls_token] + src_subtokens[:chunk_size] + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)

        cls_ids_intro = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]


        return src_subtoken_idxs, intro_labels, cls_ids_intro

    def make_chunks(self, src_sents_tokens, tgt, sent_labels=None, section_heading_txt=None, sent_rg_scores=None, chunk_size=2500,
                    paper_idx=0, sent_sect_labels=None):

        idxs = [i for i, s in enumerate(src_sents_tokens) if (len(s[0]) > self.args.min_src_ntokens_per_sent)]
        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])

        _sent_labels = [0] * len(src_sents_tokens)
        for l in sent_labels:
            _sent_labels[l] = 1

        src_sents_tokens = [src_sents_tokens[i][:self.args.max_src_ntokens_per_sent] for i in idxs]

        sent_labels = [_sent_labels[i] for i in idxs]
        section_heading_txt = [section_heading_txt[i] for i in idxs]
        sent_rg_scores = [sent_rg_scores[i] for i in idxs]
        sent_sect_labels = [sent_sect_labels[i] for i in idxs]

        src_sents_tokens = src_sents_tokens[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]
        sent_rg_scores = sent_rg_scores[:self.args.max_src_nsents]
        sent_sect_labels = sent_sect_labels[:self.args.max_src_nsents]

        section_heading_txt = section_heading_txt[:self.args.max_src_nsents]

        # calculate section importance
        section_rouge = {}
        section_text = ''
        # section_heading_txt = [s[1] for s in src_sents_tokens]

        for idx, sent in enumerate(src_sents_tokens):
            sect_txt = sent[1]

            if idx == 0:
                cursect = sect_txt
                section_text += ' '.join(sent[0])
                section_text += ' '

            if sect_txt == cursect:
                section_text += ' '.join(sent[0])
                section_text += ' '

            else:  # sect has changed...

                # rg_score = evaluate_rouge([tgt_txt.replace('<q>', ' ')], [section_text.strip()])[2]
                rg_score = 0
                section_rouge[cursect] = rg_score
                cursect = sent[1]
                section_text = ''
                section_text += ' '.join(sent[0])

        # rg_score = evaluate_rouge([tgt_txt.replace('<q>', ' ')], [section_text.strip()])[2]
        # section_rouge[cursect] = rg_score
        section_rouge[cursect] = 0

        # if "illustrative @xmath39 matrix example of a @xmath1-symmetric hamiltonian" in section_heading_txt:
        #     print('here')
        #     import pdb;pdb.set_trace()

        src_txt = [' '.join(sent[0]) for sent in src_sents_tokens]
        # section_heading_txt = [s for s in section_heading_txt]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)

        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]
        section_heading_txt = section_heading_txt[:len(cls_ids)]
        sent_rg_scores = sent_rg_scores[:len(cls_ids)]
        sent_sect_labels = sent_sect_labels[:len(cls_ids)]

        out_sents_labels = []
        out_section_heading_txt = []
        out_sents_rg_scores = []
        out_sent_sect_labels = []
        out_sect_sentwise_rg = []
        cur_len = 0
        out_src = []
        rg_score = 0
        j = 0
        last_chunk = False
        while j < len(cls_ids):
            if j == len(cls_ids) - 1:
                out_src1 = out_src.copy()
                out_sect_sentwise_rg1 = out_sect_sentwise_rg.copy()
                out_sents_labels1 = out_sents_labels.copy()
                out_section_heading_txt1 = out_section_heading_txt.copy()
                out_sents_rg_scores1 = out_sents_rg_scores.copy()
                out_sent_sect_labels1 = out_sent_sect_labels.copy()
                out_src.clear()
                out_sect_sentwise_rg.clear()
                out_sents_labels.clear()
                out_section_heading_txt.clear()
                out_sents_rg_scores.clear()
                out_sent_sect_labels.clear()
                cur_len1 = cur_len
                cur_len = 0
                last_chunk = True
                if len(out_src1) == 0:
                    j += 1
                    continue
                yield out_src1, out_sents_labels1, out_section_heading_txt1, out_sents_rg_scores1, cur_len1, last_chunk, rg_score, section_rouge, out_sent_sect_labels1

            if cur_len < chunk_size:
                out_src.append((src_sents_tokens[j][0], src_sents_tokens[j][1], src_sents_tokens[j][2]))
                out_sect_sentwise_rg.append(section_rouge[src_sents_tokens[j][1]])
                out_sents_labels.append(sent_labels[j])
                out_section_heading_txt.append(section_heading_txt[j])
                out_sents_rg_scores.append(sent_rg_scores[j])
                out_sent_sect_labels.append(sent_sect_labels[j])
                if j != 0:
                    cur_len += len(src_subtokens[cls_ids[j - 1]:cls_ids[j]])
                else:
                    cur_len += len(src_subtokens[:cls_ids[j]])
                j += 1

            else:
                j = j - 1
                out_src = out_src[:-1]
                out_sect_sentwise_rg = out_sect_sentwise_rg[:-1]
                out_sents_labels = out_sents_labels[:-1]
                out_section_heading_txt = out_section_heading_txt[:-1]
                out_sents_rg_scores = out_sents_rg_scores[:-1]
                out_sent_sect_labels = out_sent_sect_labels[:-1]
                out_src1 = out_src.copy()
                out_sect_sentwise_rg1 = out_sect_sentwise_rg.copy()
                out_sents_labels1 = out_sents_labels.copy()
                out_section_heading_txt1 = out_section_heading_txt.copy()
                out_sents_rg_scores1 = out_sents_rg_scores.copy()
                out_sent_sect_labels1 = out_sent_sect_labels.copy()
                out_src.clear()
                out_sents_labels.clear()
                out_section_heading_txt.clear()
                out_sent_sect_labels.clear()
                cur_len1 = cur_len
                cur_len = 0
                last_chunk = False
                if len(out_src1) == 0:
                    j += 1
                    continue

                yield out_src1, out_sents_labels1, out_section_heading_txt1, out_sents_rg_scores1, cur_len1, last_chunk, rg_score, section_rouge, out_sent_sect_labels1

    def preprocess_single(self, src, tgt, sent_rg_scores=None, sent_labels=None, sent_sections=None,
                          use_bert_basic_tokenizer=False, is_test=False, section_rgs=None, debug=False, sent_sect_labels=None):


        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s[0]) for s in src]
        idxs = [i for i, s in enumerate(src)]

        _sent_labels = sent_labels

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]

        sent_labels = [_sent_labels[i] for i in idxs]
        sent_sections = [sent_sections[i] for i in idxs]
        if sent_rg_scores is not None:
            sent_rg_scores = [sent_rg_scores[i] for i in idxs]

        if sent_sect_labels is not None:
            sent_sect_labels_chunk = [sent_sect_labels[i] for i in idxs]

        src = src[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]
        sent_sections = sent_sections[:self.args.max_src_nsents]

        if sent_rg_scores is not None:
            sent_rg_scores = sent_rg_scores[:self.args.max_src_nsents]

        if sent_sect_labels is not None:
            sent_sect_labels = sent_sect_labels[:self.args.max_src_nsents]

        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        src_txt = [' '.join(sent[0]) for sent in src]
        src_sent_token_count = [self.cal_token_len([(sent[0], 'test')]) for sent in src]

        src_sents_sections = [sent[1] for sent in src]
        src_sents_number = [sent[2] for sent in src]

        try:
            sents_sect_wise_rg = [section_rgs[sect] for sect in src_sents_sections]
        except:
            import pdb;pdb.set_trace()
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []

        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_indxes = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]

        sent_labels = sent_labels[:len(cls_indxes)]
        sent_sections = sent_sections[:len(cls_indxes)]
        if sent_rg_scores is not None:
            sent_rg_scores = sent_rg_scores[:len(cls_indxes)]

        if sent_sect_labels is not None:
            sent_sect_labels_chunk = sent_sect_labels[:len(cls_indxes)]

        tgt_subtokens_str = 'madeupword0000 ' + ' madeupword0002 '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt
             in tgt]) + ' madeupword0001'
        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]

        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)
        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        # if debug:
        #     import pdb;pdb.set_trace()

        return src_subtoken_idxs, sent_rg_scores, sent_labels, sent_sections, tgt_subtoken_idxs, segments_ids, cls_indxes, src_txt, tgt_txt, sents_sect_wise_rg, src_sents_number, src_sent_token_count, sent_sect_labels

class BertData():
    def __init__(self, args):
        self.CHUNK_LIMIT = 512
        self.args = args

        if args.model_name == 'scibert':
            self.tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)

        elif 'bert-base' in args.model_name or 'bert-large' in args.model_name:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'

        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def make_chunks(self, src_sents_tokens, tgt, sent_labels=None, section_heading_txt=None, sent_rg_scores=None, chunk_size=512,
                    paper_idx=0):

        idxs = [i for i, s in enumerate(src_sents_tokens) if (len(s[0]) > self.args.min_src_ntokens_per_sent)]
        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])

        _sent_labels = [0] * len(src_sents_tokens)
        for l in sent_labels:
            _sent_labels[l] = 1

        src_sents_tokens = [src_sents_tokens[i][:self.args.max_src_ntokens_per_sent] for i in idxs]

        sent_labels = [_sent_labels[i] for i in idxs]
        section_heading_txt = [section_heading_txt[i] for i in idxs]
        sent_rg_scores = [sent_rg_scores[i] for i in idxs]

        src_sents_tokens = src_sents_tokens[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]
        sent_rg_scores = sent_rg_scores[:self.args.max_src_nsents]

        section_heading_txt = section_heading_txt[:self.args.max_src_nsents]

        # calculate section importance
        section_rouge = {}
        section_text = ''
        # section_heading_txt = [s[1] for s in src_sents_tokens]

        for idx, sent in enumerate(src_sents_tokens):
            sect_txt = sent[1]

            if idx == 0:
                cursect = sect_txt
                section_text += ' '.join(sent[0])
                section_text += ' '

            if sect_txt == cursect:
                section_text += ' '.join(sent[0])
                section_text += ' '

            else:  # sect has changed...

                # rg_score = evaluate_rouge([tgt_txt.replace('<q>', ' ')], [section_text.strip()])[2]
                rg_score = 0
                section_rouge[cursect] = rg_score
                cursect = sent[1]
                section_text = ''
                section_text += ' '.join(sent[0])

        # rg_score = evaluate_rouge([tgt_txt.replace('<q>', ' ')], [section_text.strip()])[2]
        # section_rouge[cursect] = rg_score
        section_rouge[cursect] = 0

        # if "illustrative @xmath39 matrix example of a @xmath1-symmetric hamiltonian" in section_heading_txt:
        #     print('here')
        #     import pdb;pdb.set_trace()

        src_txt = [' '.join(sent[0]).replace('[CLS]', '[ CLS ]').replace('[SEP]', '[ SEP ]') for sent in src_sents_tokens]
        # section_heading_txt = [s for s in section_heading_txt]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)

        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]
        section_heading_txt = section_heading_txt[:len(cls_ids)]
        sent_rg_scores = sent_rg_scores[:len(cls_ids)]

        out_sents_labels = []
        out_section_heading_txt = []
        out_sents_rg_scores = []
        out_sect_sentwise_rg = []
        cur_len = 0
        out_src = []
        rg_score = 0
        j = 0
        last_chunk = False
        while j < len(cls_ids):
            if j == len(cls_ids) - 1:
                out_src1 = out_src.copy()
                out_sect_sentwise_rg1 = out_sect_sentwise_rg.copy()
                out_sents_labels1 = out_sents_labels.copy()
                out_section_heading_txt1 = out_section_heading_txt.copy()
                out_sents_rg_scores1 = out_sents_rg_scores.copy()
                out_src.clear()
                out_sect_sentwise_rg.clear()
                out_sents_labels.clear()
                out_section_heading_txt.clear()
                out_sents_rg_scores.clear()
                cur_len1 = cur_len
                cur_len = 0
                last_chunk = True
                if len(out_src1) == 0:
                    j += 1
                    continue
                yield out_src1, out_sents_labels1, out_section_heading_txt1, out_sents_rg_scores1, cur_len1, last_chunk, rg_score, section_rouge
            if cur_len < chunk_size:
                try:
                    out_src.append((src_sents_tokens[j][0], src_sents_tokens[j][1], src_sents_tokens[j][2]))
                    # print('1')
                except:
                    print('2')
                    # import pdb;pdb.set_trace()
                out_sect_sentwise_rg.append(section_rouge[src_sents_tokens[j][1]])
                out_sents_labels.append(sent_labels[j])
                out_section_heading_txt.append(section_heading_txt[j])
                out_sents_rg_scores.append(sent_rg_scores[j])
                if j != 0:
                    cur_len += len(src_subtokens[cls_ids[j - 1]:cls_ids[j]])
                else:
                    cur_len += len(src_subtokens[:cls_ids[j]])
                j += 1

            else:
                j = j - 1
                out_src = out_src[:-1]
                out_sect_sentwise_rg = out_sect_sentwise_rg[:-1]
                out_sents_labels = out_sents_labels[:-1]
                out_section_heading_txt = out_section_heading_txt[:-1]
                out_sents_rg_scores = out_sents_rg_scores[:-1]
                out_src1 = out_src.copy()
                out_sect_sentwise_rg1 = out_sect_sentwise_rg.copy()
                out_sents_labels1 = out_sents_labels.copy()
                out_section_heading_txt1 = out_section_heading_txt.copy()
                out_sents_rg_scores1 = out_sents_rg_scores.copy()
                out_src.clear()
                out_sents_labels.clear()
                out_section_heading_txt.clear()
                out_sents_rg_scores.clear()
                cur_len1 = cur_len
                cur_len = 0
                last_chunk = False
                if len(out_src1) == 0:
                    j += 1
                    continue

                yield out_src1, out_sents_labels1, out_section_heading_txt1, out_sents_rg_scores1, cur_len1, last_chunk, rg_score, section_rouge



    def preprocess_single(self, src, tgt, sent_rg_scores=None, sent_labels=None, sent_sections=None,
                          use_bert_basic_tokenizer=False, is_test=False, section_rgs=None, debug=False):


        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s[0]) for s in src]
        idxs = [i for i, s in enumerate(src) if (len(s[0]) > self.args.min_src_ntokens_per_sent)]

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            try:
                _sent_labels[l] = 1
            except Exception as e:
                print(e)
                import pdb;
                pdb.set_trace()
                print(sent_labels)
                print(len(src))

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]

        sent_labels = [_sent_labels[i] for i in idxs]
        sent_sections = [sent_sections[i] for i in idxs]
        if sent_rg_scores is not None:
            sent_rg_scores = [sent_rg_scores[i] for i in idxs]

        src = src[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]
        sent_sections = sent_sections[:self.args.max_src_nsents]
        if sent_rg_scores is not None:
            sent_rg_scores = sent_rg_scores[:self.args.max_src_nsents]

        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        src_txt = [' '.join(sent[0]) for sent in src]
        src_sent_token_count = [self.cal_token_len([(sent[0], 'test')]) for sent in src]

        src_sents_sections = [sent[1] for sent in src]
        src_sents_number = [sent[2] for sent in src]

        try:
            sents_sect_wise_rg = [section_rgs[sect] for sect in src_sents_sections]
        except:
            import pdb;pdb.set_trace()
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []

        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_indxes = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]

        sent_labels = sent_labels[:len(cls_indxes)]
        sent_sections = sent_sections[:len(cls_indxes)]
        if sent_rg_scores is not None:
            sent_rg_scores = sent_rg_scores[:len(cls_indxes)]

        tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt
             in tgt]) + ' [unused1]'
        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]

        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)
        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        # if debug:
        #     import pdb;pdb.set_trace()

        return src_subtoken_idxs, sent_rg_scores, sent_labels, sent_sections, tgt_subtoken_idxs, segments_ids, cls_indxes, src_txt, tgt_txt, sents_sect_wise_rg, src_sents_number, src_sent_token_count

    def cal_token_len(self, src):

        idxs = [i for i, s in enumerate(src) if (len(s[0]) > self.args.min_src_ntokens_per_sent)]
        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        src = src[:self.args.max_src_nsents]
        src_txt = [' '.join(sent[0]) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]

        return len(src_subtokens)


class BertDataOriginal():
    def __init__(self, args):
        self.args = args

        if args.model_name == 'scibert':
            self.tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)

        elif 'bert-base' in args.model_name or 'bert-large' in args.model_name:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

    def preprocess(self, src, tgt, sent_labels, use_bert_basic_tokenizer=False, is_test=False):

        if ((not is_test) and len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]

        _sent_labels = [0] * len(src)
        for l in sent_labels:
            _sent_labels[l] = 1

        src = [src[i][:self.args.max_src_ntokens_per_sent] for i in idxs]
        sent_labels = [_sent_labels[i] for i in idxs]
        src = src[:self.args.max_src_nsents]
        sent_labels = sent_labels[:self.args.max_src_nsents]

        if ((not is_test) and len(src) < self.args.min_src_nsents):
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)

        src_subtokens = self.tokenizer.tokenize(text)

        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        sent_labels = sent_labels[:len(cls_ids)]

        tgt_subtokens_str = '[unused0] ' + ' [unused2] '.join(
            [' '.join(self.tokenizer.tokenize(' '.join(tt), use_bert_basic_tokenizer=use_bert_basic_tokenizer)) for tt in tgt]) + ' [unused1]'
        tgt_subtoken = tgt_subtokens_str.split()[:self.args.max_tgt_ntokens]
        if ((not is_test) and len(tgt_subtoken) < self.args.min_tgt_ntokens):
            return None

        tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt_subtoken)

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]

        return src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt


# bert-function
def format_to_bert(args):
    test_kws = pd.read_csv('csv_files/train_papers_sects_longsum.csv')

    kws = {
        'intro': [kw.strip() for kw in test_kws['intro'].dropna()],
        'related': [kw.strip() for kw in test_kws['related work'].dropna()],
        'exp': [kw.strip() for kw in test_kws['experiments'].dropna()],
        'res': [kw.strip() for kw in test_kws['results'].dropna()],
        'conclusion': [kw.strip() for kw in test_kws['conclusion'].dropna()]
    }

    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['test']


    if len(args.sent_numbers_file) > 0:
        sent_numbers = pickle.load(open(args.sent_numbers_file, "rb"))
    else:
        sent_numbers = None

    # ARXIVIZATION
    bart = args.bart
    check_path_existence(args.save_path)
    intro_map = {}
    for corpus_type in datasets:
        a_lst = []
        c = 0
        for json_f in glob.glob(pjoin(args.raw_path, corpus_type + '.*.json')):
            real_name = json_f.split('/')[-1]
            c += 1
            a_lst.append(
                (corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('.json', '.bert.pt')), kws, bart,
                 sent_numbers, 1))
        print("Number of files: " + str(c))

        ##########################
        ###### <DEBUGGING> #######
        ##########################

        # for a in a_lst:
        #     _format_to_bert_original(a)


        # single
        # json_f = args.raw_path + '/val.9.json'
        # _format_to_bert(('test', str(json_f), args, pjoin(args.save_path, str(json_f).replace('.json', '.bert.pt')), kws, bart,
        #          sent_numbers, 8))

        ##########################
        ###### <DEBUGGING> #######
        ##########################

        pool = Pool(args.n_cpus)
        print('Processing {} set with {} json files...'.format(corpus_type, len(a_lst)))
        all_papers_count = 0
        all_paper_ids = {}
        intro_labels_count = []
        intro_labels_len_count = []
        for d in tqdm(pool.imap(_format_to_bert_original, a_lst), total=len(a_lst), desc=''):
            # all_paper_ids[d[0]] = d[1]
            # all_papers_count += d[2]
            # intro_labels_count.extend(d[3])
            # intro_labels_len_count.extend(d[4])
            pass

        # pool.close()
        # pool.join()
        # import statistics
        # print('Num of papers: {}'.format(all_papers_count))
        # print('Median of intro labels: {}'.format(statistics.median(intro_labels_count)))
        # print('Mean of intro labels: {}'.format(statistics.mean(intro_labels_count)))
        # print('Min of intro labels: {}'.format(min(intro_labels_len_count)))

def _format_to_bert(params):
    corpus_type, json_file, args, save_file, kws, bart, sent_numbers_whole, debug_idx = params
    papers_ids = set()
    intro_labels_count = []
    intro_labels_len_count = []

    def remove_ack(source, debug=False):
        out = []
        sect_idx = 2
        if debug:
            import pdb;pdb.set_trace()

        for sent in source:
            section_txt = sent[sect_idx].lower().replace(':', ' ').replace('.', ' ').replace(';', ' ')
            if \
                    'acknowledgment' in section_txt.split() \
                    or 'acknowledgments' in section_txt.split() \
                    or 'acknowledgements' in section_txt.split() \
                    or 'fund' in section_txt.split() \
                    or 'funding' in section_txt.split() \
                    or 'appendices' in section_txt.split() \
                    or 'proof of' in section_txt.split() \
                    or 'related work' in section_txt.split() \
                    or 'previous works' in section_txt.split() \
                    or 'references' in section_txt.split() \
                    or 'figure captions' in section_txt.split() \
                    or 'acknowledgement' in section_txt.split() \
                    or 'appendix' in section_txt.split()\
                    or 'appendix:' in section_txt.split():

                continue

            else:
                out.append(sent)

        return out

    def _get_intro(source, paper_id=None, retrieve_first_sect=False, section_id=None, debug=False):
        def _check_subtokens(section_heading):
            for subheading in section_heading.split():
                if subheading.lower() in kws['intro']:
                    return True

        def f7(seq):
            seen = set()
            seen_add = seen.add
            return [x for x in seq if not (x in seen or seen_add(x))]


        out = []
        sect_idx = 2

        if section_id is None:
            if retrieve_first_sect:

                sections = f7([s[2] for s in source])
                for sent in source:
                    for kw in INTRO_KWs:
                        if kw.lower() == sent[sect_idx].lower():
                            out.append(sent)
                            break

                    for kw in CONC_KWs:
                        if kw.lower() in sent[sect_idx].lower():
                            out.append(sent)
                            break

                if debug:
                    print(1)
                    import pdb;pdb.set_trace()

                    # if sent[sect_idx].lower() == first_sect.lower():
                    #     out.append(sent)
                # if len(out) == 0:
                #     first_sect = list(sections)[0]
                #     for sent in source:
                #         if sent[sect_idx].lower() == first_sect.lower():
                #             out.append(sent)

                if len(out) < 6:
                    first_sect = list(sections)[0]
                    for sent in source:
                        if sent[sect_idx].lower() == first_sect.lower():
                            out.append(sent)
                    if debug:
                        print(1.1)
                        import pdb;
                        pdb.set_trace()
                    if len(out) < 6 and len(list(sections)) > 1:
                        second_sect = list(sections)[1]
                        for sent in source:
                            if sent[sect_idx].lower() == second_sect.lower():
                                out.append(sent)
                        if len(out) < 6:
                            third_sect = list(sections)[2]
                            for sent in source:
                                if sent[sect_idx].lower() == third_sect.lower():
                                    out.append(sent)

                    # else:
                    #     return None
            else:
                for sent in source:
                    if sent[sect_idx].lower() in kws['intro'] \
                            or _check_subtokens(sent[sect_idx]):
                        out.append(sent)
        else:
            for sent in source:
                if sent[-1] in section_id:
                    out.append(sent)

        if debug:
            print(2)
            import pdb;
            pdb.set_trace()

        return out

    is_test = corpus_type == 'test'
    # if (os.path.exists(save_file)):
    #     logger.info('Ignore %s' % save_file)
    #     return

    model_name = args.model_name
    CHUNK_SIZE_CONST=-1
    if model_name == 'bert-based' or model_name == 'scibert' or model_name == 'bert-large':
        bert = BertData(args)
        CHUNK_SIZE_CONST = 512
    elif model_name == 'longformer':
        bert = LongformerData(args)
        CHUNK_SIZE_CONST = 2046

    CHUNK_INTRO = 1536

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []

    for j, data in enumerate(jobs[debug_idx-1:]):
        try:
            paper_id, data_src, data_tgt = data['id'], data['src'], data['tgt']
            # import pdb;pdb.set_trace()
            # labels = [d[-2] for d in data_src]

            # data_src = [s[1:] for s in data_src]
            if not isinstance(data_src[0][0], int):
                data_src = [[idx] + s for idx, s in enumerate(data_src)]

            data_src = remove_ack(data_src)
            # data_intro = _get_intro(data_src, section_id=[0])
            data_intro = _get_intro(data_src, retrieve_first_sect=True, debug=False)
            # data_intro = _get_intro(data_src, retrieve_first_sect=True)
            data_src = data_intro

            if len(data_intro) < 3:
                # retrieve first section as intro
                # data_intro = _get_intro(data_src, paper_id, retrieve_first_sect=True, section_id=[0,1])

                data_intro = _get_intro(data_src, paper_id, retrieve_first_sect=True, debug=False)
                # data_intro = _get_intro(data_src, paper_id, retrieve_first_sect=True)
                if data_intro is None:
                    with open('np_intro.txt', mode='a') as F:
                        F.write(save_file + ' idx: ' + str(j) + '\n')
                    continue


            # if len(data_intro) <= 7:
            #     data_intro = _get_intro(data_src, paper_id, retrieve_first_sect=True, section_id=[0,1,3])
            #
            # if len(data_intro) <= 7:
            #     data_intro = _get_intro(data_src, paper_id, retrieve_first_sect=True, section_id=[0, 1, 2, 3])
            #     if data_intro is None:
            #         with open('np_intro.txt', mode='a') as F:
            #             F.write(save_file + ' idx: ' + str(j) + '\n')
            #         continue


            # if len(data_intro)<=5:
            #     data_intro = _get_intro(data_src, paper_id, retrieve_first_sect=True, section_id=2)

            if len(data_intro) < 3:
                # import pdb;pdb.set_trace()

                with open('np_intro.txt', mode='a') as F:
                    F.write(save_file + ' idx: ' + str(j) + '\n')
                continue

        except:
            # import pdb;pdb.set_trace()
            print("NP: " + save_file + ' idx: ' + str(j) + '\n')
            with open('np_parsing.txt', mode='a') as F:
                F.write(save_file + ' idx: ' + str(j) + '\n')

        if len(data_src) < 5:
            # import pdb;pdb.set_trace()
            continue

        if sent_numbers_whole is not None:
            data_src = [d for d in data_src if d[0] in sent_numbers_whole[paper_id]]


        data_src = [s for s in data_src if len(s[1]) > args.min_src_ntokens_per_sent and len(s[1]) < args.max_src_ntokens_per_sent]
        # sent_labels = greedy_selection(source_sents, tgt, 10)
        sent_labels = [i for i, s in enumerate(data_src) if s[-2] == 1]
        sent_rg_scores = [s[3] for i, s in enumerate(data_src) if len(s[1]) > args.min_src_ntokens_per_sent and len(s[1]) < args.max_src_ntokens_per_sent]

        if (args.lower):
            source_sents = [([tkn.lower() for tkn in s[1]], s[2], s[0]) for s in data_src]
            data_tgt = [[tkn.lower() for tkn in s] for s in data_tgt]  # arxiv ––non-pubmed
        else:
            source_sents = [(s[1], s[2], s[0]) for s in data_src]

        sent_sections = [s[2] for s in data_src]
        sent_sect_labels = [s[-1] for s in data_src]
        # print(j)

        intro_text = ' '.join([' '.join(s[1]) for s in data_intro])
        intro_labels_prior = [s[-2] for s in data_intro]
        intro_subids, intro_labels, cls_ids_intro = bert._encode_intro(data_intro, intro_labels_prior, chunk_size=CHUNK_INTRO)

        # import pdb;pdb.set_trace()
        if len(intro_labels) < 5:
            # print('not: {}'.format(paper_id))
            # import pdb;pdb.set_trace()
            print(save_file + ' idx: ' + str(j) + '\n')
            continue
        try:
            intro_labels_count.append(sum(intro_labels))
            intro_labels_len_count.append(len(intro_labels))
        except:
            import pdb;pdb.set_trace()
            print(paper_id)
        # if paper_id == "cond-mat0609513":
        #     import pdb;
        #     pdb.set_trace()
        if len(intro_text.strip()) == 0:
            with open('not-intro-l1.txt', mode='a') as FD:
                FD.write(save_file + ' idx: ' + str(j) + ';paper_id: ' + str(paper_id) + '\n')
            continue
        if True:
            tkn_len = bert.cal_token_len(source_sents)
            debug = False

            if tkn_len > CHUNK_SIZE_CONST:
                try:
                    for chunk_num, chunk in enumerate(
                            bert.make_chunks(source_sents, data_tgt, sent_labels=sent_labels, section_heading_txt=sent_sections,
                                             sent_rg_scores=sent_rg_scores, sent_sect_labels=sent_sect_labels,  chunk_size=CHUNK_SIZE_CONST)):


                        src_chunk, sent_labels_chunk, sent_sections_chunk, sent_rg_scores_chunk, curlen, is_last_chunk, rg_score, section_rgs, sent_sect_labels_chunk = chunk
                        b_data = bert.preprocess_single(src_chunk, data_tgt, sent_labels=sent_labels_chunk,
                                                        sent_rg_scores=sent_rg_scores_chunk,
                                                        use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                                        is_test=is_test, sent_sections=sent_sections_chunk, section_rgs=section_rgs, debug=debug, sent_sect_labels=sent_sect_labels_chunk)


                        if (b_data is None):
                            # import pdb;pdb.set_trace()
                            with open('not_parsed_chunk_multi_processing.txt', mode='a') as F:
                                F.write(save_file + ' idx: ' + str(j) + ' paper_id: ' + str(paper_id) + '-' +str(chunk_num)+ '\n')
                            # print(paper_id)
                            continue

                        src_subtoken_idxs, sent_rg_scores, sent_labels_chunk, sent_sections_chunk, tgt_subtoken_idxs, \
                        segments_ids, cls_ids, src_txt, tgt_txt, sents_sect_wise_rg, src_sent_number, src_sent_token_count, sent_sect_labels = b_data
                        if len(sent_labels_chunk) != len(cls_ids):
                            print(len(sent_labels_chunk))
                            print('Cls length should equal sent lables: {}'.format(paper_id))

                        try:
                            # import pdb;
                            # pdb.set_trace()
                            b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                                           "src_sent_rg": sent_rg_scores.copy() if sent_rg_scores is not None else sent_sections_chunk,
                                           "sent_labels": sent_labels_chunk.copy(),
                                           "segs": segments_ids, 'clss': cls_ids,
                                           'src_txt': src_txt, "tgt_txt": tgt_txt,
                                           "paper_id": paper_id + '___' + str(chunk_num) + '___' + datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4()),
                                           "sent_sect_labels": sent_sect_labels.copy(), "segment_rg_score": rg_score,
                                           "sent_sections_txt": sent_sections_chunk,  "sent_sect_wise_rg": sents_sect_wise_rg, "sent_numbers": src_sent_number,
                                           "sent_token_count":src_sent_token_count,
                                           "intro_txt": intro_text, "src_intro": intro_subids,
                                           "intro_labels": intro_labels, "intro_cls_ids": cls_ids_intro}

                            papers_ids.add(paper_id.split('___')[0])
                            datasets.append(b_data_dict)
                        except:
                            # import pdb;pdb.set_trace()
                            with open('not_parsed_chunk_multi.txt', mode='a') as F:
                                F.write(save_file + ' idx: ' + str(j) + ' paper_id: ' + str(paper_id) + '-' +str(chunk_num)+ '\n')
                            # print('{} with {} sentences'.format(paper_id, len(src_chunk)))

                            continue
                except Exception:
                    with open('not_parsed_chunks_function.txt', mode='a') as F:
                        F.write(
                            save_file + ' idx: ' + str(j) + ' paper_id: ' + str(paper_id) + '-' + '\n')

            else:
                # non-sectionized or less than chunk size

                try:

                    for chunk_num, chunk in enumerate(
                            bert.make_chunks(source_sents, data_tgt, sent_labels=sent_labels, section_heading_txt=sent_sections,
                                             sent_rg_scores=sent_rg_scores, sent_sect_labels=sent_sect_labels,  chunk_size=CHUNK_SIZE_CONST)):
                        src_chunk, sent_labels_chunk, sent_sections_chunk, sent_rg_scores_chunk, curlen, is_last_chunk, rg_score, section_rgs, sent_sect_labels_chunk = chunk
                        b_data = bert.preprocess_single(src_chunk, data_tgt, sent_labels=sent_labels_chunk,
                                                        sent_rg_scores=sent_rg_scores_chunk,
                                                        use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                                        is_test=is_test, sent_sections=sent_sections_chunk, section_rgs=section_rgs, debug=debug, sent_sect_labels=sent_sect_labels_chunk)

                        if b_data == None:
                            with open('not_parsed_processing.txt', mode='a') as F:
                                F.write(save_file + ' idx: ' + str(j) + ' paper_id: ' + str(paper_id) + '\n')
                            print(paper_id)
                            continue

                        src_subtoken_idxs, sent_rg_scores, sent_labels_chunk, sent_sections_chunk, tgt_subtoken_idxs, \
                        segments_ids, cls_ids, src_txt, tgt_txt, sents_sect_wise_rg, src_sent_number, src_sent_token_count, sent_sect_labels = b_data

                        if len(sent_labels_chunk) != len(cls_ids):
                            print(len(sent_labels_chunk))
                            print('Cls length should equal sent lables: {}'.format(paper_id))

                        # import pdb;
                        # pdb.set_trace()
                        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                                       "src_sent_rg": sent_rg_scores.copy() if sent_rg_scores is not None else sent_sections_chunk,
                                       "sent_labels": sent_labels_chunk.copy(),
                                       "segs": segments_ids, 'clss': cls_ids,
                                       'src_txt': src_txt, "tgt_txt": tgt_txt,
                                       "paper_id": paper_id + '___' + str(
                                           chunk_num) + '___' + datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(
                                           uuid4()),
                                       "sent_sect_labels": sent_sect_labels.copy(), "segment_rg_score": rg_score,
                                       "sent_sections_txt": sent_sections_chunk,
                                       "sent_sect_wise_rg": sents_sect_wise_rg, "sent_numbers": src_sent_number,
                                       "sent_token_count": src_sent_token_count,
                                       "intro_txt": intro_text, "src_intro": intro_subids,
                                       "intro_labels": intro_labels, "intro_cls_ids": cls_ids_intro}
                        papers_ids.add(paper_id.split('___')[0])
                        datasets.append(b_data_dict)
                except:
                    # import pdb;
                    # pdb.set_trace()
                    with open('not_parsed.txt', mode='a') as F:
                        F.write(save_file + ' idx: ' + str(j) + ' paper_id: ' + str(paper_id) + '\n')
                    # print(paper_id)

                    continue

    print('Processed instances %d data' % len(datasets))
    print('Saving to %s' % save_file)
    # written_files = glob.glob('/'.join(save_file.split('/')[:-1]) + '/' + corpus_type + '*.pt')
    torch.save(datasets, save_file)
    with open('papers_' + args.model_name + '_' +corpus_type +'.txt', mode='a') as F:
        for p in papers_ids:
            F.write(str(p))
            F.write('\n')

    # if len(written_files) > 0:
    #     idxs = [int(w.split('/')[-1].split('.')[1]) for w in written_files]
    #     indx = sorted(idxs, reverse=True)[0]
    #     torch.save(datasets, '/'.join(save_file.split('/')[:-1]) + '/' + corpus_type + '.' + str(indx+1) + '.pt')
    #     datasets = []
    # else:
    #     torch.save(datasets, '/'.join(save_file.split('/')[:-1]) + '/' + corpus_type + '.' + str(0) + '.pt')

    datasets = []
    gc.collect()
    return save_file, papers_ids, len(papers_ids), intro_labels_count, intro_labels_len_count


def _format_to_bert_original(params):
    # corpus_type, json_file, args, save_file = params
    corpus_type, json_file, args, save_file, kws, bart, sent_numbers_whole, debug_idx = params

    is_test = corpus_type == 'test'
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertDataOriginal(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        source, tgt = d['src'], d['tgt']
        source = [s[0] for s in source[:args.max_src_nsents]]
        sent_labels = greedy_selection([s[0] for s in source[:args.max_src_nsents]], tgt, 3)
        if (args.lower):

            source = [' '.join(s).lower().split() for s in source]
            tgt = [' '.join(s).lower().split() for s in tgt]
            # tgt = [' '.join(s).lower() for s in tgt[0]]

        b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                 is_test=is_test)
        # b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer)

        if (b_data is None):
            print('rrrrrr')
            continue
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        # b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
        #                "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
        #                'src_txt': src_txt, "tgt_txt": tgt_txt}

        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                       "src_sent_rg": [0 for _ in sent_labels],
                       "src_sent_labels": sent_labels.copy(),
                       "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt,
                       "paper_id": d['id'],
                       "sent_sect_labels": [0 for _ in sent_labels], "segment_rg_score": [0 for _ in sent_labels],
                       "sent_sections_txt": [0 for _ in sent_labels], "sent_sect_wise_rg": [0 for _ in sent_labels],
                       "sent_numbers": [0 for _ in sent_labels],
                       "sent_token_count": [0 for _ in sent_labels],
                       "intro_txt": '', "src_intro": src_subtoken_idxs,
                       "intro_labels": [0 for _ in sent_labels], "intro_cls_ids": cls_ids}

        datasets.append(b_data_dict)
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


# line function
def format_to_lines(args):
    if args.dataset != '':
        corpuses_type = [args.dataset]
    else:
        corpuses_type = ['train', 'val', 'test']

    sections = {}
    for corpus_type in corpuses_type:
        files = []
        for f in glob.glob(args.raw_path +'/*.json'):
            files.append(f)
            # import pdb;pdb.set_trace()
        corpora = {corpus_type: files}
        for corpus_type in corpora.keys():
            a_lst = [(f, args.keep_sect_num) for f in corpora[corpus_type]]
            pool = Pool(args.n_cpus)
            dataset = []
            p_ct = 0
            all_papers_count = 0
            curr_paper_count = 0
            check_path_existence(args.save_path)

            ##########################
            ###### <DEBUGGING> #######
            ##########################

            for a in tqdm(a_lst, total=len(a_lst)):
                d = _format_to_lines(a)
                if d is not None:
                    # dataset.extend(d[0])
                    dataset.append(d)
                    if (len(dataset) > args.shard_size):
                        pt_file = "{:s}{:s}.{:d}.json".format(args.save_path + '', corpus_type, p_ct)
                        check_path_existence(args.save_path)
                        print(pt_file)
                        with open(pt_file, 'w') as save:
                            # save.write('\n'.join(dataset))
                            save.write(json.dumps(dataset))
                            print('data len: {}'.format(len(dataset)))
                            p_ct += 1
                            all_papers_count += len(dataset)
                            dataset = []
            if (len(dataset) > 0):

                pt_file = "{:s}{:s}.{:d}.json".format(args.save_path + '', corpus_type, p_ct)
                print(pt_file)
                with open(pt_file, 'w') as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    all_papers_count += len(dataset)
                    dataset = []

            print('Processed {} papers for {} set'.format(all_papers_count, corpus_type))
            ###########################
            ###### </DEBUGGING> #######
            ###########################


            # for d in tqdm(pool.imap_unordered(_format_longsum_to_lines_section_based, a_lst), total=len(a_lst)):
            # for d in tqdm(pool.imap_unordered(_format_to_lines, a_lst), total=len(a_lst)):
            #     # d_1 = d[1]
            #     if d is not None:
            #         all_papers_count+=1
            #         curr_paper_count+=1
            #
            #         # dataset.extend(d[0])
            #         dataset.append(d)
            #         # import pdb;pdb.set_trace()
            #         # if (len(dataset) > args.shard_size):
            #         if (curr_paper_count > args.shard_size):
            #             pt_file = "{:s}{:s}.{:d}.json".format(args.save_path + '', corpus_type, p_ct)
            #             print(pt_file)
            #             with open(pt_file, 'w') as save:
            #                 # save.write('\n'.join(dataset))
            #                 save.write(json.dumps(dataset))
            #                 print('data len: {}'.format(len(dataset)))
            #                 p_ct += 1
            #                 dataset = []
            #             curr_paper_count = 0
            #
            #
            # pool.close()
            # pool.join()
            #
            # if (len(dataset) > 0):
            #     pt_file = "{:s}{:s}.{:d}.json".format(args.save_path + '', corpus_type, p_ct)
            #     print(pt_file)
            #     # all_papers_count += len(dataset)
            #     with open(pt_file, 'w') as save:
            #         # save.write('\n'.join(dataset))
            #         save.write(json.dumps(dataset))
            #         p_ct += 1
            #
            #         dataset = []
            # print('Processed {} papers for {} set'.format(all_papers_count, corpus_type))


    # sections = sorted(sections.items(), key=lambda x: x[1], reverse=True)
    # sections = dict(sections)
    # with open('sect_stat.txt', mode='a') as F:
    #     for s, c in sections.items():
    #         F.write(s + ' --> '+ str(c))
    #         F.write('\n')



# line function
def format_to_lines_bart(args):
    if args.dataset != '':
        corpuses_type = [args.dataset]
    else:
        print("You should consider the dataset...")
        os._exit(0)
    #     corpuses_type = ['train', 'val', 'test']


    papers_sent_numbers = pickle.load(open(args.sent_numbers_path, mode='rb'))
    sections = {}

    for corpus_type in corpuses_type:
        files = []
        for f in glob.glob(args.raw_path +'/*.json'):
            files.append(f)


            # import pdb;pdb.set_trace()
        corpora = {corpus_type: files}
        for corpus_type in corpora.keys():
            a_lst = []
            for f in corpora[corpus_type]:
                try:
                    if 'pubmedL' in f:
                        a_lst.append(
                            (f, args.keep_sect_num, papers_sent_numbers[f.split('/')[-1].replace('.json', '.nxml')]))
                    else:
                        a_lst.append((f, args.keep_sect_num, papers_sent_numbers[f.split('/')[-1].replace('.json','')]))
                except:
                    print('non_')
                    continue

            print('Checked papers: {}'.format(len(a_lst)))
            dataset = []
            check_path_existence(args.save_path)

            ##########################
            ###### <DEBUGGING> #######
            ##########################

            for a in tqdm(a_lst, total=len(a_lst)):

                d = _format_to_lines_bart(a)
                if d is not None:
                    # dataset.extend(d[0])
                    dataset.append(d)
            if (len(dataset) > 0):

               # with open(args.save_path + "/{}.source".format(args.dataset), mode='w') as fSrc, \
               #         open(args.save_path + "/{}.target".format(args.dataset), mode='w') as fTgt:
               #          for d in dataset:
               #              fSrc.write(d['src'].strip())
               #              fSrc.write('\n')
               #              fTgt.write(d['tgt'].strip())
               #              fTgt.write('\n')

                fOut = open(args.save_path + "/{}.jsonl".format(args.dataset), mode='w')
                for d in dataset:
                    ent = {'text': d['src'], 'abstract': d['tgt']}
                    json.dump(ent, fOut)
                    fOut.write('\n')



            print('Processed {} papers for {} set'.format(len(dataset), corpus_type))
            ###########################
            ###### </DEBUGGING> #######
            ###########################


            # for d in tqdm(pool.imap_unordered(_format_longsum_to_lines_section_based, a_lst), total=len(a_lst)):
            # for d in tqdm(pool.imap_unordered(_format_to_lines, a_lst), total=len(a_lst)):
            #     # d_1 = d[1]
            #     if d is not None:
            #         all_papers_count+=1
            #         curr_paper_count+=1
            #
            #         # dataset.extend(d[0])
            #         dataset.append(d)
            #         # import pdb;pdb.set_trace()
            #         # if (len(dataset) > args.shard_size):
            #         if (curr_paper_count > args.shard_size):
            #             pt_file = "{:s}{:s}.{:d}.json".format(args.save_path + '', corpus_type, p_ct)
            #             print(pt_file)
            #             with open(pt_file, 'w') as save:
            #                 # save.write('\n'.join(dataset))
            #                 save.write(json.dumps(dataset))
            #                 print('data len: {}'.format(len(dataset)))
            #                 p_ct += 1
            #                 dataset = []
            #             curr_paper_count = 0
            #
            #
            # pool.close()
            # pool.join()
            #
            # if (len(dataset) > 0):
            #     pt_file = "{:s}{:s}.{:d}.json".format(args.save_path + '', corpus_type, p_ct)
            #     print(pt_file)
            #     # all_papers_count += len(dataset)
            #     with open(pt_file, 'w') as save:
            #         # save.write('\n'.join(dataset))
            #         save.write(json.dumps(dataset))
            #         p_ct += 1
            #
            #         dataset = []
            # print('Processed {} papers for {} set'.format(all_papers_count, corpus_type))


    # sections = sorted(sections.items(), key=lambda x: x[1], reverse=True)
    # sections = dict(sections)
    # with open('sect_stat.txt', mode='a') as F:
    #     for s, c in sections.items():
    #         F.write(s + ' --> '+ str(c))
    #         F.write('\n')


def _format_to_lines_bart(params):
    src_path, keep_sect_num, sent_numbers = params

    def load_json(src_json):
        # print(src_json)
        paper = json.load(open(src_json))

        # if len(paper['sentences']) < 10 or sum([len(sent) for sent in paper['gold']]) < 10:
        #     return -1, 0, 0
        try:
            id = paper['filename']
        except:
            id = paper['id']

        # for sent in paper['sentences']:
        #     tokens = sent[0]
        # if (lower):
        #     tokens = [t.lower() for t in tokens]
        #     sent[0] = tokens

        # for i, sent in enumerate(paper['gold']):
        #     tokens = sent
        # if (lower):
        #     tokens = [t.lower() for t in tokens]
        #     paper['gold'][i] = tokens

        return ' '.join([' '.join(p[0]) for s_num, p in enumerate(paper['sentences'])
                if s_num in sent_numbers]), ' '.join([' '.join(g) for g in paper['gold']]), id

    paper_sents, paper_tgt, id = load_json(src_path)
    if paper_sents == -1:
        return None

    return {'id': id, 'src': paper_sents, 'tgt': paper_tgt}


def _format_to_lines(params):
    src_path, keep_sect_num = params

    def load_json(src_json):
        # print(src_json)
        paper = json.load(open(src_json))

        # if len(paper['sentences']) < 10 or sum([len(sent) for sent in paper['gold']]) < 10:
        #     return -1, 0, 0
        try:
            id = paper['filename']
        except:
            id = paper['id']

        # for sent in paper['sentences']:
        #     tokens = sent[0]
        # if (lower):
        #     tokens = [t.lower() for t in tokens]
        #     sent[0] = tokens

        # for i, sent in enumerate(paper['gold']):
        #     tokens = sent
        # if (lower):
        #     tokens = [t.lower() for t in tokens]
        #     paper['gold'][i] = tokens

        return paper['sentences'], paper['gold'], id
    paper_sents, paper_tgt, id = load_json(src_path)
    if paper_sents == -1:
        return None

    return {'id': id, 'src': paper_sents, 'tgt': paper_tgt}


def _format_longsum_to_lines_section_based(params):
    src_path, keep_sect_num = params

    def load_json(src_json, lower=True):
        paper = json.load(open(src_json))
        # main_sections = _get_main_sections(paper['sentences'])
        # try:
        # main_sections = _get_main_sections(paper['sentences'])
        # except:
        #     main_sections = _get_main_sections(paper['sentences'])
        # sections_text = [''.join([s for s in v if not s.isdigit()]).replace('.','').strip().lower() for v in main_sections.values()]

        # if len(paper['sentences']) < 10 or sum([len(sent) for sent in paper['gold']]) < 10:
        #     return -1, 0, 0

        try:
            id = paper['filename']
        except:
            id = paper['id']

        # for sent in paper['sentences']:
        #     tokens = sent[0]
        # if keep_sect_num:
        #     sent[1] = _get_section_text(sent[1], main_sections) if 'Abstract' not in sent[1] else sent[1]

        sections = []
        cur_sect = ''
        cur_sect_sents = []
        sections_textual = []
        for i, sent in enumerate(paper['sentences']):
            if not str(sent[0]).isdigit():
                sent = [0] + sent
            if i == 0:
                cur_sect = sent[2]
                sections_textual.append(sent[2])
                cur_sect_sents.append(sent)
                continue
            else:
                if cur_sect == sent[2]:
                    cur_sect_sents.append(sent)
                    if i == len(paper['sentences']) - 1:
                        sections.append(cur_sect_sents.copy())
                        cur_sect_sents.clear()
                        break
                else:
                    cur_sect = sent[2]
                    sections_textual.append(sent[2])
                    sections.append(cur_sect_sents.copy())
                    cur_sect_sents.clear()


        tgts = []
        ids = []
        for j, _ in enumerate(sections):
            tgts.append(paper['gold'])
            ids.append(id + "___" + str(sections_textual[j]))

        return sections, tgts, ids, sections_textual

    paper_sect_sents, paper_tgts, ids, sections_text = load_json(src_path)

    # if paper_sect_sents == -1:
    #     return None


    out = []
    for j, sect_sents in enumerate(paper_sect_sents):
        o = {}
        o['id'] = ids[j]
        o['src'] = sect_sents
        o['tgt'] = paper_tgts[j]
        out.append(o)

    return out, sections_text


## Other utils

def count_dots(txt):
    result = 0
    for char in txt:
        if char == '.':
            result += 1
    return result


def _get_section_id(sect, main_sections):
    if 'abstract' in sect.lower() or 'conclusion' in sect.lower() or 'summary' in sect.lower():
        return sect
    base_sect = sect
    sect_num = sect.split(' ')[0].rstrip('.')
    try:
        int(sect_num)
        return str(int(sect_num))
    except:
        try:
            float(sect_num)
            return str(int(float(sect_num)))
        except:
            if count_dots(sect_num) >= 2:
                sect_num = sect_num.split('.')[0]
                return str(sect_num)
            else:
                return base_sect


def _get_main_sections(sentences):
    out = {}

    for sent in sentences:
        sect_num = sent[1].split(' ')[0].rstrip('.')
        try:
            int(sect_num)
            out[str(sect_num)] = sent[1]
        except:
            pass
    return out


def _get_main_sections_textual(sentences):
    out = {}

    for sent in sentences:
        sect_first_term = sent[1].split(' ')[0].rstrip('.')
        try:
            int(sect_first_term)
            out[str(sect_first_term)] = sent[1]
        except:
            pass

    return out


def _get_section_text(sect, main_sections):
    if 'abstract' in sect.lower() or 'conclusion' in sect.lower() or 'summary' in sect.lower():
        return sect
    base_sect = sect
    sect_num = sect.split(' ')[0].rstrip('.')
    try:
        int(sect_num)
        return sect
    except:
        try:
            float(sect_num)
            int(sect_num.split('.')[0].strip())
            if sect_num.split('.')[0].strip() in main_sections.keys():
                return main_sections[sect_num.split('.')[0].strip()]
            else:
                return base_sect
        except:
            if count_dots(sect_num) >= 2:
                try:
                    int(sect_num.split('.')[0].strip())

                    if sect_num.split('.')[0].strip() in main_sections.keys():
                        return main_sections[sect_num.split('.')[0].strip()]
                    else:
                        return base_sect
                except:
                    return base_sect
            else:
                return base_sect


def check_path_existence(dir):
    if os.path.exists(dir):
        return
    else:
        os.makedirs(dir)


# greedy rg
def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    # import pdb;pdb.set_trace()
    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = (rouge_1 + rouge_2) / 2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            # return selected
            continue
        selected.append(cur_id)
        max_rouge = 0
    return sorted(selected)
