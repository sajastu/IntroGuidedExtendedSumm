import glob
import json

import re

from tqdm import tqdm

from utils.rouge_score import evaluate_rouge

_RE_COMBINE_WHITESPACE = re.compile(r"\s+")

import scispacy
import spacy

nlp = spacy.load("en_core_sci_lg")
nlp.disable_pipes('ner', 'tagger', 'parser')
nlp.add_pipe(nlp.create_pipe('sentencizer'))

def tokenize_sent(text_src):
    doc_src = nlp(text_src)
    doc_src = list(doc_src.sents)
    toks = []
    for sent in doc_src:
        sent_toks = []
        for tok in sent:
            if len(tok.text.strip()) > 0 and '\\u' not in tok.text.strip():
                sent_toks.append(tok.text)
        toks.append(sent_toks)
    return toks

def tokenize_par_sents_(text_src):
    text_src = _RE_COMBINE_WHITESPACE.sub(" ", text_src).strip()
    doc_src = nlp(text_src)
    doc_src = list(doc_src.sents)
    sents = []
    for sent in doc_src:
        toks = []
        for tok in doc_src:
            if len(tok.text.strip()) > 0 and '\\u' not in tok.text.strip():
                toks.append(tok.text)
        sents.append(toks)
    return sents


for file in tqdm(glob.glob("/disk1/sajad/datasets/sci/longsumm/my-format-splits/test/*.json"), total=len(glob.glob("/disk1/sajad/datasets/sci/longsumm/my-format-splits/test/*.json"))):
    with open(file) as F:
        for li in F:
            paper = json.loads(li.strip())

    # abs_txt = paper["abstract"]
    #
    # abs_sents = tokenize_sent(abs_txt)
    # abs_tokenized = [[s, "0 Abstract", 0.0 , ' '.join(s), 1] for s in abs_sents]
    # paper["sentences"] = abs_tokenized + paper['sentences']

    for idx, sent in enumerate(paper['sentences']):

        rg_score = evaluate_rouge([' '.join(sent[0])], [' '.join([' '.join(g) for g in paper['gold']])])[2]
        paper['sentences'][idx][2] = rg_score


    with open(file, mode='w') as FF:
        json.dump(paper, FF)
