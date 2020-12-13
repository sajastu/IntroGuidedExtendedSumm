import collections
import pickle
from multiprocessing.pool import Pool

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')

saved_list_name = "LSUM-test-longformer-multi.p"
saved_list = pickle.load(open("save_lists/" + saved_list_name, "rb"))


def _mult_sent_bert(val):
    p_id, sent_scores, paper_srcs, paper_tgt, sent_sects_whole_true, sent_sects_whole_true, section_textual, sent_true_labels, sent_sectwise_rg = val

    out = []
    for sent in paper_srcs:
        out.append(model.encode(sent))
    return out, p_id


files = []
sentbert_embs = collections.defaultdict(dict)
for s, val in tqdm(saved_list.items(), total=len(saved_list)):
    files.append((val))

pool = Pool(24)

for f in tqdm(files, total=len(files)):
    embs, p_id = _mult_sent_bert(f)
    sentbert_embs[p_id] = embs

pickle.dump(sentbert_embs, open("save_lists/embs-"+saved_list_name, "wb"))
