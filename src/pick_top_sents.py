import pickle
from multiprocessing.pool import Pool

from tqdm import tqdm

from prepro.data_builder import LongformerData


def _mult_top_sents(params):

    sent_scores, p_id, p_src, p_sent_numbers, p_sent_sent_sects_true, bert, idx= params
    indices = [i for i in range(len(sent_scores))]
    print("idx {}; paper {}".format(idx, p_id))

    zip_sents_score = zip(indices, sent_scores, p_sent_sent_sects_true)

    sent_scores = sorted(zip_sents_score, key=lambda element: (element[2], -element[1]))

    sent_scores_0 = [s for s in sent_scores if s[-1] == 0]
    sent_scores_1 = [s for s in sent_scores if s[-1] == 1]
    sent_scores_2 = [s for s in sent_scores if s[-1] == 2]
    sent_scores_3 = [s for s in sent_scores if s[-1] == 3]
    sections = []
    pointers = {}
    sent_scores_dict = {}
    checked_full = {}

    if len(sent_scores_0) > 0:
        sent_scores_dict[0] = sent_scores_0
        sections.append(0)
        pointers[0] = 0
        checked_full[0] = False

    if len(sent_scores_1) > 0:
        sent_scores_dict[1] = sent_scores_1
        sections.append(1)
        pointers[1] = 0
        checked_full[1] = False



    if len(sent_scores_2) > 0:
        sent_scores_dict[2] = sent_scores_2
        sections.append(2)
        pointers[2] = 0
        checked_full[2] = False



    if len(sent_scores_3) > 0:
        sent_scores_dict[3] = sent_scores_3
        sections.append(3)
        pointers[3] = 0
        checked_full[3] = False



    section_pointer = 0
    _pred = []
    paper_sampling_sent_numbers = []
    paper_sent_sects = []

    while bert.cal_token_len_prep(_pred) <= 2500:
        try:
            pred_item = sent_scores_dict[sections[section_pointer]][pointers[section_pointer]]
        except:
            checked_full[sections[section_pointer]]=True

            # sections.remove(section_pointer)
            if sections[section_pointer] == sections[-1]:
                section_pointer = 0
            else:
                print('here {}'.format(sections[section_pointer]))
                section_pointer += 1
                full_flag = True
                for k, v in checked_full.items():
                    full_flag = v and full_flag

                if full_flag:
                    break
            continue
        # import pdb;
        # pdb.set_trace()

        pointers[section_pointer] += 1
        _pred.append((p_src[pred_item[0]], pred_item[0]))
        paper_sampling_sent_numbers += [p_sent_numbers[pred_item[0]]]
        paper_sent_sects += [p_sent_sent_sects_true[pred_item[0]]]

        if section_pointer == len(sections) - 1:
            section_pointer = 0
        else:
            section_pointer += 1



        # if bert.cal_token_len_prep(_pred) > 2500:
        #     break
    _pred = _pred[:-1]
    # print("processed {}".format(p_id))
    # print("----")
    _pred = sorted(_pred, key=lambda x: x[1])
    return p_id, sorted(paper_sampling_sent_numbers)


top_pick = pickle.load(open("save_lists/pubmedL-test-scibert-bertsum-top-sents.p", "rb"))

pool = Pool(24)
preds_sent_numbers = {}

bert = LongformerData()
paper_sent_scores = []
for idx, (p_id, vals) in enumerate(top_pick.items()):
    # if p_id=='PMC6791213.nxml':
    paper_sent_scores.append(vals + (bert,idx))
    # _mult_top_sents(paper_sent_scores[-1])

# paper_sent_scores = paper_sent_scores[3549:]
# for idx, vals in tqdm(enumerate(paper_sent_scores), total=len(paper_sent_scores)):
#     _mult_top_sents(vals)
#
for d in tqdm(pool.imap_unordered(_mult_top_sents, paper_sent_scores), total=len(paper_sent_scores)):
    preds_sent_numbers[d[0]] = d[1]
pool.close()
pool.join()
#
pickle.dump(preds_sent_numbers, open("second_phase/pubmedL-test-scibert-bertsum.p", "wb"))