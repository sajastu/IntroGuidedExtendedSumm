# # # import re
# # #
# # # import spacy
# # #
# # # from utils.rouge_score import evaluate_rouge
# # # from utils.rouge_utils import greedy_selection
# # #
# # # nlp = spacy.load("en_core_sci_lg")
# # # nlp.disable_pipes('ner', 'tagger', 'parser')
# # # nlp.add_pipe(nlp.create_pipe('sentencizer'))
# # #
# # # _RE_COMBINE_WHITESPACE = re.compile(r"\s+")
# # #
# # # def tokenize_par_sents_(text_src):
# # #     text_src = _RE_COMBINE_WHITESPACE.sub(" ", text_src).strip()
# # #     doc_src = nlp(text_src)
# # #     doc_src = list(doc_src.sents)
# # #     sents = []
# # #     for sent in doc_src:
# # #         sents.append(sent.text)
# # #     return sents
# # #
# # # def tokenize_sent(text_src):
# # #     doc_src = nlp(text_src)
# # #     toks = []
# # #     for tok in doc_src:
# # #         if len(tok.text.strip()) > 0 and '\\u' not in tok.text.strip():
# # #             toks.append(tok.text)
# # #     return toks
# # #
# # #
# # # summary1 = "In this article, the authors discuss herd immunity, including its definition and examples of successful vaccine-induced herd immunity. Implications in the setting of COVID-19 infection are also discussed."
# # #
# # # # summary2="A bacterial infection is not just an unpleasant experience -- it can also be a major health problem. Some bacteria develop resistance to otherwise effective treatment with antibiotics. Therefore, researchers are trying to develop new types of antibiotics that can fight the bacteria, and at the same time trying to make the current treatment with antibiotics more effective."
# # #
# # # src = "Herd immunity, also known as indirect protection, community immunity, or community protection, refers to the protection of susceptible individuals against an infection when a sufficiently large proportion of immune individuals exist in a population. In other words, herd immunity is the inability of infected individuals to propagate an epidemic outbreak due to lack of contact with sufficient numbers of susceptible individuals. It stems from the individual immunity that may be gained through natural infection or through vaccination. The term herd immunity was initially introduced more than a century ago. In the latter half of the 20th century, the use of the term became more prevalent with the expansion of immunization programs and the need for describing targets for immunization coverage, discussions on disease eradication, and cost-effectiveness analyses of vaccination programs. Eradication of smallpox and sustained reductions in disease incidence in adults and those who are not vaccinated following routine childhood immunization with conjugated Haemophilus influenzae type B and pneumococcal vaccines are successful examples of the effects of vaccine-induced herd immunity."
# # #
# # #
# # # # summary1 = summary1 + ' ' + summary2
# # #
# # # src_sents = tokenize_par_sents_(src)
# # # tgt_sents = tokenize_par_sents_(summary1)
# # #
# # #
# # # labels = greedy_selection([tokenize_sent(s) for s in src_sents],  [tokenize_sent(s) for s in tgt_sents], summary_size=5)
# # #
# # # oracle = ' '.join([src_sents[i] for i in labels])
# # # import pdb;pdb.set_trace()
# # # print(evaluate_rouge([oracle], [summary1]))
# # import collections
# # import glob
# # import os
# # import pickle
# # from multiprocessing.pool import Pool
# #
# # import torch
# # from tqdm import tqdm
# #
# #
# # def modify_labels_multi_section_based(params):
# #     src_f, dest_file, labels = params
# #     bert_files_labels = labels[dest_file]
# #
# #     print(f"Processing: {dest_file}")
# #     # if os.path.exists(PT_DIRS_DEST + f.split('/')[-1]):
# #     #     return
# #     instances = torch.load(src_f)
# #     for inst_idx, instance in enumerate(instances):
# #         paper_id = instance['paper_id'].split('___')[0]
# #         chunk_num = instance['paper_id'].split('___')[1]
# #
# #         try:
# #             instance_labels = bert_files_labels[paper_id + '___' + chunk_num]
# #             instance['sent_labels'] = instance_labels
# #         except:
# #             continue
# #     torch.save(instances, dest_file.replace('512-whole-segmented-seqLabelled-20', '512-whole-segmented-seqLabelled-20-full'))
# #
# # for se in ["test", "val"]:
# #     favorite_color = pickle.load(open(f"sent_labels_files/pubmedL/{se}.labels.p", "rb"))
# #     tmp_list = []
# #     for j, f in enumerate(glob.glob(os.path.join("/disk1/sajad/datasets/sci/pubmed-long//bert-files/512-whole-segmented-seqLabelled-20-1/") + se + '*.pt')):
# #         #################################
# #         ########## DEBUG #############
# #         #################################
# #
# #         tmp_list.append((f, "/disk1/sajad/datasets/sci/pubmed-long//bert-files/512-whole-segmented-seqLabelled-20/"+f.split('/')[-1], favorite_color))
# #         # modify_labels_multi_section_based(tmp_list[-1])
# #
# #     pool = Pool(24)
# #     for d in tqdm(pool.imap_unordered(modify_labels_multi_section_based, tmp_list), total=len(tmp_list)):
# #         pass
#
#
# # bs=['5', '0', '3', '1', '1', '2', '0', '3', '3', '1', '3', '0', '0', '2', '1', '1', '2', '0', '2', '1', '1', '1', '1', '2', '2', '2', '2', '2', '1', '3']
# bs=['0', '0', '0', '1', '1', '4', '1', '0', '0', '0', '1', '1', '2', '1', '2']
# #{'5': 0.1, '4': 0.03333333333333333, '1': 0.5333333333333333, '3': 0.06666666666666667, '2': 0.1, '0': 0.16666666666666666}
#
#
# # ours= ['5', '0', '2', '2', '0', '1', '1', '0', '0', '3', '1', '1', '1', '1', '0', '3', '1', '1', '1', '1', '2', '3', '1', '0', '2', '2', '2', '0', '0', '5']
# ours= ['0', '0', '0', '0', '1', '1', '1', '1', '1', '4', '4', '1', '2', '1', '0']
# #{'5': 0.06666666666666667, '1': 0.4666666666666667, '2': 0.43333333333333335, '0': 0.03333333333333333}
#
#
# # gold = [2, 2, 0, 0, 0, 1, 1, 2, 1, 3, 1, 3, 1, 1, 2, 3, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 2, 2, 2, 2]
# gold = [1, 0, 2, 0, 1, 2, 2, 0, 1, 1, 2, 0, 1, 1, 1]
#
# #{'5': 0.06666666666666667, '2': 0.43333333333333335, '0': 0.1, '1': 0.36666666666666664, '3': 0.03333333333333333}
#
# dist = {}
# for g in gold:
#     if str(g) not in dist:
#         dist[str(g)] = 1
#     else:
#         dist[str(g)] += 1
# print('----')
# print('Oracle:')
# dist_per = {}
# for k, g in dist.items():
#     dist_per[k] = g / sum(dist.values())
#
# print(dist_per)
# print('----')
#
# dist = {}
# for g in bs:
#     if str(g) not in dist:
#         dist[str(g)] = 1
#     else:
#         dist[str(g)] += 1
# print('----')
#
# print('BS:')
# dist_per={}
# for k, g in dist.items():
#     dist_per[k] = g / sum(dist.values())
#
# print(dist_per)
# print('----')
#
#
# dist={}
#
# for g in ours:
#     if str(g) not in dist:
#         dist[str(g)] = 1
#     else:
#         dist[str(g)] += 1
# print('----')
#
# print('Ours:')
# dist_per={}
# for k, g in dist.items():
#     dist_per[k] = g / sum(dist.values())
#
# print(dist_per)
#
# print('----')




#####################################################
# import glob
# import shutil
#
# from tqdm import tqdm
#
# from prepro.data_builder import check_path_existence

# all_files = glob.glob("/disk1/sajad/twitter-data/*.txt")
# all_files = [f.split('/')[-1].replace('.txt', '')  for f in all_files]
# # check_path_existence("/disk1/sajad/twitter-rem/")
# for f in tqdm(glob.glob("/disk1/sajad/tf_records/*"), total=len(glob.glob("/disk1/sajad/tf_records/*"))):
#     # import pdb;
#     # pdb.set_trace()
#     if f.split('.')[-1] in all_files:
#         all_files.remove(f.split('.')[-1])
#
# for f in all_files:
#     shutil.copy('/disk1/sajad/twitter-data/' + f + '.txt', "/disk1/sajad/twitter-rem/")


# folders = ['barolo', 'brunello', 'chianti', 'barbaresco']
#
# f_pointer = 0
# for idx, f in tqdm(enumerate(glob.glob("/disk1/sajad/twitter-rem/*.txt")), total=len(glob.glob("/disk1/sajad/twitter-rem/*.txt"))):
#     if f_pointer==4:
#         f_pointer=0
#     shutil.move(f, "/disk1/sajad/twitter-rem/" + folders[f_pointer])
#     f_pointer+=1



arr=[1,2,3,4,5,6,7,8,9,10]

print(arr[2:20])


