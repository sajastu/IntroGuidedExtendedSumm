import pickle

import numpy as np
import matplotlib.pyplot as plt


def give_numbers(preds, gold):
    final_dist_preds = {'0': [], '1': [], '2': [], '3': [], '4': []}
    final_dist_true = {'0': [], '1': [], '2': [], '3': [], '4': []}

    for paper_id, pred in preds.items():
        golds = gold[paper_id]
        dist_preds = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
        dist_true = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}
        for num_sent, sect in enumerate(pred):
            if 'tensor' not in str(sect):
                sect = str(sect)
                dist_preds[sect] += 1
                dist_true[str(golds[num_sent])] += 1
            else:
                dist_preds[str(sect.item())] += 1
                dist_true[str(golds[num_sent].item())] += 1

        total = list(dist_preds.values())
        totall_sum = np.sum(total, axis=0)
        dist_preds = {key: val/totall_sum for key, val in dist_preds.items()}

        total = list(dist_true.values())
        totall_sum = np.sum(total, axis=0)
        dist_true = {key: val / totall_sum for key, val in dist_true.items()}



        for sect_d, dist in dist_preds.items():
            final_dist_preds[sect_d].append(dist)
            final_dist_true[sect_d].append(dist_true[sect_d])

    final_dist_preds = {key: np.mean(value) for key, value in final_dist_preds.items()}

    final_dist_true = {key: np.mean(value) for key, value in final_dist_true.items()}
    out_dict = {'pred':{}, 'true':{}}

    out_dict['pred'] = final_dist_preds
    out_dict['true'] = final_dist_true

    return out_dict

def pie_chart(dict, filename):
    # Data to plot
    labels = 'Objective', 'Background', 'Method', 'Results', 'Other'
    # sizes = dict.values()
    sizes = [45, 41, 25, 13, 1]
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'red']
    explode = (0, 0.1, 0, 0, 0)  # explode 1st slice

    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)

    plt.axis('equal')
    plt.show()
    plt.savefig(filename)

# with open('section/preds_sects_arxiv_long_baseline.pickle', 'rb') as handle:
#     pred_sects = pickle.load(handle)
#
# with open('section/true_preds_sects_arxiv_long_baseline.pickle', 'rb') as handle:
#     true_sects = pickle.load(handle)

with open('section/preds_sects_arxiv_long.pickle', 'rb') as handle:
    pred_sects = pickle.load(handle)

with open('section/true_preds_sects_arxiv_long.pickle', 'rb') as handle:
    true_sects = pickle.load(handle)

stat_results = give_numbers(pred_sects, true_sects)

pie_chart(stat_results['pred'], 'pred.pdf')
# pie_chart(stat_results['true'], 'true.pdf')