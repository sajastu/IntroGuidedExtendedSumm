
import json
import csv

sample_ids = []
with open('src/arxivL-samples.txt') as fR:
    for l in fR:
        sample_ids.append(l.strip())

papers = []
flg = False
p = {}
with open('src/arxivL-anal.json') as fR:
    for l in fR:
        if 'paper_id' in l:
            # if str(l.replace('"paper_id": ', '').replace(',', '').strip().replace('"', '')) not in sample_ids:
            flg = True
            p['id'] = str(l.replace('"paper_id": ', '').replace(',', '').strip().replace('"', ''))
            continue

        if flg and 'pred_1' in l:
            p['pred_1'] = l.replace('"pred_1": ', '').replace('"', '').strip()
        elif flg and 'pred_2' in l:
            p['pred_2'] = l.replace('"pred_2": ', '').replace('"', '').strip()
        elif flg and 'target' in l:
            p['target'] = l.replace('"target": ', '').replace('"', '').strip()
            papers.append(p.copy())
            p={}
            flg=False

fields = ['pred_1', 'pred_2', 'target']
rows = []
# name of csv file

papers = [p for p in papers if p['id'] not in sample_ids][:40]

filename = "arxivL-annotation-2.csv"

for p in papers:
    try:
        rows.append([p['pred_1'], p['pred_2'], p['target']])
    except:
        print('here')
        rows.append([p['pred_1'], ' ', p['target']])

# writing to csv file
with open(filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the fields
    csvwriter.writerow(fields)

    # writing the data rows
    csvwriter.writerows(rows)