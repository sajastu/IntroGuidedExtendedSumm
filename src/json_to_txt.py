import json

filename="results_LSUM-test-longformer-bertsum.p.json"
oracles = []
golds = []
ents = []
with open('txt_out/' + filename) as F:
    for l in F:
        try:
            instance = json.loads(l.strip())
        except:
            import pdb;pdb.set_trace()
        ents.append(instance)

ents = sorted(ents, key = lambda i: i['p_id'])

for e in ents:
    oracles.append(e['pred'])
    golds.append(e['gold'])

with open('txt_out/' + filename + '.pred', mode='w') as D:
    for o in oracles:
        D.write(o.strip())
        D.write('\n')

with open('txt_out/' + filename + '.gold', mode='w') as D:
    for o in golds:
        D.write(o.strip())
        D.write('\n')