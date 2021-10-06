import pickle

with open('wikidata_train.pkl', 'rb') as f:
    train = pickle.load(f)

with open('wikidata_val.pkl', 'rb') as f:
    test = pickle.load(f)

r_stat = {}
for s in train+test:
    for r in s['relations'].values():
        try:
            r_stat[r] += 1
        except:
            r_stat[r] = 1

n = 5
rels = dict(sorted(r_stat.items(), key=lambda x: x[1], reverse=True)[:n])

for t in (train, test):
    for s in t:
        new_rel = {}
        for k,v in s['relations'].items():
            if v in rels.keys():
                new_rel[k] = v
        s['relations'] = new_rel

train = [s for s in train if len(s['relations']) > 0]q
test = [s for s in test if len(s['relations']) > 0]

with open('wikidata_train_top_{}_rel.pkl'.format(n), 'wb') as f:
    pickle.dump(train, f)

with open('wikidata_val_top_{}_rel.pkl'.format(n), 'wb') as f:
    pickle.dump(test, f)
