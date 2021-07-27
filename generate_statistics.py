import sys, pickle, numpy, copy
import matplotlib.pyplot as plt
from itertools import repeat

pkl = {}
with open(sys.argv[1], 'rb') as f:
    pkl['train'] = pickle.load(f)

with open(sys.argv[2], 'rb') as f:
    pkl['test'] = pickle.load(f)

data = {}
discarded_sents = []
for s, d in pkl.items():
    data[s] = {
        'entity_types': {},
        'relation_types': {},
        'kb_entities': {},
        'entities':{}
    }
    if s == 'test': # this is done to preserve keys order in test/train
        for k in data[s].keys():
            data['test'][k] = dict(zip(
                data['train'][k].keys(),
                repeat(0, len(data['train'][k].keys()))
            ))
    for v in d:
        discard = False
        for e in v['entities'].values():
            try:
                emb_flag = e['embedding'].any() != None
            except:
                emb_flag = False
            if e['type'] != None and emb_flag:
                try:
                    data[s]['entities'][e['name']] += 1
                except:
                    data[s]['entities'][e['name']] = 1
                try:
                    data[s]['kb_entities'][e['id']] += 1
                except:
                    data[s]['kb_entities'][e['id']] = 1
                try:
                    data[s]['entity_types'][e['type']] += 1
                except:
                    data[s]['entity_types'][e['type']] = 1
            else:
               discard = True 
        for r in v['relations'].values():
            try:
                data[s]['relation_types'][r] += 1
            except:
                data[s]['relation_types'][r] = 1
        if discard:
            discarded_sents.append((v['sentence'], v['entities'], v['relations']))

# This is done to complete the train dict with elements only present in the test dict
tot = {**copy.deepcopy(data['test']), **copy.deepcopy(data['train'])}
for v in tot.values():
    for k in v.keys():
        v[k] = 0
data['train'] = {**tot, **data['train']}
data['test'] = {**tot, **data['test']}
#print(data['train']['entity_types'])
#print(data['test']['entity_types'])

print('> Discarded {} sentences, due to incomplete annotations.'.format(len(discarded_sents)))

try:
    data['train']['relation_types'].pop('NO_RELATION')
    data['test']['relation_types'].pop('NO_RELATION')
except:
    try:
        data['train']['relation_types'].pop('no_relation')
        data['test']['relation_types'].pop('no_relation')
    except:
        pass

#kb_top100 = {
#    'train': dict(sorted(data['train']['kb_entities'].items(), key=lambda x: x[1], reverse=True)[:100]),
#    'test': dict(sorted(data['test']['kb_entities'].items(), key=lambda x: x[1], reverse=True)[:100])
#}
#print(kb_top100)

th = 5
kb_supp = [i for i in data['train']['kb_entities'].items() if i[1]>=th]
print('> {} KB entities ({}%) have support >= {} in the train set.'.format(len(kb_supp), int(100*len(kb_supp)/len(data['train']['kb_entities'])), th))
#print(numpy.mean(list(data['train']['kb_entities'].values())))

for k in data['train'].keys():
    fig, axs = plt.subplots(2,1, figsize=(16, 12), sharex=True)
    for i, s in enumerate(['train', 'test']):
        x, y = zip(*list(data[s][k].items()))
        axs[i].set_ylim(0, max(y))
        axs[i].bar(x,y)
        axs[i].set_xticklabels(x, rotation='vertical')
    plt.tight_layout()
    plt.savefig(k)


# Train-Test overlapping
# entities



# Ambiguous entities statistics
