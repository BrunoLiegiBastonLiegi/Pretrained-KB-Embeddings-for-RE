import sys, pickle, numpy, copy, tqdm, re
import matplotlib.pyplot as plt
from itertools import repeat

pkl = {}
with open(sys.argv[1], 'rb') as f:
    pkl['train'] = pickle.load(f)

with open(sys.argv[2], 'rb') as f:
    pkl['test'] = pickle.load(f)

def normalize(text):
    return re.sub(r'[^\w\s]','', text.lower()).strip()
    
dir = re.search('.+<?\/', sys.argv[1]).group(0)
assert dir == re.search('.+<?\/', sys.argv[2]).group(0)

data = {}
discarded_sents = []
common = {'kb2txt':{}, 'txt2kb':{}}
for s, d in pkl.items():
    data[s] = {
        'entity_types': {},
        'relation_types': {},
        'kb_entities': {},
        'entities': {},
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
                for k, l in zip(('entities', 'kb_entities', 'entity_types'), ('name', 'id', 'type')):
                    try:
                        data[s][k][e[l]] += 1
                    except:
                        data[s][k][e[l]] = 1
                for k, l in zip(('kb2txt', 'txt2kb'), (('id', 'name'), ('name', 'id'))):
                    try:
                        if k == 'kb2txt':
                            common[k][e[l[0]]][normalize(e[l[1]])] += 1
                        elif k == 'txt2kb':
                            common[k][normalize(e[l[0]])][e[l[1]]] += 1
                    except:
                        try:
                            if k == 'kb2txt':
                                common[k][e[l[0]]][normalize(e[l[1]])] = 1
                            elif k == 'txt2kb':
                                common[k][normalize(e[l[0]])][e[l[1]]] = 1
                        except:
                            if k == 'kb2txt':
                                common[k][e[l[0]]] = {normalize(e[l[1]]): 1}
                            elif k == 'txt2kb':
                                common[k][normalize(e[l[0]])] = {e[l[1]]: 1}
            else:
               discard = True 
        for r in v['relations'].values():
            try:
                data[s]['relation_types'][r] += 1
            except:
                data[s]['relation_types'][r] = 1
        if discard:
            discarded_sents.append((v['sentence'], v['entities'], v['relations']))
        

print('> Discarded {} sentences, due to incomplete annotations.'.format(len(discarded_sents)))

            
# Calculate the number of examples available for entities and KB entities
print('TRAINING EXAMPLES AVAILABLE')
th = 3
kb_ex = [i for i in data['train']['kb_entities'].items() if i[1]>=th]
print('> {} KB entities ({}%) appear a number of times >= {} in the train set.'.format(len(kb_ex), int(100*len(kb_ex)/len(data['train']['kb_entities'])), th))
ent_ex = [i for i in data['train']['entities'].items() if i[1]>=th]
print('> {} text entities ({}%) appear a number of times >= {} in the train set.'.format(len(ent_ex), int(100*len(ent_ex)/len(data['train']['entities'])), th))

# This is done to complete the train dict with elements only present in the test dict
tot = {}
for k in data['train'].keys():
    keys = list(data['test'][k].keys()) + list(data['train'][k].keys())
    tot[k] = dict(zip(
        keys,
        repeat(0, len(keys))
    ))
    data['train'][k] = {**tot[k], **data['train'][k]}
    data['test'][k] = {**tot[k], **data['test'][k]}


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

print(data['train']['entity_types'])

#for k in data['train'].keys():
for k in ('entity_types', 'relation_types'):
    fig, axs = plt.subplots(2,1, figsize=(16, 12), sharex=True)
    for i, s in enumerate(['train', 'test']):
        x, y = zip(*list(data[s][k].items()))
        axs[i].set_ylim(0, max(y))
        axs[i].bar(x,y)
        axs[i].set_xticklabels(x, rotation='vertical')
    plt.tight_layout()
    plt.savefig(dir+k)


# Test support in train
print('TEST SET SUPPORT')
th = 1
kb_supp_tot = {k:data['train']['kb_entities'][k] for k in data['test']['kb_entities'].keys()}
kb_supp = {k:data['train']['kb_entities'][k] for k in data['test']['kb_entities'].keys() if data['train']['kb_entities'][k] >= th}
print('> {} KB entities ({}%) in the test set have support >= {} in the train set.\n>> Average support: {:.2f}'.format(len(kb_supp), int(100*len(kb_supp)/len(data['test']['kb_entities'])), th, numpy.mean(list(kb_supp_tot.values()))))
ent_supp = {k:data['train']['entities'][k] for k in data['test']['entities'].keys() if data['train']['entities'][k] >= th}
ent_supp_tot = {k:data['train']['entities'][k] for k in data['test']['entities'].keys()}
print('> {} text entities ({}%) in the test set have support >= {} in the train set.\n>> Average support: {:.2f}'.format(len(ent_supp), int(100*len(ent_supp)/len(data['test']['entities'])), th, numpy.mean(list(ent_supp_tot.values()))))


# Ambiguous entities statistics
data['common'] = common
print('KB TO TEXT MAPPING')
kb2txt = { k:v for k,v in data['common']['kb2txt'].items() if len(v) > 1 }
print('> {} KB entities ({}%) have multiple text representations.\n>> Average ambiguity: {:.2f}'.format(len(kb2txt), int(100*len(kb2txt)/len(data['common']['kb2txt'])), numpy.mean(list(map(len,data['common']['kb2txt'].values())))))
#print(kb2txt)
print('TEXT TO KB MAPPING')
txt2kb = { k:v for k,v in data['common']['txt2kb'].items() if len(v) > 1 }
print('> {} text entities ({}%) are ambiguous and refer to different concepts.\n>> Average ambiguity: {:.2f}'.format(len(txt2kb), int(100*len(txt2kb)/len(data['common']['txt2kb'])), numpy.mean(list(map(len,data['common']['txt2kb'].values())))))
