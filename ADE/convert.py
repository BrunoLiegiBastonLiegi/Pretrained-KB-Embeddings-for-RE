import pickle, sys
from transformers import AutoTokenizer

bert = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(bert)

with open(sys.argv[1], 'rb') as f:
    d = pickle.load(f)

for fold, data in d.items():
    for set, sents in data.items():
        for i, s in enumerate(sents):
            ents = {}
            name2span = {}
            for name, e in s['entities'].items():
                name2span[name] = e['span']
                ents[e['span']] = {
                    'id': '-'.join(e['concept']),
                    'name': name,
                    'type': e['type'],
                    'embedding': e['embedding']
                }
            rels = {}
            for tup, r in s['relations'].items():
                rels[(name2span[tup[0]], name2span[tup[1]])] = r['type']
            sents[i] = {
                'sentence': [s['sentence']['sentence'], tokenizer(s['sentence']['sentence'], add_special_tokens=False).tokens()],
                'entities': ents,
                'relations': rels
            }

with open(sys.argv[1].replace('.pkl', '_converted.pkl'), 'wb') as f:
    pickle.dump(d, f)
