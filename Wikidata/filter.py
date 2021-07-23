import pickle, sys, random

with open(sys.argv[1], 'rb') as f:
    train_sents = pickle.load(f)
with open(sys.argv[2], 'rb') as f:
    val_sents = pickle.load(f)

sents = train_sents + val_sents

rels = {}
"""
for i,s in enumerate(sents):
    for r in s['relations'].values():
        try:
            rels[r] += 1
        except:
            rels[r] = 1
    print('{} / {}'.format(i,len(sents)), end='\r')

rels = sorted(rels.items(), key=lambda x: x[1], reverse=True)
rels = dict(rels[:10])


filtered = []
for i,s in enumerate(sents):
    discard = False
    for r in s['relations'].values():
        try:
            tmp = rels[r]
        except:
            discard = True

    if discard:
        pass
    else:
        filtered.append(s)
    print('{} / {}'.format(i,len(sents)), end='\r')

print('\n', len(filtered))
"""
ents = {}
for i,s in enumerate(sents):
    for e in s['entities'].values():
        try:
            ents[e['id']] += 1
        except:
            ents[e['id']] = 1
    print('{} / {}'.format(i,len(sents)), end='\r')

ents = sorted(ents.items(), key=lambda x: x[1], reverse=True)
ents = dict(ents[:1000])

filtered = []
for i,s in enumerate(sents):
    discard = False
    for e in s['entities'].values():
        try:
            tmp = ents[e['id']]
        except:
            discard = True

    if discard:
        pass
    else:
        filtered.append(s)
    print('{} / {}'.format(i,len(sents)), end='\r')

print('\n', len(filtered))

val_split = 0.2

val = []
random.shuffle(filtered)
for i in range(int(val_split*len(filtered))):
    val.append(filtered.pop(-1))
#for i in range(int(val_split*len(filtered))):
 #   random.shuffle(filtered)
 #   val.append(filtered.pop(-1))
 #   print('{} / {}'.format(i,int(val_split*len(filtered))), end='\r')

print(len(val), len(filtered))

with open(sys.argv[1].replace('.pkl', '_filtered.pkl'), 'wb') as f:
    pickle.dump(filtered, f)

with open(sys.argv[2].replace('.pkl', '_filtered.pkl'), 'wb') as f:
    pickle.dump(val, f)
