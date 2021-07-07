import sys, pickle, random

w = 100 - int(sys.argv[1])

with open('fewrel_train.pkl', 'rb') as f:
    train = pickle.load(f)
with open('fewrel_val.pkl', 'rb') as f:
    val = pickle.load(f)

train_ents, val_ents = {}, {}
for s in train:
    for e in s['entities'].values():
        try:
            train_ents[e['id']] += 1
        except:
            train_ents[e['id']] = 1

for s in val:
    for e in s['entities'].values():
        try:
            val_ents[e['id']] += 1
        except:
            val_ents[e['id']] = 1


overlap = list(train_ents.keys() & val_ents.keys())
new_train, new_val, pop = [], [], []

for i, s in enumerate(train):
    flag = True
    for e in s['entities'].values():
        if e['id'] in overlap:
            pass
        else:
            flag = False
            break
    if flag:
        new_train.append(s)
        pop.append(i)

for i in sorted(pop, reverse=True):
    del train[i]
pop = []

for i, s in enumerate(val):
    flag = True
    for e in s['entities'].values():
        if e['id'] in overlap:
            pass
        else:
            flag = False
            break
    if flag:
        new_val.append(s)
        pop.append(i)

for i in sorted(pop, reverse=True):
    del val[i]

new_train += random.choices(train, k=5000)
new_val += random.choices(val, k=int(w/100*len(new_val)))
random.shuffle(new_train)
random.shuffle(new_val)

with open('fewrel_train_{}%_overlap.pkl'.format(int(sys.argv[1])), 'wb') as f:
    pickle.dump(new_train, f)
with open('fewrel_val_{}%_overlap.pkl'.format(int(sys.argv[1])), 'wb') as f:
    pickle.dump(new_val, f)

print(len(train))
print(len(new_train))
print(len(val))
print(len(new_val))
