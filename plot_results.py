import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import json, numpy, sys

with open(sys.argv[1], 'r') as f:
    res = json.load(f)
with open(sys.argv[2], 'r') as f:
    res_kg = json.load(f)

fig, ax = plt.subplots(figsize=(16,9))
f1, f1_kg = {}, {}
for r,f in zip((res, res_kg), (f1, f1_kg)):
    for v in r.values():
        for k,c in v['scores'][0].items():
            if k == 'accuracy':
                try:
                    f[k].append(c)
                except:
                    f[k] = [c]
            elif k != 'weighted avg':
                try:
                    f[k].append(c['f1-score'])
                except:
                    f[k] = [c['f1-score']]
    try:
        f['micro avg'] += f['accuracy']
        f.pop('accuracy')
    except:
        f['micro avg'] = f['accuracy']
        f.pop('accuracy')
    print(list(f.keys()))
    classes, scores = list(zip(*f.items()))
    #errs = numpy.array(scores).std(-1)
    #scores = numpy.array(scores).mean(-1)
    #plt.plot(classes, scores)
    #lab = 'no graph embeddings' if f == f1 else 'graph embeddings'
    #plt.errorbar(classes, scores, yerr=errs, label=lab)
    #plt.xticks(rotation=90)
    ax.scatter(range(1,len(classes)+1), numpy.array(scores).mean(-1))
    ax.violinplot(scores)
    ax.set_xticks(range(1,len(classes)+1))
    ax.set_xticklabels(classes, rotation=90)
ax.legend([Patch(color='cornflowerblue'), Patch(color='orange')] ,['no graph embeddings', 'graph embeddings'])
fig.tight_layout()
plt.show()
