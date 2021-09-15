import matplotlib.pyplot as plt
import json, numpy, sys

with open(sys.argv[1], 'r') as f:
    res = json.load(f)
with open(sys.argv[2], 'r') as f:
    res_kg = json.load(f)

print(list(res['run_1']['scores'][0].keys()))

f1, f1_kg = {}, {}
for r,f in zip((res, res_kg), (f1, f1_kg)):
    for v in r.values():
        for k,c in v['scores'][0].items():
            if k not in {'accuracy', 'weighted avg'}:
                try:
                    f[k].append(c['f1-score'])
                except:
                    f[k] = [c['f1-score']]


    classes, scores = list(zip(*f.items()))
    scores = numpy.array(scores).mean(-1)
    errs = numpy.array(scores).std(-1)
    print(scores)
    print(errs)
    plt.plot(classes, scores)
plt.show()
