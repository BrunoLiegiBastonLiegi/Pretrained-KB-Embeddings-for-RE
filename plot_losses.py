import matplotlib.pyplot as plt
import sys, pickle

with open(sys.argv[1], 'rb') as f:
    d = pickle.load(f)

for i in ('ner', 'ned1', 'ned2', 're'):
    fig = plt.figure(1, figsize=(12,12))
    plt.ylim((
        min(min(d['test'][i]), min(d['train'][i])),
        max(d['test'][i])
    ))
    plt.plot(d['train'][i])
    k = int(len(d['train'][i]) / len(d['test'][i])
)
    plt.plot(
        [k*j for j in range(len(d['test'][i]))],
        d['test'][i]
    )
    name = input('Save plots to:')
    plt.savefig(name + '_{}.png'.format(i))
    plt.clf()
    #plt.show()
