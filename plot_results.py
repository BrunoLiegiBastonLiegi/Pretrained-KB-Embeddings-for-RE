import matplotlib.pyplot as plt
import json, numpy, sys, pickle, argparse, re
from utils import collect_results, violin_plot
from dataset import Stat

parser = argparse.ArgumentParser(description='Plot results.')
parser.add_argument('--res', nargs='+')
parser.add_argument('--compare', nargs='+')
parser.add_argument('--stat', nargs='+')
args = parser.parse_args()

supp = None
if args.stat != None:
    assert len(args.stat) == 2
    with open(args.stat[0], 'rb') as f:
        train = pickle.load(f)
    with open(args.stat[1], 'rb') as f:
        test = pickle.load(f)
    stat = Stat(train, test)
    stat.scan()
    supp = { k: stat.stat['train']['relation_types'][k] 
             for k in stat.stat['test']['relation_types'].keys() }

if args.res != None:
    res, res_kg = (args.res[0::2], args.res[1::2]) if args.res != None else ([], [])
    for r, rkg in zip(res, res_kg):
        fig, ax = plt.subplots(1, 1, figsize=(16,9))
        rr, rrkg = collect_results(r), collect_results(rkg)
        sp = { k: supp[k] for k in rr.keys() if k not in {'micro avg', 'macro avg'}} if supp != None else None
        violin_plot(rr, rrkg, support=sp)
        plt.show()

if args.compare != None:
    res, res_kg = (args.compare[0::2], args.compare[1::2]) if args.compare != None else ([], [])
    rr, c = [], []
    for r, rkg in zip(res, res_kg):
        rr.append((collect_results(r), collect_results(rkg)))
        c.append(set(rr[-1][0].keys()))
    c = c[0].intersection(*c[1:])
    c.remove('micro avg')
    c.remove('macro avg')
    c = list(c)
    c.append('micro avg')
    c.append('macro avg')
    supp = { k: supp[k] for k in c if k not in {'micro avg', 'macro avg'}} if supp != None else None
    fig, axes = plt.subplots(1,len(rr), figsize=(21,9))
    for i, r in enumerate(rr):
        violin_plot(*r, ax=axes[i], classes=c, support=supp)
        axes[i].set_title(res[i].replace('results_','').replace('.json',''))
    plt.show()

