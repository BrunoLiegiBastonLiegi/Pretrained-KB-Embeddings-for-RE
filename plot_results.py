import matplotlib.pyplot as plt
import json, numpy, sys, pickle, argparse, re
from utils import collect_scores, violin_plot, collect_confusion_m, confusion_m_heat_plot
from dataset import Stat

parser = argparse.ArgumentParser(description='Plot results.')
parser.add_argument('--res', nargs='+')
parser.add_argument('--compare', nargs='+')
parser.add_argument('--stat')
args = parser.parse_args()

supp = None
if args.stat != None:
    with open(args.stat, 'r') as f:
        stat = json.load(f)
    supp = { k: stat['train']['relation_types'][k] 
             for k in stat['test']['relation_types'].keys() }

if args.res != None:
    res, res_kg, rels = (args.res[0::2], args.res[1::2], []) if args.res != None else ([], [], [])
    for r, rkg in zip(res, res_kg):
        fig, ax = plt.subplots(1, 1, figsize=(16,9))
        rr, rrkg = collect_scores(r), collect_scores(rkg)
        rels.append(list(rr.keys())[:-2])
        sp = { k: supp[k] for k in rr.keys() if k not in {'micro avg', 'macro avg'}} if supp != None else None
        violin_plot(rr, rrkg, support=sp)
        fig.tight_layout()
        plt.show()
    for r, rkg, rel in zip(res, res_kg, rels):
        fig, ax = plt.subplots(1, 3, figsize=(16,9))
        [ print(numpy.asarray(i).shape) for i in collect_confusion_m(rkg)]
        print('MEAN:\n',numpy.mean(collect_confusion_m(r), axis=0))
        print('MEAN:\n',numpy.mean(collect_confusion_m(rkg), axis=0))
        m, mkg = numpy.mean(collect_confusion_m(r), axis=0), numpy.mean(collect_confusion_m(rkg), axis=0)
        confusion_m_heat_plot(m-mkg, rel, ax=ax[2])
        numpy.fill_diagonal(m, 0.) # manually set diagonal to zero to better see off-diagonal elements
        numpy.fill_diagonal(mkg, 0.) # manually set diagonal to zero to better see off-diagonal elements
        print(m.shape, mkg.shape)
        print(len(rel), rel)
        confusion_m_heat_plot(m, rel, ax=ax[0])
        confusion_m_heat_plot(mkg, rel, ax=ax[1])
        fig.tight_layout()
        plt.show()
        
if args.compare != None:
    res, res_kg = (args.compare[0::2], args.compare[1::2]) if args.compare != None else ([], [])
    rr, c = [], []
    for r, rkg in zip(res, res_kg):
        rr.append((collect_scores(r), collect_scores(rkg)))
        c.append(set(rr[-1][0].keys()))
    c = c[0].intersection(*c[1:])
    c.remove('micro avg')
    c.remove('macro avg')
    c = list(c)
    c.append('micro avg')
    c.append('macro avg')
    supp = { k: supp[k] for k in c if k not in {'micro avg', 'macro avg'}} if supp != None else None
    fig, axes = plt.subplots(1,len(rr), figsize=(21,9))
    ylim = 1
    for i, r in enumerate(rr):
        violin_plot(*r, ax=axes[i], classes=c, support=supp)
        axes[i].set_title(res[i].replace('results_','').replace('.json',''))
        ylim = min(axes[i].get_ylim()[0], ylim)
    for i in range(len(rr)):
        axes[i].set_ylim((ylim,1.))
    plt.show()

