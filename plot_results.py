import matplotlib.pyplot as plt
from matplotlib import rc
import json, numpy, sys, pickle, argparse, re
from utils import collect_scores, violin_plot, collect_results, collect_confusion_m, confusion_m_heat_plot, collect_PR_curve, PR_curve_plot, f1_var_supp_correlation
from dataset import Stat

parser = argparse.ArgumentParser(description='Plot results.')
parser.add_argument('--res', nargs='+')
parser.add_argument('--compare', nargs='+')
parser.add_argument('--stat')
args = parser.parse_args()

font = {'size': 30}
rc('font', **font)

supp = None
if args.stat != None:
    with open(args.stat, 'r') as f:
        stat = json.load(f)
    supp = { k: stat['train']['relation_types'][k] 
             for k in stat['test']['relation_types'].keys() }
    t_supp = stat['test']['relation_types']

if args.res != None:
    res, res_kg, rels = (args.res[0::2], args.res[1::2], []) if args.res != None else ([], [], [])
    for r, rkg in zip(res, res_kg):
        fig, ax = plt.subplots(1, 1, figsize=(32,16), dpi=400)
        # collect data
        rr, m, pr = collect_results(r)
        rrkg, mkg, pr_kg = collect_results(rkg)
        # violin
        #rr, rrkg = collect_scores(r), collect_scores(rkg)
        rels.append(list(rr.keys())[:-2])
        sp = { k: supp[k] for k in rr.keys() if k not in {'micro avg', 'macro avg'}} if supp != None else None
        violin_plot(rr, rrkg, classes=None, support=sp, ax=ax)
        ax.set_ylabel(r'$F1$')
        plt.tight_layout()
        if input('> Save figure? (y/n)\n') == 'y':
            plt.savefig(input('> Save figure to:\n'), format='pdf')
        plt.show()
        f1_var_supp_correlation(rr, rrkg, rel_supp=sp)
        try:
            a
            # P-R curve
            font = {'size': 30}
            rc('font', **font)
            fig, ax = plt.subplots(1, 2, figsize=(32,16), dpi=400)
            #pr, pr_kg = collect_PR_curve(r), collect_PR_curve(rkg)
            print('>> PR Curve Baseline')
            PR_curve_plot(*pr, ax=ax)
            print('>> PR Curve with Graph Embedding')
            PR_curve_plot(*pr_kg, ax=ax)
            plt.tight_layout()
            if input('> Save figure? (y/n)\n') == 'y':
                plt.savefig(input('> Save figure to:\n'), format='pdf')
            plt.show()
        except:
            print('> PR curve not available.')
    print('>> Confusion Matrix')
    for r, rkg, rel in zip(res, res_kg, rels):
        fig, ax = plt.subplots(1, 1, figsize=(16,9), dpi=400)
        #[ print(numpy.asarray(i).shape) for i in collect_confusion_m(rkg)]
        #print('MEAN:\n',numpy.mean(collect_confusion_m(r), axis=0))
        #print('MEAN:\n',numpy.mean(collect_confusion_m(rkg), axis=0))
        m, mkg = numpy.mean(m, axis=0), numpy.mean(mkg, axis=0)
        #confusion_m_heat_plot(m-mkg, rels=rel, ax=ax[2])
        numpy.fill_diagonal(m, 0.) # manually set diagonal to zero to better see off-diagonal elements
        numpy.fill_diagonal(mkg, 0.) # manually set diagonal to zero to better see off-diagonal elements
        print(m.shape, mkg.shape)
        print(len(rel), rel)
        font = {'size': 15}
        rc('font', **font)
        confusion_m_heat_plot(m, rels=rel, ax=ax)
        #confusion_m_heat_plot(mkg, rels=rel, ax=ax[1])
        fig.tight_layout()
        if input('> Save figure? (y/n)\n') == 'y':
            plt.savefig(input('> Save figure to:\n'), format='pdf')
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
    fig, axes = plt.subplots(1, len(rr), figsize=(16,9), dpi=400)
    ylim = 1
    for i, (a, r) in enumerate(zip(axes.flatten(), rr)):
        violin_plot(*r, ax=a, classes=c, support=supp)
        if i > 0:
            a.yaxis.set_ticklabels([])
        #a.set_title(res[i].replace('results_','').replace('.json',''))
        ylim = min(a.get_ylim()[0], ylim)
    for a in axes.flatten():
        a.set_ylim((ylim,1.))
    fig.tight_layout()
    if input('> Save figure? (y/n)\n') == 'y':
        plt.savefig(input('> Save figure to:\n'))
    plt.show()

