import json, numpy, torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.spatial import distance_matrix
import seaborn as sns
from sklearn.cluster import SpectralCoclustering


def plot_embedding(embeddings, colors='blue', method='TSNE'):
    """
    Plot the projection in the 2d space of the graph embeddings.
    """
    assert method in {'PCA', 'TSNE'}
    if method == 'PCA':
        pca = PCA(n_components=2)
        pca.fit(embeddings)
        comp = pca.transform(embeddings)
        fig, axs = plt.subplots()
        axs.scatter(x=comp[:,0], y=comp[:,1])
        plt.show()
    elif method == 'TSNE':
        proj = TSNE(n_components=2).fit_transform(embeddings)
        fig, axs = plt.subplots()
        axs.scatter(x=proj[:,0], y=proj[:,1], c=colors, cmap='Accent')
        plt.show()


def collect_scores(res_file):
    """
    Collect the final scores for each class for each experiment from the results file.
    """
    with open(res_file, 'r') as f:
        res = json.load(f)
    f1 = {}
    for v in res.values():
        for k,c in v['scores'].items(): # [0] because in principle we could have the results also for the NER and
            if k == 'accuracy':            # NED task, however at the moment we are just considering RE
                try:
                    f1['micro avg'].append(c)
                except:
                    f1['micro avg'] = [c]
            elif k != 'weighted avg':
                try:
                    f1[k].append(c['f1-score'])
                except:
                    f1[k] = [c['f1-score']]
    return f1

def collect_confusion_m(res_file: str) -> list:
    """
    Collect the confusion matrices for each experiment from the results file.
    """
    with open(res_file, 'r') as f:
        res = json.load(f)
    #for v in res.values():
    #    print(len(v['scores']), list(v['scores'].keys()))
    #    print(numpy.array(v['confusion matrix']).shape)
    return [ v['confusion matrix'] for v in res.values() ]

def collect_PR_curve(res_file: str) -> list:
    with open(res_file, 'r') as f:
        res = json.load(f)
    #for run in res.values():
    #    for metric in ('precision', 'recall'):
    #        for k,v in run['pr_curve'][metric].items():
    #            run['pr_curve'][metric][k] = numpy.array(v)
    return [ r['pr_curve'] for r in res.values() ]

def violin_plot(*results, legend=['no graph embeddings', 'graph embeddings'], classes=[], support=None, **kwargs):
    """
    Prepare the violin plot for comparing *results. 
    By default we want to compare results with and without 
    graph embeddings enabled.
    """
    try:
        ax = kwargs['ax']
    except:
        fig, ax = plt.subplots()
    assert len(legend) == len(results)
    results = list(results)
    if len(classes) > 0:
        for i,r in enumerate(results):
            results[i] = { c: r[c] for c in classes }
    if support != None:
        support['micro avg'] = -1
        support['macro avg'] = -2
        support = dict(sorted(support.items(), key=lambda x: x[1], reverse=True))
        results = [ { k: r[k] for k in support.keys() } for i,r in enumerate(results) ]
    i = 0
    for r,l in zip(results, legend):
        col, score = list(zip(*r.items()))
        ax.scatter(range(1,len(col)+1), numpy.array(score).mean(-1), label=l)
        if support != None and i < 1:
            for s,x,y in zip(list(support.values())[:-2], range(1,len(col)-1), numpy.array(score[:-2]).mean(-1)): # [-2:] to avoid annotation of micro/macro avg, they don't have a support value!
                ax.annotate(s, xy=(x, y), xytext=(x+0.1,y))
            i += 1
        ax.violinplot(score)
        ax.set_xticks(range(1,len(col)+1))
        ax.set_xticklabels([ c.split('/')[-1] for c in col], rotation=45)
    #ax.legend()
    

def confusion_m_heat_plot(m, rels, **kwargs):
    """
    ax.imshow(m, **kwargs)
    
    # We want to show all ticks...
    ax.set_xticks(numpy.arange(len(rels)))
    ax.set_yticks(numpy.arange(len(rels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(rels)
    ax.set_yticklabels(rels)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(rels)):
        for j in range(len(rels)):
            text = ax.text(j, i, "{:.1f}".format(m[i, j]),
                           ha="center", va="center", color="w")
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=m, display_labels=rels)
    disp.plot(ax=kwargs['ax'])
    kwargs['ax'].tick_params(axis='x', labelrotation = 45)    
    

def rel_embedding_plot(triplets: list, head_tail_diff: bool = False, proj=TSNE(n_components=2), **kwargs):
    try:
        axs = kwargs['ax']
    except:
        fig, axs = plt.subplots(1,2)
        
    rel_emb = {}
    head, tail = {}, {}
    X = []
    for t in triplets:
        X.append(t[0])
        X.append(t[1])
        try:
            if head_tail_diff:
                rel_emb[t[2]].append(t[1]-t[0])
            else:
                rel_emb[t[2]].append(t[0])
                rel_emb[t[2]].append(t[1])
                head[t[2]].append(t[0])
                tail[t[2]].append(t[1])
        except:
            if head_tail_diff:
                rel_emb[t[2]] = [t[1]-t[0]]
            else:
                rel_emb[t[2]] = [t[0], t[1]]
                head[t[2]] = [t[0]]
                tail[t[2]] = [t[1]]

    proj.fit(torch.vstack(X))
    colors = ['r' for i in range(len(head['Work_For']))]
    colors += ['b' for i in range(len(tail['Work_For']))]
    ht = proj.fit_transform(torch.vstack(head['Work_For'] + tail['Work_For']))
    plt.scatter(ht[:,0], ht[:,1], c=colors)
    plt.show()
    
    
    mean_emb = torch.vstack( 
        list(map( lambda x: torch.vstack(x).mean(0),
             rel_emb.values() ))
    )
    #g = sns.clustermap(mean_emb.numpy(), col_cluster=False)
    #plt.show()
    p = proj.fit_transform(mean_emb)
    axs[0].scatter(x=p[:,0], y=p[:,1], c=range(len(rel_emb)), cmap='Accent')
    for (x,y),t in zip(p, rel_emb.keys()):
        axs[0].annotate(t, xy=(x, y), xytext=(x+0.1,y))

    dist = distance_matrix( mean_emb, mean_emb )
    cluster_m = SpectralCoclustering(n_clusters=2, random_state=0)
    cluster_m.fit(dist)
    fit_data = dist[numpy.argsort(cluster_m.row_labels_)]
    fit_data = fit_data[:, numpy.argsort(cluster_m.column_labels_)]
    #g = sns.clustermap(dist)
    #plt.show()
    im = axs[1].imshow(dist)
    #im = axs[1].imshow(fit_data)
    cbar = axs[1].figure.colorbar(im, ax=axs[1])
    axs[1].set_xticks(numpy.arange(dist.shape[0]))
    axs[1].set_yticks(numpy.arange(dist.shape[1]))
    axs[1].set_xticklabels(rel_emb.keys())
    axs[1].set_yticklabels(rel_emb.keys())
    plt.setp(axs[1].get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    #for i in range(dist.shape[0]):
    #    for j in range(dist.shape[1]):
    #        text = axs[1].text(j, i, "{:.5f}".format(dist[i, j]),
    #                       ha="center", va="center", color="w")
    
    return dist

def PR_curve_plot(*curves, **kwargs):
    mean, low, high = {}, {}, {}
    for metric in ('precision', 'recall'):
        for k in curves[0][metric].keys():
            tmp = numpy.vstack([ c[metric][k] for c in curves ])
            mean[metric][k] = tmp.mean(0)
            low[metric][k] = tmp.min(0)
            high[metric][k] = tmp.max(0)
    for c in (mean, low, high):
        display = PrecisionRecallDisplay(
            recall=c['recall']["micro"],
            precision=c['precision']["micro"],
        )
        display.plot()
    plt.show()
    try:
        ax = kwargs['ax']
    except:
        fig, ax = plt.subplots()
    ax.plot(mean['recall']['micro'], mean['precision']['micro'])
    ax.fill_between(mean['recall']['micro'], low['precision']['micro'], high['precision']['micro'])
    plt.show()
