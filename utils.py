import json, numpy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
from sklearn.manifold import TSNE


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

def violin_plot(*results, ax=plt.subplots()[1], legend=['no graph embeddings', 'graph embeddings'], classes=[], support=None):
    """
    Prepare the violin plot for comparing *results. 
    By default we want to compare results with and without 
    graph embeddings enabled.
    """
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
        ax.set_xticklabels(col, rotation=45)
    ax.legend()
    

def confusion_m_heat_plot(m, rels, ax=plt.subplots()[1], **kwargs):
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
            text = ax.text(j, i, m[i, j],
                           ha="center", va="center", color="w")
    
