import torch, random, json
import numpy as np
import networkx as nx
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

# Performance metrics
from seqeval.metrics import classification_report
from seqeval.scheme import IOBES
from sklearn.metrics import f1_score
import sklearn.metrics as skm


class ClassificationReport(object):

    def __init__(self, ner_predictions, ner_groundtruth, ned_predictions, ned_groundtruth, re_predictions, re_groundtruth, re_classes, ned_embeddings, ner_scheme='IOBES'):
        self.ner_pred = ner_predictions
        self.ner_gt = ner_groundtruth
        self.ned_pred = ned_predictions
        self.ned_gt = ned_groundtruth
        self.re_pred = re_predictions
        self.re_gt = re_groundtruth
        self.re_classes = re_classes
        self.ner_scheme = ner_scheme
        self.ned_embeddings = ned_embeddings
        self.embedding2id = { tuple(v.tolist()): k for k,v in ned_embeddings.items() }
                
    def ner_report(self):
        print('-------------------------- NER SCORES ------------------------------------')
        ner_cr = classification_report(self.ner_gt, ner_pred, mode='strict', scheme=self.ner_scheme)
        print(ner_cr)

    def re_report(self):
        gt = []
        pred = []
        for i in range(len(self.re_gt)):
            if self.re_pred[i] != None:
                d = dict(zip(
                    zip(self.re_gt[i][:,0].tolist(), self.re_gt[i][:,1].tolist()),
                    self.re_gt[i][:,2]
                ))
                for j in self.re_pred[i]:
                    try:
                        gt.append(d.pop(tuple(j[:2].tolist())).item())
                        pred.append(j[2].item())
                    except:
                        pass
                for v in d.values():
                    gt.append(v.item())
                    pred.append(not j[2])
                    #pred.append(-1)
            else:
                for j in self.re_gt[i]:
                    gt.append(j[2].item())
                    pred.append(not j[2])
                    #pred.append(-1)
        
                        
        print(skm.confusion_matrix(np.array(gt), np.array(pred), labels=[0,1]))
        return skm.classification_report(np.array(gt), np.array(pred), labels=[0,1], target_names=['NO_RELATION', 'ADVERSE_EFFECT_OF'])
        #return skm.classification_report(np.array(gt), np.array(pred), labels=[-1,0,1], target_names=['WRONG_ENTITIES','NO_RELATION', 'ADVERSE_EFFECT_OF'])

    def ned_report(self):
        gt = []
        pred = []
        mean = 0.
        for i in range(len(self.ned_gt)):
            for j in range(len(self.ned_gt[i][0])):
                try:
                    ind = self.ned_pred[i][0].index(self.ned_gt[i][0][j])
                    pred.append(self.ned_pred[i][1][ind])
                    gt.append(self.ned_gt[i][1][j])
                    mean += distance.euclidean(self.ned_pred[i][1][ind], self.ned_gt[i][1][j])
                except:
                    pred.append('***ERR***')
                    gt.append(self.ned_gt[i][1][j])
        self.ned_gt = gt
        self.ned_pred = pred
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto')
        nbrs.fit(torch.vstack(list(self.ned_embeddings.values())))
        concepts = list(self.ned_embeddings.keys())
        embs =  list(self.ned_embeddings.values())
        for i in range(len(gt)):
            #for l, e in enumerate(embs):
             #   if torch.sum(gt[i]!=e) == 0:
              #      j = l
            gt[i] = self.embedding2id[tuple(gt[i].tolist())]
            #gt[i] = concepts[j]
            if pred[i] != '***ERR***':
                _, k = nbrs.kneighbors(pred[i].view(1,-1))
                pred[i] = concepts[k[0][0]]
            else:
                pred[i] = random.choice(concepts)

        print('> Mean distance between predictions and groundtruth for NED:', mean / len(gt))
        labels = { v: k for k, v in enumerate(gt)}
        labels = list(labels.keys())
        return skm.classification_report(gt, pred, labels=labels)

        
# Plot graph embedding
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_embedding(predicted, groundtruth, method='PCA'):
    if method == 'PCA':
        pca = PCA(n_components=2)
        pca.fit(predicted)
        pca_pred = pca.transform(predicted)
        pca.fit(groundtruth)
        pca_gt = pca.transform(groundtruth)
        fig, axs = plt.subplots(2)
        axs[0].scatter(x=pca_pred[:,0], y=pca_pred[:,1])
        axs[0].scatter(x=pca_gt[:,0], y=pca_gt[:,1])
        plt.show()
    elif method == 'TSNE':
        pass


# Evaluate distances in embedding

def mean_neighbors_distance(embeddings, n_neighbors=10):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
    nbrs.fit(embeddings)
    mean = 0.
    for e in embeddings:
        d, _ = nbrs.kneighbors(e.reshape(1,-1))
        mean += np.mean(d[0][1:])
    return mean / len(embeddings)

def mean_distance(embeddings):
    mean = 0.
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            if i != j:
                mean += distance.euclidean(embeddings[i], embeddings[j])
    return mean / (len(embeddings)**2 - len(embeddings))



# KG
def KG(ned_predictions, re_predictions, embeddings, relations, save=None, remove_disconnected_components=False):
    #embedding2id = { tuple(v.tolist()): k for k,v in ned_embeddings.items() }
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nbrs.fit(torch.vstack(list(embeddings.values())))
    concepts = list(embeddings.keys())
    kg = nx.Graph()
    for i in range(len(ned_predictions)):
        if re_predictions[i] != None:
            ids = {
                k : concepts[ nbrs.kneighbors(p.reshape(1,-1))[1][0][0] ]
                for k,p in zip(ned_predictions[i][0], ned_predictions[i][1])
            }
            for r in re_predictions[i]:
                #kg.append({'head': ids[r[0].item()], 'tail': ids[r[1].item()], 'rel': relations[r[2].item()]})
                if relations[r[2].item()] != 'NO_RELATION':
                    kg.add_edge(ids[r[0].item()], ids[r[1].item()])
    if remove_disconnected_components:
        pass # to implement
    if save != None:
        data = nx.readwrite.json_graph.node_link_data(kg)
        json.dump(data, save, indent=4)
    nx.draw(kg, with_labels=True)
    plt.show()
    return kg
