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

from torch.utils.data import DataLoader
#from bidict import bidict


class Evaluator(object):
    
    def __init__(self, model, ner_scheme, kb_embeddings, re_classes, batchsize=64):
        self.model = model
        self.model.eval()
        self.ner_scheme = ner_scheme
        self.batchsize = batchsize
        self.embedding2id = { tuple(v.flatten().tolist()): k for k,v in kb_embeddings.items() }
        #self.embedding2id = bidict(kb_embeddings)
        self.re_classes = re_classes
        try:
            model.NER
            self.NER = True
        except:
            self.NER = False
        try:
            model.NED
            self.NED = True
        except:
            self.NED = False
        
    def eval(self, data):
        self.ner_groundtruth, self.ner_prediction = [], []
        self.ned_groundtruth, self.ned_prediction = [], []
        self.re_groundtruth, self.re_prediction = [], []
        
        sm1 = torch.nn.Softmax(dim=1)
        sm0 = torch.nn.Softmax(dim=0)

        self.model.eval()
        test_loader = DataLoader(data,
                                 batch_size=self.batchsize,
                                 collate_fn=data.collate_fn)

        for i, batch in enumerate(test_loader):
            print('Evaluating on the test set. ({} / {})'.format(i, len(test_loader)), end='\r')
            with torch.no_grad():
                inputs, targets = self.model.prepare_inputs_targets(batch)
                #inputs = batch['sent'].to(next(self.model.parameters()).device)
                #entities = batch['pos']
                outs = self.model(*inputs)
                #if self.gold:
                #    ned_out, re_out = self.model(inputs, entities)
                #    ner_out = None
                #else:
                #    ner_out, ned_out, re_out = self.model(inputs)
                #for i in range(len(inputs['input_ids'])):
                for i in range(batch['ner'].shape[0]):
                    # NER
                    #if ner_out != None:
                    if self.NER:
                        self.ner_groundtruth.append([ self.ner_scheme.index2tag[int(j)] for j in batch['ner'][i] ])
                        self.ner_prediction.append([ self.ner_scheme.to_tag(j) for j in sm1(outs[0][i]) ])
                    # NED
                    if self.NED:
                        self.ned_groundtruth.append( dict(zip(
                            batch['ned'][i][:,0].int().tolist(),
                            batch['ned'][i][:,1:]))
                        )
                        idx = 1 if self.NER else 0
                        if outs[idx] != None:
                            prob = sm1(outs[idx][2][i][:,:,0])
                            candidates = outs[idx][2][i][:,:,1:]
                            self.ned_prediction.append(dict(zip(
                                outs[idx][0][i].view(-1,).tolist(),
                                torch.vstack([ c[torch.argmax(w)] for w,c in zip(prob, candidates) ])
                            )))
                        else:
                            self.ned_prediction.append(None)
                    # RE
                    self.re_groundtruth.append(dict(zip(
                        zip(
                            batch['re'][i][:,0].tolist(),
                            batch['re'][i][:,1].tolist()
                        ),
                        batch['re'][i][:,2].tolist()
                    )))
                    if outs[-1] != None:
                        self.re_prediction.append(dict(zip(
                            zip(
                                outs[-1][0][i][:,0].tolist(),
                                outs[-1][0][i][:,1].tolist(),                    
                            ),
                            torch.argmax(sm1(outs[-1][1][i]), dim=1).view(-1).tolist()
                        )))
                    else:
                        self.re_prediction.append(None)
                        
    def ner_report(self):
        print(classification_report(self.ner_groundtruth, self.ner_prediction, mode='strict', scheme=IOBES))
        return classification_report(self.ner_groundtruth, self.ner_prediction, mode='strict', scheme=IOBES, output_dict=True)

    def ned_report(self):
        target, pred = [], []
        classes = {}
        for gt, p in zip(self.ned_groundtruth, self.ned_prediction):
            for k,v in gt.items():
                target.append(self.embedding2id[tuple(v.tolist())])
                classes[self.embedding2id[tuple(v.tolist())]] = 0
                try:
                    tmp = p[k]
                    pred.append(self.embedding2id[tuple(tmp.tolist())])
                    #pred.append(self.embedding2id.inverse[tmp])

                except:
                    pred.append('***ERR***')
                #target.append(self.embedding2id.inverse[v.view(1,-1)])
                #classes[self.embedding2id.inverse[v.view(1,-1)]] = 0
        for i in range(len(pred)):
            try:
                classes[pred[i]]
            except:
                pred[i] = random.choice(list(classes.keys()))
        print(skm.classification_report(target, pred, labels=list(classes.keys())))
        return skm.classification_report(target, pred, labels=list(classes.keys()), output_dict=True)

    def re_report(self):
        target, pred= [], []
        classes = {}
        for gt, p in zip(self.re_groundtruth, self.re_prediction):
            for k,v in gt.items():
                target.append(self.re_classes[gt[k]])
                classes[self.re_classes[gt[k]]] = 0
                try:
                    pred.append(self.re_classes[p[k]])
                except:
                    pred.append('***ERR***')
        print(skm.classification_report(target, pred, labels=list(classes.keys())))
        return skm.classification_report(target, pred, labels=list(classes.keys()), output_dict=True)

    def classification_report(self, data):
        self.eval(data)
        cr = []
        if self.NER:
            cr.append(self.ner_report())
        if self.NED:
            cr.append(self.ned_report())
        cr.append(self.re_report())
        return cr


    

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
        self.embedding2id = { tuple(v.flatten().tolist()): k for k,v in ned_embeddings.items() }
                
    def ner_report(self):
        if self.ner_scheme == 'IOBES':
            print(classification_report(self.ner_gt, self.ner_pred, mode='strict', scheme=IOBES))
            return classification_report(self.ner_gt, self.ner_pred, mode='strict', scheme=IOBES, output_dict=True)
        else:
            print('> NER Scheme not supported at the moment.')

    def re_report(self):
        target, pred= [], []
        classes = {}
        for gt, p in zip(self.re_gt, self.re_pred):
            for k,v in gt.items():
                target.append(self.re_classes[gt[k]])
                classes[self.re_classes[gt[k]]] = 0
                try:
                    pred.append(self.re_classes[p[k]])
                    #target.append(self.re_classes[gt[k]])
                    #classes[self.re_classes[gt[k]]] = 0
                except:
                    pred.append('***ERR***')
                    #pass
        print(skm.classification_report(target, pred, labels=list(classes.keys())))
        return skm.classification_report(target, pred, labels=list(classes.keys()), output_dict=True)

    def ned_report(self):
        target, pred = [], []
        classes = {}
        for gt, p in zip(self.ned_gt, self.ned_pred):
            for k,v in gt.items():
                try:
                    tmp = p[k]
                    pred.append(self.embedding2id[tuple(tmp.tolist())])
                    #target.append(self.embedding2id[tuple(v.tolist())])
                    #classes[self.embedding2id[tuple(v.tolist())]] = 0
                except:
                    pred.append('***ERR***')
                    #pass
                target.append(self.embedding2id[tuple(v.tolist())])
                classes[self.embedding2id[tuple(v.tolist())]] = 0
        print(skm.classification_report(target, pred, labels=list(classes.keys())))
        return skm.classification_report(target, pred, labels=list(classes.keys()), output_dict=True)

        
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
