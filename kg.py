import networkx as nx
from sklearn.neighbors import NearestNeighbors
import torch, json
import numpy as np
import matplotlib.pyplot as plt

class KG(object):

    def __init__(self, ned, re, KB, relations):
        self.kg = nx.Graph()
        self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto')
        self.KB = KB
        self.relations = relations
        self.ned = ned
        self.re = re
        
    def build(self):
        #print(list(self.KB.values())[0])
        self.nbrs.fit(np.vstack(list(self.KB.values())))
        concepts = list(self.KB.keys())
        for i in range(len(self.ned)):
            if self.re[i] != None:
                ids = {
                    k : concepts[ self.nbrs.kneighbors(p.reshape(1,-1))[1][0][0] ]
                    for k,p in zip(self.ned[i][0], self.ned[i][1])
                }
                for r in self.re[i]:
                    #kg.append({'head': ids[r[0].item()], 'tail': ids[r[1].item()], 'rel': relations[r[2].item()]})
                    if self.relations[r[2].item()] != 'NO_RELATION':
                        self.kg.add_edge(ids[r[0].item()], ids[r[1].item()])
                    
    def draw(self, **kwargs):
        nx.draw(self.kg, **kwargs)
        plt.show()

    def json(self, save_to=None):
        g = nx.readwrite.json_graph.node_link_data(self.kg)
        if save_to != None:
            json.dump(g, save_to, indent=4)
        return g
