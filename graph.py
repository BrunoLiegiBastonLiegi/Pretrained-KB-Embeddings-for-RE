import networkx as nx
import matplotlib.pyplot as plt

class KnowledgeGraph(nx.MultiDiGraph):
    """
    Networkx MultiDiGraph wrapper for Knoweldge Graphs.
    """
    def __init__(self, edges=None, **kwargs):
        super(KnowledgeGraph, self).__init__(**kwargs)
        if edges != None:
            self.add_edges(edges)

    def add_edges(self, edges):
        return list(map(self.add_edge, *list(zip(*edges))))

    def add_edge(self, head, tail, rel):
        try:
            redundant = False
            for e in self[head][tail].values():
                if e['type'] == rel:
                    redundant = True
            if not redundant:
                return super(KnowledgeGraph, self).add_edge(head, tail, type=rel)
        except:
            return super(KnowledgeGraph, self).add_edge(head, tail, type=rel)

    def draw(self, **kwargs):
        nx.draw(self, **kwargs)
        plt.show()
