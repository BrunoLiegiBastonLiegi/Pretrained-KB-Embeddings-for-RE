from utils import rel_embedding_plot
from dataset import Stat
import pickle, sys
import matplotlib.pyplot as plt

train_f = sys.argv[1]#'TACRED/train_dev_no-relation_cut.pkl'#'Wikidata/wikidata_train.pkl'#'CONLL04/conll04_train_dev_without_no-rel.pkl'
test_f = sys.argv[2]#'TACRED/test_no-relation_cut.pkl'#'Wikidata/wikidata_val.pkl'#'CONLL04/conll04_test_without_no-rel.pkl'
with open(train_f, 'rb') as f:
    train = pickle.load(f)
with open(test_f, 'rb') as f:
    test = pickle.load(f)

s = Stat(train,test)
s.scan()
#s.filter_rels(5)

triplets = []
for e in s.edges:
    try:
        triplets.append((s.kb[e[0]], s.kb[e[1]], e[2]))
    except:
        pass

#print(triplets[0])
rel_embedding_plot(triplets, head_tail_diff=True)
plt.show()
