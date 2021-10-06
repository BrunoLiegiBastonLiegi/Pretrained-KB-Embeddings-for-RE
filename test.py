from dataset import Stat
from utils import collect_results, violin_plot
import pickle
import matplotlib.pyplot as plt

with open('TACRED/test_no-relation_cut.pkl', 'rb') as f:
    test = pickle.load(f)

with open('TACRED/train_dev_no-relation_cut.pkl', 'rb') as f:
    train = pickle.load(f)

print(len(train), len(test))

st = Stat(train, test)
st.scan()
rels, train, test = st.filter_rels(10)

print(len(train), len(test))

res = [ collect_results(i) for i in ('TACRED/results_top_10_rels.json', 'TACRED/results_kg_top_10_rels.json') ]
violin_plot(*res, support=rels)
plt.show()
