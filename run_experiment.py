import torch, argparse, pickle, re, json
from trainer import Trainer
from ner_schemes import BIOES
from dataset import IEData, Stat
from pipeline import Pipeline, GoldEntities
from model import BaseIEModel, BaseIEModelGoldEntities, IEModel, IEModelGoldEntities, IEModelGoldKG
from transformers import AutoTokenizer
from evaluation import Evaluator

# Arguments parser
parser = argparse.ArgumentParser(description='Train a model and evaluate on a dataset.')
parser.add_argument('train_data', help='Path to train data file.')
parser.add_argument('test_data', help='Path to test data file.')
parser.add_argument('--load_model', metavar='MODEL', help='Path to pretrained model.')
args = parser.parse_args()

# Input/Output directory
dir = re.search('.+<?\/', args.train_data).group(0)
assert dir == re.search('.+<?\/', args.test_data).group(0)

pkl = {}
# Load the data
with open(args.train_data, 'rb') as f:               
    pkl['train'] = pickle.load(f)
with open(args.test_data, 'rb') as f:               
    pkl['test'] = pickle.load(f)

# Do some statistics
stat = Stat(pkl['train'], pkl['test'])
stat.gen()

# Organize usable data in a suitable way
kb, data, e_types, r_types = {}, {}, {}, {}
discarded_sents = []
for s, d in pkl.items():
    data[s] = {
        'sent': [],
        'ents': [],
        'rels': []
    }
    for v in d:
        discard = False
        for e in v['entities'].values():
            try:
                emb_flag = e['embedding'].any() != None
            except:
                emb_flag = False
            if e['type'] != None and emb_flag:
                kb[e['id']] = torch.tensor(e['embedding'], dtype=torch.float32).view(1, -1)
                e['embedding'] = torch.tensor(e['embedding'], dtype=torch.float32).view(1, -1)
                e_types[e['type']] = 0
            else:
               discard = True 
        for r in v['relations'].values():
            r_types[r] = 0
        if discard:
            discarded_sents.append((v['sentence'], v['entities'], v['relations']))
        else:
            data[s]['sent'].append(v['sentence'][0])
            data[s]['ents'].append(v['entities'])
            data[s]['rels'].append(v['relations'])
print('> Discarded {} sentences, due to incomplete annotations.'.format(len(discarded_sents)))


# Define the tagging scheme
bioes = BIOES(list(e_types.keys()))
# Define the relation scheme
rel2index = dict(zip(r_types.keys(), range(len(r_types))))
print(rel2index)
# Define the pretrained model
#bert = 'bert-base-uncased'
bert = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(bert)


# Prepare data for training
train_data = IEData(
    sentences=data['train']['sent'],
    ner_labels=data['train']['ents'],
    re_labels=data['train']['rels'],
    preprocess=True,
    tokenizer=tokenizer,
    ner_scheme=bioes,
    rel2index=rel2index#,
    #save_to=args.train_data.replace('.pkl', '_preprocessed.pkl')
)
    
test_data = IEData(
    sentences=data['test']['sent'],
    ner_labels=data['test']['ents'],
    re_labels=data['test']['rels'],
    preprocess=True,
    tokenizer=tokenizer,
    ner_scheme=bioes,
    rel2index=rel2index#,
    #save_to=args.test_data.replace('.pkl', '_preprocessed.pkl')
)

# check if GPU is avilable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('> Found device:', device, ', setting it as the principal device.')

"""
model = BaseIEModel(
    language_model = bert,
    ner_dim = bioes.space_dim,
    ner_scheme = bioes,
    re_dim = len(r_types),
    device = device
)
"""

model = BaseIEModelGoldEntities(
    language_model = bert,
    re_dim = len(r_types),
    device = device
)

"""
model = IEModel(
    language_model = bert,
    ner_dim = bioes.space_dim,
    ner_scheme = bioes,
    ned_dim = list(kb.values())[0].shape[-1],
    KB = kb,
    re_dim = len(r_types),
    device = device
)
"""
"""
model = IEModelGoldEntities(
    language_model = bert,
    ned_dim = list(kb.values())[0].shape[-1],
    KB = kb,
    re_dim = len(r_types),
    device = device
)
"""
"""
model = IEModelGoldKG(
    language_model = bert,
    ned_dim = list(kb.values())[0].shape[-1],
    re_dim = len(r_types),
    device = device
)
"""
# move model to device
#if device == torch.device("cuda:0"):
#    model.to(device)

# define the optimizer
lr = 2e-5
#optimizer = torch.optim.SGD(model.parameters(), lr=3e-5, momentum=0.9)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# set up the trainer
batchsize = 8
trainer = Trainer(
    train_data=train_data,
    test_data=test_data,
    model=model,
    optim=optimizer,
    device=device,
    rel2index=rel2index,
    save=False,
    batchsize=batchsize,
    tokenizer=tokenizer,
)

n_epochs = 12
# load pretrained model or train
if args.load_model != None:
    model.load_state_dict(torch.load(args.load_model))
else:
    plots = trainer.train(n_epochs)
    #yn = input('Save loss plots? (y/n)')
    yn = 'n'
    if yn == 'y':
        with open(dir + '/loss_plots.pkl', 'wb') as f:
            pickle.dump(plots, f)

# Evaluation
results = {}
ev = Evaluator(
    model=model,
    ner_scheme=bioes,
    kb_embeddings=kb,
    re_classes=dict(zip(rel2index.values(),rel2index.keys())),
)

results = {
    'model': re.search('model\.(.+?)\'\>', str(type(model))).group(1),
    'learning_rate': lr,
    'epochs': n_epochs,
    'batchsize': batchsize,
    'scores': ev.classification_report(test_data)
}
                                       
with open(dir + '/results_kg.json', 'a') as f:
    json.dump(results, f, indent=4)


