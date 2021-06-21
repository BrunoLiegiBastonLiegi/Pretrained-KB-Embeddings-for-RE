import torch, sys, random, argparse, pickle, time
sys.path.append('../')
from trainer import Trainer
from ner_schemes import BIOES
from dataset import IEData
from pipeline import Pipeline
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# Arguments parser
parser = argparse.ArgumentParser(description='Train the model for ADE.')
parser.add_argument('train_data', help='Path to train data file.')
parser.add_argument('test_data', help='Path to test data file.')
parser.add_argument('--load_model', metavar='MODEL', help='Path to pretrained model.')
parser.add_argument('--NED_weight', metavar='NED', help='Weight for NED.')
args = parser.parse_args()

# Disambiguation weight
wNED = 1 if args.NED_weight == None else args.NED_weight

pkl = {}
# Load the data
with open(args.train_data, 'rb') as f:               
    pkl['train'] = pickle.load(f)
with open(args.test_data, 'rb') as f:               
    pkl['test'] = pickle.load(f)

discarded_sents = []
kb, data, e_types, r_types = {}, {}, {}, {}
for s, d in pkl.items():
    data[s] = {
        'sent': [],
        'ents': [],
        'rels': []
    }
    for i, v in d.items():
        discard = False
        for e in v['entities'].values():
            #print(v)
            try:
                emb_flag = e['embedding'].any() != None
            except:
                emb_flag = False
            #print(emb_flag)
            if e['type'] != None and emb_flag:
                kb[e['id']] = torch.tensor(e['embedding'], dtype=torch.float32).view(1, -1)
                e['embedding'] = torch.tensor(e['embedding'], dtype=torch.float32).view(1, -1)
                e_types[e['type']] = 0
            else:
               discard = True 
        for r in v['relations'].values():
            r_types[r] = 0
        if discard:
            discarded_sents.append((i, v['entities'], v['relations']))
        else:
            data[s]['sent'].append(i)
            data[s]['ents'].append(v['entities'])
            data[s]['rels'].append(v['relations'])
print('> Discarded {} sentences, due to incomplete annotations.'.format(len(discarded_sents)))

# Define the tagging scheme
bioes = BIOES(list(e_types.keys()))
# Define the relation scheme
rel2index = dict(zip(r_types.keys(), range(len(r_types))))
# Define the pretrained model
bert = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(bert)

train_data = IEData(
    sentences=data['train']['sent'],
    ner_labels=data['train']['ents'],
    re_labels=data['train']['rels'],
    tokenizer=tokenizer,
    ner_scheme=bioes,
    rel2index=rel2index
)

test_data = IEData(
    sentences=data['test']['sent'],
    ner_labels=data['test']['ents'],
    re_labels=data['test']['rels'],
    tokenizer=tokenizer,
    ner_scheme=bioes,
    rel2index=rel2index
)

model = Pipeline(bert,
                 ner_dim=bioes.space_dim,
                 ner_scheme=bioes,
                 ned_dim=list(kb.values())[0].shape[-1],
                 KB=kb,
                 re_dim=len(r_types))



# check if GPU is avilable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('> Found device:', device, ', setting it as the principal device.')
# move model to device
if device == torch.device("cuda:0"):
    model.to(device)

# define the optimizer
#optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5)

# set up the trainer
trainer = Trainer(train_data=train_data,
                  test_data=test_data,
                  model=model,
                  optim=optimizer,
                  device=device,
                  save=True,
                  wNED=wNED,
                  batchsize=32
)

plots = trainer.train(6)
