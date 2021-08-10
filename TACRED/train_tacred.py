import torch, sys, random, argparse, pickle, time, ngtpy
sys.path.append('../')
from trainer import Trainer
from ner_schemes import BIOES
from dataset import IEData, Stat
from pipeline import Pipeline, GoldEntities
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from evaluation import ClassificationReport, KG, mean_distance, Evaluator


# Arguments parser
parser = argparse.ArgumentParser(description='Train the model for ADE.')
parser.add_argument('train_data', help='Path to train data file.')
parser.add_argument('test_data', help='Path to test data file.')
parser.add_argument('--load_model', metavar='MODEL', help='Path to pretrained model.')
parser.add_argument('--NED_weight', metavar='NED', help='Weight for NED.')
parser.add_argument('--gold_entities', metavar='GOLDENT', default=False, help='Use gold entities if True.')
args = parser.parse_args()

# Disambiguation weight
wNED = 1 if args.NED_weight == None else args.NED_weight
# Gold entities?
gold = args.gold_entities

kb, data, e_types, r_types = {}, {}, {}, {}

pkl = {}
# Load the data
with open(args.train_data, 'rb') as f:               
    pkl['train'] = pickle.load(f)
with open(args.test_data, 'rb') as f:               
    pkl['test'] = pickle.load(f)
    
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
# Define the pretrained model
#bert = 'bert-base-uncased'
bert = 'bert-base-cased'
#bert = 'bert-large-cased'
#bert = 'EleutherAI/gpt-neo-2.7B'
#bert = "facebook/bart-large-mnli"
#bert = "typeform/distilbert-base-uncased-mnli"
tokenizer = AutoTokenizer.from_pretrained(bert)

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

if gold:
    model = GoldEntities(
        bert,
        ned_dim=list(kb.values())[0].shape[-1],
        KB=kb,
        re_dim=len(r_types)
    )
else:
    model = Pipeline(
        bert,
        ner_dim=bioes.space_dim,
        ner_scheme=bioes,
        ned_dim=list(kb.values())[0].shape[-1],
        KB=kb,
        re_dim=len(r_types)
    )

# check if GPU is avilable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('> Found device:', device, ', setting it as the principal device.')
# move model to device
if device == torch.device("cuda:0"):
    model.to(device)

# define the optimizer
#optimizer = torch.optim.SGD(model.parameters(), lr=3e-5, momentum=0.9)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

# set up the trainer
trainer = Trainer(
    train_data=train_data,
    test_data=test_data,
    model=model,
    optim=optimizer,
    device=device,
    rel2index=rel2index,
    save=True,
    wNED=wNED,
    batchsize=32,
    tokenizer=tokenizer,
    gold_entities=gold
)


# load pretrained model or train
if args.load_model != None:
    model.load_state_dict(torch.load(args.load_model))
else:
    plots = trainer.train(24)
    yn = input('Save loss plots? (y/n)')
    if yn == 'y':
        with open('loss_plots.pkl', 'wb') as f:
            pickle.dump(plots, f)


# ------------------------- Evaluation

ev = Evaluator(
    model=model,
    ner_scheme=bioes,
    kb_embeddings=kb,
    re_classes=dict(zip(rel2index.values(),rel2index.keys())),
    gold_entities=gold
)
ev.classification_report(test_data)


stat = Stat(pkl['train'], pkl['test'])
stat.gen()

f1 = {'NER': cr.ner_report(), 'NED': cr.ned_report(), 'RE': cr.re_report()}
scores = {k: v['f1-score'] for k,v in f1['NER'].items() if k not in ('micro avg', 'macro avg', 'weighted avg')}
stat.score_vs_support(scores)


