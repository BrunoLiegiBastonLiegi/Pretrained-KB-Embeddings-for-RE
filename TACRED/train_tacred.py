import torch, sys, random, argparse, pickle, time, ngtpy
sys.path.append('../')
from trainer import Trainer
from ner_schemes import BIOES
from dataset import IEData
from pipeline import Pipeline
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from evaluation import ClassificationReport, KG, mean_distance


# Arguments parser
parser = argparse.ArgumentParser(description='Train the model for ADE.')
parser.add_argument('train_data', help='Path to train data file.')
parser.add_argument('test_data', help='Path to test data file.')
parser.add_argument('--load_model', metavar='MODEL', help='Path to pretrained model.')
parser.add_argument('--NED_weight', metavar='NED', help='Weight for NED.')
args = parser.parse_args()

# Disambiguation weight
wNED = 1 if args.NED_weight == None else args.NED_weight

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

model = Pipeline(bert,
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
trainer = Trainer(train_data=train_data,
                  test_data=test_data,
                  model=model,
                  optim=optimizer,
                  device=device,
                  rel2index=rel2index,
                  save=True,
                  wNED=wNED,
                  batchsize=32,
                  tokenizer=tokenizer
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
    
sm1 = torch.nn.Softmax(dim=1)
sm0 = torch.nn.Softmax(dim=0)

ner_groundtruth, ner_prediction = [], []
ned_groundtruth, ned_prediction = [], []
re_groundtruth, re_prediction = [], []

model.eval()
test_loader = DataLoader(test_data,
                         batch_size=256,
                         collate_fn=test_data.collate_fn)

for i, batch in enumerate(test_loader):
    print('Evaluating on the test set. ({} / {})'.format(i, len(test_loader)), end='\r')
    with torch.no_grad():
        inputs = batch['sent']
        if device != torch.device("cpu"):
            inputs = inputs.to(device)
        
            ner_out, ned_out, re_out = model(inputs)
            for i in range(len(inputs['input_ids'])):
                # NER
                ner_groundtruth.append([ bioes.index2tag[int(j)] for j in batch['ner'][i] ])
                ner_prediction.append([ bioes.to_tag(j) for j in sm1(ner_out[i]) ])
                # NED
                ned_groundtruth.append( dict(zip(
                    batch['ned'][i][:,0].int().tolist(),
                    batch['ned'][i][:,1:]))
                )
                #print('>>>>> NED\n', ned_out)
                if ned_out != None:
                    prob = sm1(ned_out[2][i][:,:,0])
                    candidates = ned_out[2][i][:,:,1:]
                    ned_prediction.append(dict(zip(
                        ned_out[0][i].view(-1,).tolist(),
                        torch.vstack([ c[torch.argmax(w)] for w,c in zip(prob, candidates) ])
                    )))
                else:
                    ned_prediction.append(None)
                # RE
                #print('>>>>> RE\n', re_out)
                re_groundtruth.append(dict(zip(
                    zip(
                        batch['re'][i][:,0].tolist(),
                        batch['re'][i][:,1].tolist()
                    ),
                    batch['re'][i][:,2].tolist()
                )))
                if re_out != None:
                    re_prediction.append(dict(zip(
                        zip(
                            re_out[0][i][:,0].tolist(),
                            re_out[0][i][:,1].tolist(),                    
                        ),
                        torch.argmax(sm1(re_out[1][i]), dim=1).view(-1).tolist()
                    )))
                else:
                    re_prediction.append(None)


#print('NER:\n',ner_groundtruth[0], ner_prediction[0])
#print('NED:\n',ned_groundtruth[0], ned_prediction[0])
#print('RE:\n',re_groundtruth[0], re_prediction[0])



cr = ClassificationReport(
    ner_predictions=ner_prediction,
    ner_groundtruth=ner_groundtruth,
    ned_predictions=ned_prediction,
    ned_groundtruth=ned_groundtruth,
    re_predictions=re_prediction,
    re_groundtruth=re_groundtruth,
    re_classes=dict(zip(rel2index.values(),rel2index.keys())),
    ner_scheme='IOBES',
    ned_embeddings=kb
)

f1 = {'NER': cr.ner_report(), 'NED': cr.ned_report(), 'RE': cr.re_report()}
print('NER')
print(f1['NER']['macro avg'])
print(f1['NER']['micro avg'])
print('NED')
print(f1['NED']['macro avg'])
print(f1['NED']['micro avg'])
print('RE')
print(f1['RE']['macro avg'])
print(f1['RE']['micro avg'])
