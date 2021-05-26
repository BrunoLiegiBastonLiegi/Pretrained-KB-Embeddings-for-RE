import torch, sys, random, argparse, pickle
sys.path.append('../')
from trainer import Trainer
from ner_schemes import BIOES
from dataset import IEData
from pipeline import Pipeline
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


# Arguments parser
parser = argparse.ArgumentParser(description='Train the model for ADE.')
parser.add_argument('input_data', help='Path to input data file.')
parser.add_argument('--load_model', metavar='MODEL', help='Path to pretrained model.')
parser.add_argument('--fold', metavar='FOLD', help='Number of the fold for k-fold crossvalidation')
parser.add_argument('--NED_weight', metavar='NED', help='Weight for NED.')
args = parser.parse_args()

# Disambiguation weight
wNED = 1 if args.NED_weight == None else args.NED_weight

# Define the tagging scheme
bioes = BIOES(['AE','DRUG'])

# Load the data
with open(args.input_data, 'rb') as f:               
    pkl = pickle.load(f)
if args.fold != None:
    pkl = pkl['fold_' + str(args.fold)]
else:
    pkl = pkl['fold_0']

kb = {}
data = {}
for s, d in pkl.items():
    data[s] = {
        'sent': [],
        'ents': [],
        'rels': []
    }
    for i in d:
        for v in i['entities'].values():
            kb['-'.join(v['concept'])] = torch.mean(v['embedding'], dim=0)
        data[s]['sent'].append(i['sentence']['sentence'])
        data[s]['ents'].append(i['entities'])
        data[s]['rels'].append(i['relations'])

bert = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(bert)
model = Pipeline(bert, ner_dim=bioes.space_dim, ner_scheme=bioes, ned_dim=50, KB=kb, re_dim=2)
rel2index = {'NO_RELATION': 0, 'ADVERSE_EFFECT_OF': 1}

train_data = IEData(
    sentences=data['train']['sent'],
    ner_labels=data['train']['ents'],
    re_labels=data['train']['rels'],
    tokenizer=tokenizer,
    ner_scheme=bioes,
    rel2index=rel2index#,
    #save_to='ADE.pkl'
)

test_data = IEData(
    sentences=data['test']['sent'],
    ner_labels=data['test']['ents'],
    re_labels=data['test']['rels'],
    tokenizer=tokenizer,
    ner_scheme=bioes,
    rel2index=rel2index#,
    #save_to='ADE.pkl'
)

# check if GPU is avilable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('> Found device:', device, ', setting it as the principal device.')
# move model to device
if device == torch.device("cuda:0"):
    model.to(device)

# define the optimizer
#optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00002)

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

# load pretrained model or train
if args.load_model != None:
    model.load_state_dict(torch.load(args.load_model))
else:
    plots = trainer.train(6)

    

# ------------------------- Evaluation 
    
sm1 = torch.nn.Softmax(dim=1)
sm0 = torch.nn.Softmax(dim=0)

ner_groundtruth, ner_prediction = [], []
ned_groundtruth, ned_prediction = [], []
re_groundtruth, re_prediction = [], []

for i in range(len(test_data)):
    with torch.no_grad():
        inputs = test_data[i]['sent']
        if device != torch.device("cpu"):
            inputs = inputs.to(device)

        ner_out, ned_out, re_out = model(inputs)
        # NER
        ner_groundtruth.append([ bioes.index2tag[int(j)] for j in test_data[i]['ner'].view(-1) ])
        ner_prediction.append([ bioes.to_tag(j) for j in sm1(ner_out.squeeze(0)) ])
        # NED
        ned_groundtruth.append( dict(zip(
            test_data[i]['ned'][:,0].int().tolist(),
            test_data[i]['ned'][:,1:]))
            )
        if ned_out != None:
            prob = sm1(ned_out[2].squeeze(0)[:,:,0])
            candidates = ned_out[2].squeeze(0)[:,:,1:]
            ned_prediction.append(dict(zip(
                ned_out[0].squeeze(0).view(-1,).tolist(),
                torch.vstack([c[torch.argmax(w)] for w,c in zip(prob, candidates)])
            )))
        else:
            ned_prediction.append(None)
        # RE
        re_groundtruth.append(test_data[i]['re'])
        if re_out != None:
            re_prediction.append(torch.hstack((
                re_out[0].squeeze(0),
                torch.argmax(sm1(re_out[1].squeeze(0)), dim=1).view(-1,1)
            )))
        else:
            re_prediction.append(None)
        
                
from evaluation import ClassificationReport, KG

cr = ClassificationReport(
    ner_predictions=ner_prediction,
    ner_groundtruth=ner_groundtruth,
    ned_predictions=ned_prediction,
    ned_groundtruth=ned_groundtruth,
    re_predictions=re_prediction,
    re_groundtruth=re_groundtruth,
    re_classes=rel2index,
    ner_scheme='IOBES',
    ned_embeddings=kb
)

f1 = {'NER': cr.ner_report(), 'NED': cr.ned_report(), 'RE': cr.re_report()}
print('NER')
print(f1['NER'])
#print(f1['NER']['micro avg'])
print('NED')
print(f1['NED']['macro avg'])
print(f1['NED']['weighted avg'])
print('RE')
print(f1['RE']['macro avg'])
print(f1['RE']['micro avg'])
