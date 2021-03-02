import re, torch, sys, random, argparse, pickle
sys.path.append('../')
from pipeline import Pipeline
from transformers import AutoTokenizer
from utils import Trainer, BIOES
import numpy as np


# Arguments parser
parser = argparse.ArgumentParser(description='Train the model.')
parser.add_argument('input_data', help='Path to input data file.')
parser.add_argument('--load_model', metavar='MODEL', help='Path to pretrained model.')
args = parser.parse_args()

# Define the tagging scheme
bioes = BIOES(['AE','DRUG'])

# Load the data
with open(args.input_data, 'rb') as f:               
    pkl = pickle.load(f)
pkl = pkl['fold_0'] # get only the 0-fold for testing

rel2index = {'NO_RELATION': 0, 'ADVERSE_EFFECT_OF': 1, 'HAS_ADVERSE_EFFECT': 2} # consider adding a thrid relation HAS_AE
                                                                                # i.e. AE_OF^-1
                                                 
# Prepare train and test set with labels
data = {}
for s, d in pkl.items():
    data[s] = []
    for i in d:
        # sentence (str)
        sent = i['sentence']['sentence']
        # tags for the sentence ['O', 'B-DRUG', 'E-DRUG', 'O', 'O', ...]
        tagged_sent = bioes.to_tensor(*i['sentence']['tag'], index=True)
        # entities
        ents = []
        # pretrained graph embeddings of the disambiguated entities
        # for simplicity we take only the last token of each entity as its identifier
        embs = torch.vstack([
            torch.hstack((
                torch.tensor(v['tokenized'][-1]),    # last token of the entity
                torch.mean(v['embedding'], dim=0)))  # mean graph embedding
            for v in i['entities'].values() ]).float()
        """
        # this is the more strict approach where we consider the full list of tokens to identify
        # an entity and its graph embedding
        embs = []
        for v in i['entities'].values():
            ents.append(torch.tensor(v['tokenized']))
            # for multi-token entities we take the mean of the single embeddings
            embs.append(torch.mean(v['embedding'], 0))
        # pad the entities to the same lenght in order to store them in a single torch.tensor
        # [125, 3679, 24]         [125, 3679, 24]
        # [90]             ---->  [90,  -1,   -1]
        # [567, 8971]             [567, 8971, -1]
        max_len = len(max(ents, key=len))
        ents = torch.stack([ torch.nn.functional.pad(i, pad=(0, max_len - i.numel()), mode='constant', value=-1) for i in ents ], dim=0)
        embs = torch.stack(embs, dim=0)
        # store padded entities and mean embeddings in the same tensor
        # [125, 3679, 24, mean_emb1]
        # [90,  -1,   -1, mean_emb2]
        # [567, 8971, -1, mean_emb3]
        # NOTE: the entities are automatically casted from int to float
        embs = torch.hstack((ents,embs)).float()
        """
        # relations 
        rels = torch.tensor([ (
            v['tokenized'][0][-1],
            v['tokenized'][1][-1],
            torch.tensor([rel2index[v['type']]]) )
                              for v in i['relations'].values() ])
        
        data[s].append( (sent, tagged_sent, embs, rels) )                                                   # 1 stands for ADVERSE_EFFECT_OF, 0 for NO_RELATION
                                                                                                            # format : (head, tail, relation)
                                                                                                            # -1 cause our RE module express relations between
                                                                                                            # the last tokens of the two entities

                                                                                                     
# set up the tokenizer and the pre-trained BERT
bert = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(bert)
model = Pipeline(bert, ner_dim=bioes.space_dim, ner_scheme=bioes, ned_dim=50, re_dim=2)

# check if GPU is avilable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('> Found device:', device, ', setting it as the principal device.')

# move model to device
if device == torch.device("cuda:0"):
    model.to(device)

# define the loss
criterion = torch.nn.CrossEntropyLoss()

# define the optimizer
#optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00002)

# set up the trainer
trainer = Trainer(train_data=data['train'],
                  test_data=data['test'],
                  model=model,
                  tokenizer=tokenizer,
                  optim=optimizer,
                  loss_f=criterion,
                  device=device)

# load pretrained model
if args.load_model != None:
    model.load_state_dict(torch.load(args.load_model))
else:
    trainer.train(3)


# ------------------------- Evaluation 
    
sm1 = torch.nn.Softmax(dim=1)
sm0 = torch.nn.Softmax(dim=0)

ner_groundtruth = []
ner_prediction = []
re_groundtruth = []
re_prediction = []
re_gt_pred = []

# making predictions on the validation set
for d in trainer.test_set:
    inputs = tokenizer(d[0], return_tensors="pt").to(device)
    ner_outs, ned_outs, re_outs = model(inputs)
    ner_prediction.append([bioes.to_tag(i) for i in sm1(ner_outs)])
    ner_groundtruth.append([ bioes.index2tag[int(i)] for i in d[1] ])
    
    re_outs = list(zip(*re_outs)) if re_outs != None else None
    if re_outs != None:
        re_outs = { (i[0][0].item(), i[0][1].item()): torch.argmax(sm0(i[1])).item() for i in re_outs }
    else:
        re_outs = {}
    re_gt = { (r[0].item(), r[1].item()): r[2].item() for r in d[3] }
    re_gt_pred = {}
    # filling groundtruth and prediction with the corresponding missing elements
    for k, v in re_outs.items():
        if k not in re_gt.keys():
            re_gt_pred[k] = (0, v)
        else:
            re_gt_pred[k] = (re_gt[k], v)
    for k, v in re_gt.items():
        if k not in re_outs.keys():
            re_gt_pred[k] = (v, 0)
    re_gt_pred = torch.tensor(list(re_gt_pred.values()))
    re_groundtruth.append(re_gt_pred[:,0].tolist())
    re_prediction.append(re_gt_pred[:,1].tolist())
    
    
# Some testing by hand
for i in range(5):
    print('>> NER GROUNDTRUTH\n', ner_groundtruth[i])
    print('>> NER PREDICTION\n', ner_prediction[i])
    print('>> RE GROUNDTRUTH\n', re_groundtruth[i])
    print('>> RE PREDICTION\n', re_prediction[i], '\n')
    
# Performance metrics
from seqeval.metrics import classification_report
from seqeval.scheme import IOBES
from sklearn.metrics import f1_score
import sklearn.metrics as skm


# NER
print('------------------------------ NER SCORES ----------------------------------------')
print(classification_report(ner_groundtruth, ner_prediction, mode='strict', scheme=IOBES))

# RE
print('------------------------------ RE SCORES ----------------------------------------')
print(skm.classification_report(np.concatenate(re_groundtruth), np.concatenate(re_prediction), labels=[1], target_names=['ADVERSE_EFFECT_OF']))
