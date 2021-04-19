import re, torch, sys, random, argparse, pickle
sys.path.append('../')
from pipeline import Pipeline
from transformers import AutoTokenizer
from utils import Trainer, BIOES
import numpy as np
import matplotlib.pyplot as plt


# Arguments parser
parser = argparse.ArgumentParser(description='Train the model.')
parser.add_argument('input_data', help='Path to input data file.')
parser.add_argument('--load_model', metavar='MODEL', help='Path to pretrained model.')
parser.add_argument('--fold', metavar='FOLD', help='Number of the fold for k-fold crossvalidation')
args = parser.parse_args()

# Define the tagging scheme
bioes = BIOES(['AE','DRUG'])

# Load the data
with open(args.input_data, 'rb') as f:               
    pkl = pickle.load(f)
if args.fold != None:
    pkl = pkl['fold_' + str(args.fold)]
else:
    pkl = pkl['fold_0'] 

rel2index = {'NO_RELATION': 0, 'ADVERSE_EFFECT_OF': 1}#, 'HAS_ADVERSE_EFFECT': 2} # consider adding a third relation HAS_AE
                                                                                # i.e. AE_OF^-1

# check if GPU is avilable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('> Found device:', device, ', setting it as the principal device.')
                                                                                
# Prepare train and test set with labels
data = {}
embeddings = {}
for s, d in pkl.items():
    data[s] = []
    for i in d:
        # sentence (str)
        sent = i['sentence']['sentence']
        # tags for the sentence ['O', 'B-DRUG', 'E-DRUG', 'O', 'O', ...]
        tagged_sent = bioes.to_tensor(*i['sentence']['tag'], index=True)
        # pretrained graph embeddings of the disambiguated entities
        # for simplicity we take only the last token of each entity as its identifier
        embs = []
        for v in i['entities'].values():
            mean = torch.mean(v['embedding'], dim=0)
            embs.append(torch.hstack((
                #torch.tensor(v['tokenized'][-1]),    # last token of the entity
                torch.tensor(v['span'][-1]),          # last token of the entity
                mean                                  # mean graph embedding
            )))
            embeddings['-'.join(v['concept'])] = mean       
        embs = torch.vstack(embs)
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
            #v['tokenized'][0][-1],
            v['span'][0][-1],
            #v['tokenized'][1][-1],
            v['span'][1][-1],
            torch.tensor([rel2index[v['type']]]) )
                              for v in i['relations'].values() ])
        
        data[s].append( (sent, tagged_sent, embs, rels) )                                                   # 1 stands for ADVERSE_EFFECT_OF, 0 for NO_RELATION
                                                                                                            # format : (head, tail, relation)
                                                                                                            # -1 cause our RE module express relations between
                                                                                                            # the last tokens of the two entities




# set up the tokenizer and the pre-trained BERT
bert = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
#bert = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(bert)
model = Pipeline(bert, ner_dim=bioes.space_dim, ner_scheme=bioes, ned_dim=50, KB=embeddings, re_dim=2)
#model = Pipeline(bert, ner_dim=bioes.space_dim, ner_scheme=bioes, ned_dim=0, re_dim=2)

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

# load pretrained model or train
if args.load_model != None:
    model.load_state_dict(torch.load(args.load_model))
else:
    plots = trainer.train(24)
    #ones = np.ones(10)
    #ner_plot = np.convolve(plots['train']['NER'], ones, 'valid') / len(ones)
    #ned_plot = np.convolve(plots['train']['NED'], ones, 'valid') / len(ones)
    #re_plot = np.convolve(plots['train']['RE'], ones, 'valid') / len(ones)
    # show the plots
    #r = range(len(ner_plot))
    #plt.plot(
     #   r, ner_plot, 'r-',
     #   r, ned_plot, 'b-',
     #   r, re_plot, 'g-'
    #)
    #plt.axis([0,len(r),0,1])
    #plt.show()

# ------------------------- Evaluation 
    
sm1 = torch.nn.Softmax(dim=1)
sm0 = torch.nn.Softmax(dim=0)

ner_groundtruth, ner_prediction = [], []
ned_groundtruth, ned_prediction = [], []
re_groundtruth, re_prediction = [], []

# making predictions on the test set
model.eval()
for d in trainer.test_set:
    inputs = tokenizer(d[0], return_tensors="pt").to(device)
    ner_outs, ned_outs, re_outs = model(inputs)

    # NER
    ner_groundtruth.append([ bioes.index2tag[int(i)] for i in d[1] ])
    ner_prediction.append([ bioes.to_tag(i) for i in sm1(ner_outs) ])

    # NED
    ned_groundtruth.append( (d[2][:,0].tolist(), d[2][:,1:]) )
    if ned_outs != None:
        ned_prediction.append( (torch.flatten(ned_outs[0]).tolist(), ned_outs[1].detach().cpu()) )
    else:
        ned_prediction.append(None)

    # RE
    re_groundtruth.append(d[3])
    if re_outs != None:
        re_prediction.append(torch.hstack((
            re_outs[0],
            torch.argmax(sm1(re_outs[1]), dim=1).view(-1,1)
        )))
    else:
        re_prediction.append(None)


# plot predicted/pretrained graph embeddings
from evaluation import plot_embedding
#plot_embedding(ned_prediction, ned_groundtruth)

# mean neighbors distance in graph embedding space
from evaluation import mean_neighbors_distance
#print('Mean distance between neighbors:',
 #     mean_neighbors_distance(torch.vstack(list(embeddings.values()))))

from evaluation import ClassificationReport, KG

#with open('UMLS-embeddings.pkl', 'rb') as f:
 #   KB = pickle.load(f)

cr = ClassificationReport(
    ner_predictions=ner_prediction,
    ner_groundtruth=ner_groundtruth,
    ned_predictions=ned_prediction,
    ned_groundtruth=ned_groundtruth,
    re_predictions=re_prediction,
    re_groundtruth=re_groundtruth,
    re_classes=rel2index,
    ner_scheme='IOBES',
    ned_embeddings=embeddings
)

index2rel = {v: k for k,v in rel2index.items()}
with open('graph.json', 'w') as f:
    kg = KG(ned_prediction, re_prediction, embeddings, index2rel, save=f)
    
print('------------------------------ NED SCORES ---------------------------------------')
print(cr.ned_report())
print('------------------------------ RE SCORES ----------------------------------------')
print(cr.re_report())

# Performance metrics
from seqeval.metrics import classification_report
from seqeval.scheme import IOBES
from sklearn.metrics import f1_score
import sklearn.metrics as skm


# NER
print('------------------------------ NER SCORES ----------------------------------------')
print(classification_report(ner_groundtruth, ner_prediction, mode='strict', scheme=IOBES))


"""
# finding nearest concepts in the KB
with open('UMLS-embeddings.pkl', 'rb') as f:
    KB = pickle.load(f)

from sklearn.neighbors import NearestNeighbors

ids = list(KB.keys())
#print(len(ids))
#print(len(list(KB.values())))
#[ print(i.shape) for i in KB.values() ] 
#print(torch.vstack(list(KB.values())).shape)
nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(list(KB.values()))
for p in ned_prediction:
    distances, indices = nbrs.kneighbors(p)
    for i in range(len(distances)):
        print('ID:', ids[int(indices[i])], 'Distance:', distances[i])
    #print('indices:',index)
    #print('distances:',distance)
    #for i in nbrs.kneighbors(p):
     #   print(i)
      #  print('ID:', ids[int(i[1])])
"""
