from utils import BIOES, IEData, Trainer
import pickle, torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from pipeline import Pipeline

with open('ADE/DRUG-AE_BIOES_dmis-lab-biobert-v1.1_10-fold.pkl', 'rb') as f:
    d = pickle.load(f)

d = d['fold_0']['train']
sents, ents, ned, rels = [], [], [], []
for i in d:
    sents.append(i['sentence']['sentence'])
    ents.append(i['entities'])
    ned.append(1)
    rels.append(i['relations'])

#print(sents)
#print(ents)
#print(rels)


bioes = BIOES(['AE','DRUG'])

bert = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(bert)
model = AutoModel.from_pretrained(bert)
rel2index = {'NO_RELATION':0, 'ADVERSE_EFFECT_OF':1}

data = IEData(
    sentences=sents,
    ner_labels=ents,
    ned_labels=ned,
    re_labels=rels,
    tokenizer=tokenizer,
    ner_scheme=bioes,
    rel2index=rel2index#,
    #save_to='ADE.pkl'
)

#print(data[34])
#print(model(data.__getitem__(457)['sent']))
with open('ADE/entity2embedding.pkl','rb') as f:
    kb = pickle.load(f)
for k,v in kb.items():
    kb[k]=v[0].float()

batchsize=32
dataloader = DataLoader(data, batch_size=batchsize, shuffle=True, collate_fn=data.collate_fn)
model = Pipeline(bert, ner_dim=bioes.space_dim, ner_scheme=bioes, ned_dim=50, KB=kb, re_dim=2).cuda()

trainer = Trainer(
    data,
    data,
    model,
    torch.optim.AdamW(model.parameters(), lr=0.00002),
    torch.device("cuda:0")
)

trainer.train(5)

#print(batch)
#print(model(batch['sent']).last_hidden_state)
#print(batch)
#l = batch[0].shape[1]
#t=[]
#for i in range(len(batch)):
 #   if batch[i].shape[1] == l:
  #      t.append(batch[i])
#t = torch.vstack(t)
#print(t.shape)
#print(model(t).last_hidden_state.shape)

#for i, batch in enumerate(dataloader):
 #   print(i)
  #  print(model(batch['sent']))
