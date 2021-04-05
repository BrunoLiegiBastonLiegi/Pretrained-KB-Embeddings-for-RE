import torch, pickle
import matplotlib.pyplot as plt
from transformers import Autotokenizer


with open('DRUG-AE_BIOES_10-fold.pkl', 'rb') as f:               
    pkl = pickle.load(f)
pkl = pkl['fold_0'] # get only the 0-fold for testing

bert_dim = 778

bert = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(bert)

data = {}
embeddings = {} 
for s, d in pkl.items():
    data[s] = []
    for i in d:
        sent = i['sentence']['sentence']
        embs = []
        for v in i['entities'].values():
            mean = torch.mean(v['embedding'], dim=0)
            #embs.append(torch.hstack((
                #torch.tensor(v['tokenized'][-1]),    # last token of the entity
                #torch.tensor(v['span'][-1]),          # last token of the entity
                #mean                                  # mean graph embedding
            #)))
            embeddings[v['concept'][-1]] = mean       # taking last concept as identifier
            embs.append(mean)
        labels = torch.vstack(embs)
        inputs = torch.randn(labels.shape[0], bert_dim)
        data[s].append((inputs, labels))
        
train_set = data['train']
test_set = data['test']

act = torch.nn.ReLU()
#act = torch.nn.Tanh()
dout = torch.nn.Dropout(p=0.1)
layer = torch.nn.Linear(512,512)
#layer = torch.nn.Conv1d(512,512,1)

class mapper(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(mapper, self).__init__()
        h = 128
        self.in_layer = torch.nn.Linear(in_dim, h)
        self.out_layer = torch.nn.Linear(h, out_dim)
        self.act = torch.nn.ReLU()
        self.dout = torch.nn.Dropout(p=0.5)
        #self.h_layer = torch.nn.Conv1d(512,512,1)
        self.h_layer = torch.nn.Linear(h,h)

    def forward(self, x):
        #x = self.act(self.in_layer(x).view(-1,512,1))
        x = self.dout(self.act(self.in_layer(x)))
        x = self.dout(self.act(self.h_layer(x)))
        x = self.dout(self.act(self.h_layer(x)))
        x = self.dout(self.act(self.h_layer(x)))
        #x = self.act(self.h_layer(x).view(-1,512))
        x = self.dout(self.act(self.h_layer(x)))
        x = self.dout(self.act(self.h_layer(x)))
        x = self.dout(self.act(self.h_layer(x)))
        x = self.out_layer(x)
        return x


"""
model = torch.nn.Sequential(
    torch.nn.Linear(bert_dim, 512),
    act,
    layer,
    act,
    layer,
    act,
    layer,
    act,
    torch.nn.Linear(512, 50)
)
"""
model = mapper(bert_dim, 50)

def loss_f(gt, pred):
    return torch.sum(torch.sqrt(torch.sum((gt-pred)**2, dim=1)))

optim = torch.optim.AdamW(model.parameters(), lr=0.00002)

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if dev == torch.device("cuda:0"):
    model.to(dev)

train_losses = []
test_losses = []
    
epochs = 32
for e in range(epochs):
    r_loss = 0.
    model.train()
    for i, d in enumerate(train_set):
        inputs = d[0].to(dev)
        labels = d[1].to(dev)
        optim.zero_grad()
        outputs = model(inputs)
        loss = loss_f(labels, outputs)
        train_losses.append(loss.item())
        loss = 100*loss
        loss.backward()
        optim.step()
        r_loss += loss.item()
        if i % 500 == 499:    # print every 500 sentences
            print('[%d, %5d] loss: %.3f' %
                  (e + 1, i + 1, r_loss / 500))
            r_loss = 0.
    model.eval()
    with torch.no_grad():
        r_loss = 0.
        for d in test_set:
            inputs = d[0].to(dev)
            labels = d[1].to(dev)
            outputs = model(inputs)
            loss = loss_f(labels, outputs).item()
            test_losses.append(loss)
            r_loss += loss
        print('> Test Loss: %.3f' % (r_loss/len(test_set)))


