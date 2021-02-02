import re, torch, sys, random, argparse
sys.path.append('../')
from pipeline import Pipeline
from transformers import AutoTokenizer
from utils import Trainer, BIOES, F1


# arguments parser
parser = argparse.ArgumentParser(description='Train the model.')
parser.add_argument('input_data', help='Path to input data file.')
parser.add_argument('--load_model', metavar='MODEL', help='Path to pretrained model.')
args = parser.parse_args()

# define the tagging scheme
bioes = BIOES(['AE','DRUG'])

# Load the data
data = []
with open('DRUG-AE_BIOES.rel', 'r') as f:
    for x in f:
        tmp = re.split('\|', x)
        data.append((tmp[0],bioes.to_tensor(*tmp[1:-1], index=True))) # -1 cause we need to get rid of the '\n'
                                                                      # directly convert tag labels to: torch tensors
                                                                      # in the ner embedding space/class indexes

# set up the tokenizer and the pre-trained BERT
bert = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(bert)
model = Pipeline(bert, ner_dim=bioes.space_dim)

# check if GPU is avilable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('> Found device:', device, ', setting it as the principal device.')

if device == torch.device("cuda:0"):
    model.to(device)

# define the loss
criterion = torch.nn.CrossEntropyLoss()

# define the optimizer
#optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00002)

trainer = Trainer(data=data,
                  model=model,
                  tokenizer=tokenizer,
                  optim=optimizer,
                  loss_f=criterion,
                  device=device,
                  validation=0.2)

if args.load_model != None:
    # load pretrained model
    model.load_state_dict(torch.load('bert_gradually_unfreezed.pth'))
else:
    trainer.train(2)
    
sm = torch.nn.Softmax(dim=1)
            
groundtruth = []
prediction = []
for d in trainer.val_set:
    inputs = tokenizer(d[0], return_tensors="pt").to(device)
    prediction.append([bioes.to_tag(i) for i in sm(model(inputs))])
    groundtruth.append([ bioes.index2tag[int(i)] for i in d[1] ])
    

for i in range(20):
    print('>> PREDICTION', prediction[i])
    print('>> GROUNDTRUTH', groundtruth[i], '\n')


from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np


mlb = MultiLabelBinarizer()
groundtruth = mlb.fit_transform(groundtruth)
prediction  = mlb.fit_transform(prediction)
print('> Accuracy:', accuracy_score(groundtruth, prediction))
print('> Micro F1:', f1_score(groundtruth, prediction, average='micro'))
print('> Macro F1:', f1_score(groundtruth, prediction, average='macro'))
