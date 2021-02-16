import re, torch, sys, random, argparse, pickle
sys.path.append('../')
from pipeline import Pipeline
from transformers import AutoTokenizer
from utils import Trainer, BIOES


# arguments parser
parser = argparse.ArgumentParser(description='Train the model.')
parser.add_argument('input_data', help='Path to input data file.')
parser.add_argument('--load_model', metavar='MODEL', help='Path to pretrained model.')
args = parser.parse_args()

# define the tagging scheme
bioes = BIOES(['AE','DRUG'])

# Load the data
with open(args.input_data, 'rb') as f:               
    pkl = pickle.load(f)
    
data = []
for d in pkl:
    data.append( ( d['sentence']['sentence'],
                   bioes.to_tensor(*d['sentence']['tag'], index=True),
                   torch.tensor([ ( v['tokenized'][0][-1], v['tokenized'][1][-1], torch.tensor([1]) ) for v in d['relations'].values() ])
    ))                                                                                  # 1 stands for ADVERSE_EFFECT_OF, 0 for NO_RELATION
                                                                                        # format : (head, tail, relation)
                                                                                        # -1 cause our RE module express relations between
                                                                                        # the last tokens of the two entities
                                                                      
# set up the tokenizer and the pre-trained BERT
bert = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(bert)
model = Pipeline(bert, ner_dim=bioes.space_dim, ner_scheme=bioes, re_dim=2)

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
    model.load_state_dict(torch.load(args.load_model))
else:
    trainer.train(3)
    
sm = torch.nn.Softmax(dim=1)
            
groundtruth = []
prediction = []
for d in trainer.val_set:
    inputs = tokenizer(d[0], return_tensors="pt").to(device)
    prediction.append([bioes.to_tag(i) for i in sm(model(inputs)[0])])
    groundtruth.append([ bioes.index2tag[int(i)] for i in d[1] ])
    #groundtruth.append(d[1].tolist())
    
# Some testing by hand
for i in range(20):
    print('>> PREDICTION', prediction[i])
    print('>> GROUNDTRUTH', groundtruth[i], '\n')

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.scheme import IOBES

print(classification_report(groundtruth, prediction, mode='strict', scheme=IOBES))
