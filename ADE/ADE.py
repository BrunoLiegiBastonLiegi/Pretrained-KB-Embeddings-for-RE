import re, torch, sys, random
sys.path.append('../')
from pipeline import Pipeline
from ner_schemes import BIOES
from preprocessing import split_sets
from transformers import AutoTokenizer
from utils import Trainer



# define the tagging scheme
bioes = BIOES(['AE','DRUG'])

# Load the data
data = []
#sents, labels = ([],[])
with open('DRUG-AE_BIOES.rel', 'r') as f:
    for x in f:
        tmp = re.split('\|', x)
        data.append((tmp[0],bioes.to_tensor(*tmp[1:-1], index=True)))
        #sents.append(tmp[0])
        #labels.append(bioes.to_tensor(*tmp[1:-1], index=True)) # -1 cause we need to get rid of the '\n'
                                                               # directly convert tag labels to: torch tensors
                                                               # in the ner embedding space/class indexes

#print(sents[1])
#print(labels[1])

# split train and vaildation sets
#train, val = split_sets(data, test=0.)

# set up the tokenizer and the pre-trained BERT
bert = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(bert)
model = Pipeline(bert, ner_dim=bioes.space_dim)

# check if GPU is avilable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('> Found device:', device, ', setting it as the principal device')

if device == torch.device("cuda:0"):
    model.to(device)

# define the loss
criterion = torch.nn.CrossEntropyLoss()

# define the optimizer
#optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00002)

trainer = Trainer(data, model, tokenizer, optimizer, criterion, device)
trainer.train(2)

'''

# training
for epoch in range(3):

    running_loss = 0.0
    val_loss = 0.0
    
    for i in range(len(train)):
    
        inputs = tokenizer(train[i][0], return_tensors="pt")
        target = train[i][1]

        # move inputs and labels to device
        if device == torch.device("cuda:0"):
            inputs = inputs.to(device)
            target = target.to(device)
    
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
        # calculate validation loss
        with torch.no_grad():
            r = random.choice(val)
            inputs = tokenizer(r[0], return_tensors="pt")
            target = r[1]

            if device == torch.device("cuda:0"):
                inputs = inputs.to(device)
                target = target.to(device)
                
            outputs = model(inputs)
            loss = criterion(outputs, target)
            val_loss += loss.item()
        
        if i % 500 == 499:    # print every 500 sentences
            print('[%d, %5d] loss: %.3f val_loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 500, val_loss / 500))
            running_loss = 0.0
            val_loss = 0.0


# test model
print('### SENT\n', val[1][0])
inputs = tokenizer(val[1][0], return_tensors="pt").to(device)
print('### GROUNDTRUTH\n', [ bioes.index2tag[int(i)] for i in val[1][1] ])
sm = torch.nn.Softmax(dim=1)
sm = sm(model(inputs))
print('### PREDICTION\n', [bioes.to_tag(i) for i in sm])
            
'''
