import re, torch, sys
sys.path.append('../')
from pipeline import Pipeline
from ner_schemes import BIOES
from transformers import AutoTokenizer



# define the tagging scheme
bioes = BIOES(['AE','DRUG'])

# Load the data
#data = []
sents, labels = ([],[])
with open('DRUG-AE_BIOES.rel', 'r') as f:
    for x in f:
        tmp = re.split('\|', x)
        #data.append(re.split('\|', x))
        sents.append(tmp[0])
        labels.append(bioes.to_tensor(*tmp[1:-1], index=True)) # -1 cause we need to get rid of the '\n'
                                                               # directly convert tag labels to: torch tensors
                                                               # in the ner embedding space/class indexes

#print(sents[1])
#print(labels[1])

# set up the tokenizer and the pre-trained BERT
bert = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(bert)
model = Pipeline(bert, ner_dim=bioes.space_dim)

# define the loss
criterion = torch.nn.CrossEntropyLoss()

# define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# check if GPU is avilable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('> Found device:', device, ', setting it as the principal device')

if device == torch.device("cuda:0"):
    model.to(device)

# training
for epoch in range(1):

    running_loss = 0.0
    
    for i in range(len(sents)):
    
        inputs = tokenizer(sents[i], return_tensors="pt")
        target = labels[i]

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
        
        if i % 500 == 499:    # print every 500 sentences
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0


print('### SENT\n', sents[1])
inputs = tokenizer(sents[1], return_tensors="pt").to(device)
print('### GROUNDTRUTH\n',labels[1])
sm = torch.nn.Softmax(dim=1)
sm = sm(model(inputs))
print('### PREDICTION\n', [bioes.to_tag(i) for i in sm])
            
