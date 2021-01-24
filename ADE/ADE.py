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
        labels.append(bioes.to_tensor(*tmp[1:-1])) # -1 cause we need to get rid of the '\n'
                                                   # directly convert tag labels to torch tensors
                                                   # in the ner embedding space

#print(sents[1])
#print(labels[1])

# set up the tokenizer and the pre-trained BERT
bert = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(bert)
model = Pipeline(bert)

# define the loss
criterion = torch.nn.CrossEntropyLoss()

# check if GPU is avilable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('> Found device:', device, ', setting it as the principal device')

if device == torch.device("cuda:0"):
    model.to(device)


    
for i in range(len(sents)):
    inputs = tokenizer(sents[i], return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    outputs = model(inputs)
    #print(outputs)
    print(criterion(model(inputs),labels[i].to(device)))
