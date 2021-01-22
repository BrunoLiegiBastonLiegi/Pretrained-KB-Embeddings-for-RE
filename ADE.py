from ke import Pipeline
import re, torch
from transformers import AutoTokenizer


# Load the data
data = []
with open('ADE/DRUG-AE_BIOES.rel', 'r') as f:
    for x in f:
        data.append(re.split('\|', x))


# set up the tokenizer and the pre-trained BERT
bert = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(bert)
model = Pipeline(bert)

# define the loss
criterion = torch.nn.CrossEntropyLoss

# check if GPU is avilable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('> Found device:', device, ', setting it as the principal device')

if device == torch.device("cuda:0"):
    model.to(device)

for i in data:
    inputs = tokenizer(i[0], return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    print(model(inputs))
