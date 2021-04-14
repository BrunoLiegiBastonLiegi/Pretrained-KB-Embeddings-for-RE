import argparse, pickle, torch, re
from pipeline import Pipeline
from transformers import AutoTokenizer
from kg import KG
from utils import BIOES

# Arguments parser
parser = argparse.ArgumentParser(description='Build the Knowledge Graph of input.')
parser.add_argument('input_data', help='Path to input data file.')
parser.add_argument('--load_model', metavar='MODEL', help='Path to pretrained model.')
parser.add_argument('--graph_embeddings', metavar='EMBS', help='Pretrained graph embeddings of the Knowledge Base.')
args = parser.parse_args()

# Load the data
print('> Loading text to process.')
with open(args.input_data, 'r') as f:
    data = f.readlines()

# Load the graph embeddings
print('> Loading the Knowledge Base.')
with open(args.graph_embeddings, 'rb') as f:
    KB = pickle.load(f)

# set up the tokenizer and the pre-trained BERT
bert = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
#bert = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(bert)

# Define the tagging scheme
bioes = BIOES(['AE','DRUG'])

model = Pipeline(bert, ner_dim=bioes.space_dim, ner_scheme=bioes, ned_dim=50, re_dim=2)

# check if GPU is avilable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('> Found device:', device, '\n> setting it as the principal device.')

# move model to device
if device == torch.device("cuda:0"):
    model.to(device)

print('> Loading the Model.')
model.load_state_dict(torch.load(args.load_model))
model.eval()

print('> Processing text.')
ned, re = [], []
sm1 = torch.nn.Softmax(dim=1)
for d in data:
    inputs = tokenizer(d, return_tensors="pt").to(device)
    ner_outs, ned_outs, re_outs = model(inputs)
    if ned_outs != None and re_outs != None:
        ned.append((torch.flatten(ned_outs[0]).tolist(), ned_outs[1].detach().cpu()))
        re.append(torch.hstack((
            re_outs[0],
            torch.argmax(sm1(re_outs[1]), dim=1).view(-1,1)
        )))

rel2index = {'NO_RELATION': 0, 'ADVERSE_EFFECT_OF': 1}
index2rel = {v: k for k,v in rel2index.items()}

print('> Building the Knowledge Graph.')
g = KG(ned=ned, re=re, KB=KB, relations=index2rel)
g.build()
with open(re.sub('.txt', '.json', args.input_data), 'w') as f:
    g.json(save_to=f)
g.draw(with_labels=True)

