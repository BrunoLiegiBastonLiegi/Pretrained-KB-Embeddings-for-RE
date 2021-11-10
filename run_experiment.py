import torch, argparse, pickle, re, json, random, time
from trainer import Trainer
from ner_schemes import BIOES
from dataset import IEData, Stat
from pipeline import Pipeline, GoldEntities
from model import BaseIEModel, BaseIEModelGoldEntities, IEModel, IEModelGoldEntities, IEModelGoldKG
from transformers import AutoTokenizer
from evaluation import Evaluator
from graph import KnowledgeGraph

# Arguments parser
parser = argparse.ArgumentParser(description='Train a model and evaluate on a dataset.')
parser.add_argument('train_data', help='Path to train data file.')
parser.add_argument('test_data', help='Path to test data file.')
parser.add_argument('--load_model', metavar='MODEL', help='Path to pretrained model.')
parser.add_argument('--out_file', default='results.json', type=str)
parser.add_argument('--n_epochs', default=6, type=int)
parser.add_argument('--n_exp', default=1, type=int)
args = parser.parse_args()

# Input/Output directory
dir = re.search('.+<?\/', args.train_data).group(0)
assert dir == re.search('.+<?\/', args.test_data).group(0)

pkl = {}
# Load the data
with open(args.train_data, 'rb') as f:               
    pkl['train'] = pickle.load(f)
with open(args.test_data, 'rb') as f:               
    pkl['test'] = pickle.load(f)

# Do some statistics and reorganize the data
stat = Stat(pkl['train'], pkl['test'])
data = stat.scan()
rels = {**stat.stat['train']['relation_types'], **stat.stat['test']['relation_types']}
#rels = ['P37','P407','P134','P364','P1018','P282','P103']
#rels, data = stat.filter_rels(len(rels), rels=rels, random=False)
#rels, data = stat.filter_rels(10, random=False, support_range=(100,10000))
#stat.gen()
kg = KnowledgeGraph(stat.edges)
#kg.draw()

# Visualize pretrained embedding space
from utils import plot_embedding
colors = dict(zip(stat.id2type.values(), range(len(stat.id2type)))) # setting colors associated to entity types
colors = dict(zip(colors.keys(), range(len(colors))))
#plot_embedding(torch.vstack(list(stat.kb.values())), [colors[stat.id2type[k]] for k in stat.kb.keys()])

# Define the tagging scheme
bioes = BIOES(list(stat.entity_types.keys()))
# Define the relation scheme
rel2index = dict(zip(rels.keys(), range(len(rels))))
print(rel2index)
# Define the pretrained model
bert = 'bert-base-cased'
#bert = 'dmis-lab/biobert-v1.1'
tokenizer = AutoTokenizer.from_pretrained(bert)

# Prepare data for training
train_data = IEData(
    sentences=data['train']['sent'],
    ner_labels=data['train']['ents'],
    re_labels=data['train']['rels'],
    preprocess=True,
    tokenizer=tokenizer,
    ner_scheme=bioes,
    rel2index=rel2index#,
    #save_to=args.train_data.replace('.pkl', '_preprocessed.pkl')
)
    
test_data = IEData(
    sentences=data['test']['sent'],
    ner_labels=data['test']['ents'],
    re_labels=data['test']['rels'],
    preprocess=True,
    tokenizer=tokenizer,
    ner_scheme=bioes,
    rel2index=rel2index#,
    #save_to=args.test_data.replace('.pkl', '_preprocessed.pkl')
)

# check if GPU is avilable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('> Found device:', device, ', setting it as the principal device.')

# -------------------------------------------------------------------------------------------------
def experiment(model, train_data, test_data, **kwargs):
        
    if model == 'BaseIEModel':
        model = BaseIEModel(
            language_model = kwargs['lang_model'],
            ner_dim = kwargs['ner_dim'],
            ner_scheme = kwargs['ner_scheme'],
            re_dim = kwargs['re_dim'],
            device = kwargs['dev']
        )
    elif model == 'BaseIEModelGoldEntities':
        model = BaseIEModelGoldEntities(
            language_model = kwargs['lang_model'],
            re_dim = kwargs['re_dim'],
            device = kwargs['dev']
        )
    elif model == 'IEModel':
        model = IEModel(
            language_model = kwargs['lang_model'],
            ner_dim = kwargs['ner_dim'],
            ner_scheme = kwargs['ner_scheme'],
            ned_dim = kwargs['ned_dim'],
            KB = kwargs['kb'],
            re_dim = kwargs['re_dim'],
            device = kwargs['dev']
        )
    elif model == 'IEModelGoldEntities':
        model = IEModelGoldEntities(
            language_model = kwargs['lang_model'],
            ned_dim = kwargs['ned_dim'],
            KB = kwargs['kb'],
            re_dim = kwargs['re_dim'],
            device = kwargs['dev']
        )
    elif model == 'IEModelGoldKG':
        model = IEModelGoldKG(
            language_model = kwargs['lang_model'],
            ned_dim = kwargs['ned_dim'],
            re_dim = kwargs['re_dim'],
            device = kwargs['dev']
        )
        
    # move model to device
    #if device == torch.device("cuda:0"):
    #    model.to(device)

    # define the optimizer
    lr = 2e-5
    #optimizer = torch.optim.SGD(model.parameters(), lr=3e-5, momentum=0.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # set up the trainer
    batchsize = 12
    trainer = Trainer(
        train_data=train_data,
        test_data=test_data,
        model=model,
        optim=optimizer,
        device=kwargs['dev'],
        rel2index=kwargs['rel2index'],
        save=False,
        batchsize=batchsize,
        tokenizer=kwargs['tokenizer'],
    )

    # load pretrained model or train
    if args.load_model != None:
        model.load_state_dict(torch.load(args.load_model))
    else:
        plots = trainer.train(kwargs['n_epochs'])
        #yn = input('Save loss plots? (y/n)')
        yn = 'n'
        if yn == 'y':
            with open(dir + '/loss_plots.pkl', 'wb') as f:
                pickle.dump(plots, f)

    # Evaluation
    results = {}
    ev = Evaluator(
        model=model,
        ner_scheme=kwargs['ner_scheme'],
        kb_embeddings=kwargs['kb'],
        re_classes=dict(zip(kwargs['rel2index'].values(), kwargs['rel2index'].keys())),
    )
    scores, matrix = ev.classification_report(test_data)[-1]
    results = {
        'model': re.search('model\.(.+?)\'\>', str(type(model))).group(1),
        'learning_rate': lr,
        'epochs': kwargs['n_epochs'],
        'batchsize': batchsize,
        'scores': scores,
        'confusion matrix': matrix
    }

    return results

# ---------------------------------------------------------------------------------------------------------

runs = {}
#key = {'BaseIEModelGoldEntities': 'without graph embeddings', 'IEModelGoldKG': 'with graph embeddings'}
for n,m in enumerate(['BaseIEModelGoldEntities', 'IEModelGoldKG']):
    for i in range(args.n_exp):
        print('\n################################## RUN {} OF {} ({}) #################################\n'.format(i+1, args.n_exp, m))
        if __name__ == '__main__':
            torch.multiprocessing.set_start_method('spawn', force=True)
            runs['run_'+str(i+1)] = experiment(
                model = m,
                train_data = train_data,
                test_data = test_data,
                lang_model = bert,
                ner_dim = bioes.space_dim,
                ner_scheme = bioes,
                ned_dim = list(stat.kb.values())[0].shape[-1],
                kb = stat.kb,
                re_dim = len(rels),
                dev = device,
                rel2index = rel2index,
                tokenizer = tokenizer,
                n_epochs = args.n_epochs
            )
    out_file = dir + '/' + args.out_file if n == 0 else dir + '/' + args.out_file.replace('results', 'results_kg')
    with open(out_file, 'w') as f:
        json.dump(runs, f, indent=4)

    


