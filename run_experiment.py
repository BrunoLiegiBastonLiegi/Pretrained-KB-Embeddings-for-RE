import torch, argparse, pickle, re, json, random, time
from trainer import Trainer
from ner_schemes import BIOES
from dataset import IEData, Stat
from model import BaseIEModel, BaseIEModelGoldEntities, IEModel, IEModelGoldEntities, IEModelGoldKG
from transformers import AutoTokenizer
from evaluation import Evaluator
from graph import KnowledgeGraph

# Arguments parser
parser = argparse.ArgumentParser(description='Train a model and evaluate on a dataset.')
parser.add_argument('train_data', help='Path to train data file.')
parser.add_argument('test_data', help='Path to test data file.')
parser.add_argument('--load_model', metavar='MODEL', help='Path to pretrained model/s.', nargs='+')
parser.add_argument('--res_file', default='results.json', type=str)
parser.add_argument('--n_epochs', default=6, type=int)
parser.add_argument('--n_exp', default=1, type=int)
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--preprocess', action='store_true')
parser.add_argument('--conf', help='Path to training configuration file.')
args = parser.parse_args()

# Input/Output directory
dir = re.search('.+<?\/', args.train_data).group(0)
assert dir == re.search('.+<?\/', args.test_data).group(0)

# Load the data
print('> Loading train data ...')
with open(args.train_data, 'rb') as f:               
    train = pickle.load(f)
print('Done.')
print('> Loading test data ...')
with open(args.test_data, 'rb') as f:               
    test = pickle.load(f)
print('Done.')

try:
    with open(args.conf, 'r') as f:
        conf = json.load(f)
except:
    conf = {
        'n_epochs': args.n_epochs,
        'n_exp': args.n_exp,
        'batchsize': 8,
        'lr': 2e-5,
        'device': 0,
        'pretrained_lang_model': 'bert-base-cased',
        'evaluate': True,
        'negative_rel_class': 'NA'
    }

# Define the pretrained model
bert = conf['pretrained_lang_model']
#bert = 'roberta-base'
#bert = 'gpt2'
#bert = 'dmis-lab/biobert-v1.1'
tokenizer = AutoTokenizer.from_pretrained(bert)

if args.preprocess:
    # Do some statistics and reorganize the data
    #train = random.sample(train, int(0.8*len(train)))
    stat = Stat(train, test)
    data = stat.scan()
    with open(dir+'stat.json', 'w') as f:
        json.dump(stat.stat, f)
    rels = {**stat.stat['train']['relation_types'], **stat.stat['test']['relation_types']}
    print('###\n{}\n###'.format(len(rels)))
    #rels = ['P37','P407','P134','P364','P1018','P282','P103']
    #rels = ['P37', 'P282', 'P619', 'P620', 'P647', 'P54', 'P2098', 'P1308', 'P2546', 'P1429', 'P59', 'P399', 'P629', 'P655', 'P437', 'P400', 'P140', 'P611', 'P1192', 'P81', 'P684', 'P688', 'P27', 'P17', 'P19', 'P20']
    #rels = ['P37', 'P282', 'P619', 'P620', 'P647', 'P54', 'P2098', 'P1308', 'P2546', 'P1429', 'P59', 'P399', 'P629', 'P655', 'P437', 'P400', 'P140', 'P611', 'P1192', 'P81', 'P684', 'P688', 'P19', 'P20']
    #rels, data = stat.filter_rels(len(rels), rels=rels, random=False)
    [print(k, stat.stat['train']['relation_types'][k]) for k in rels.keys()]
    #rels, data = stat.filter_rels(10, random=False, support_range=(100,10000))
    # Define the tagging scheme
    bioes = BIOES(list(stat.entity_types.keys()))
    # Define the relation scheme
    rel2index = dict(zip(rels.keys(), range(len(rels))))
    print(rel2index)
    ned_dim = list(stat.kb.values())[0].shape[-1]
    kb = stat.kb
    #kg = KnowledgeGraph(stat.edges)
    #kg.draw()

    # Visualize pretrained embedding space
    #from utils import plot_embedding
    #colors = dict(zip(stat.id2type.values(), range(len(stat.id2type)))) # setting colors associated to entity types
    #colors = dict(zip(colors.keys(), range(len(colors))))
    #plot_embedding(torch.vstack(list(stat.kb.values())), [colors[stat.id2type[k]] for k in stat.kb.keys()])

    # Prepare data for training
    train_data = IEData(
        sentences=data['train']['sent'],
        ner_labels=data['train']['ents'],
        re_labels=data['train']['rels'],
        preprocess=True,
        tokenizer=tokenizer,
        ner_scheme=bioes,
        rel2index=rel2index#,
        #save_to=dir+'train_IEData_tmp.pkl'
    )
    
    test_data = IEData(
        sentences=data['test']['sent'],
        ner_labels=data['test']['ents'],
        re_labels=data['test']['rels'],
        preprocess=True,
        tokenizer=tokenizer,
        ner_scheme=bioes,
        rel2index=rel2index#,
        #save_to=dir+'test_IEData_tmp.pkl'
    )
else:
    train_data = train
    test_data = test
    bioes = train.scheme
    rel2index = train.rel2index
    ned_dim = train.samples[0]['emb'].shape[-1]
    kb = torch.unique(
        torch.vstack([ s['emb'] for s in train.samples+test.samples ]),
        dim=0)
    kb = dict(zip(range(kb.shape[0]), kb))
    
# check if GPU is avilable
device = torch.device("cuda:"+str(conf['device']) if torch.cuda.is_available() else "cpu")
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
    lr = kwargs['lr']
    #optimizer = torch.optim.SGD(model.parameters(), lr=3e-5, momentum=0.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # set up the trainer
    batchsize = kwargs['batchsize']
    trainer = Trainer(
        train_data=train_data,
        test_data=test_data,
        model=model,
        optim=optimizer,
        device=kwargs['dev'],
        rel2index=kwargs['rel2index'],
        save=kwargs['save'],
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

    if kwargs['evaluate']:
        # Evaluation
        results = {}
        ev = Evaluator(
            model=model,
            ner_scheme=kwargs['ner_scheme'],
            kb_embeddings=kwargs['kb'],
            re_classes=dict(zip(kwargs['rel2index'].values(), kwargs['rel2index'].keys())),
        )
        scores, matrix, curve = ev.classification_report(test_data, ignore_classes=[kwargs['negative_rel_class']])[-1]
        results = {
            'model': re.search('model\.(.+?)\'\>', str(type(model))).group(1),
            'learning_rate': lr,
            'epochs': kwargs['n_epochs'],
            'batchsize': batchsize,
            'scores': scores,
            'confusion matrix': matrix,
            'pr_curve': curve
        }

        return results
    else:
        return []

# ---------------------------------------------------------------------------------------------------------

runs = {}
#key = {'BaseIEModelGoldEntities': 'without graph embeddings', 'IEModelGoldKG': 'with graph embeddings'}
if args.load_model != None:
    params = {
        'language_model' : bert,
        'ner_dim' : bioes.space_dim,
        'ner_scheme' : bioes,
        'ned_dim' : ned_dim,
        'KB' : kb,
        're_dim' : len(rel2index),
        'device' : device,
    }
    import model
    mtypes = ['BaseIEModel', 'BaseIEModelGoldEntities', 'IEModel', 'IEModelGoldEntities', 'IEModelGoldKG']
    results = []
    for i in args.load_model:
        print(f"> loading model: {i}")
        m = re.findall('(?<=\/)[a-zA-Z]+(?=_)', i)[-1]
        print(f">> inferred Model object: {m}")
        assert m in mtypes
        m = getattr(model, m)(**params)
        m.load_state_dict(torch.load(i))
        ev = Evaluator(
            model=m,
            ner_scheme=bioes,
            kb_embeddings=kb, #dict(zip(range(kb.shape[0]), kb)),
            re_classes=dict(zip(rel2index.values(), rel2index.keys())),
            batchsize=128
        )
        scores, matrix, curve = ev.classification_report(test_data, ignore_classes=[conf['negative_rel_class']])[-1]
        results.append({
            'model': re.search('model\.(.+?)\'\>', str(type(m))).group(1),
            'learning_rate': lr,
            'epochs': kwargs['n_epochs'],
            'batchsize': batchsize,
            'scores': scores,
            'confusion matrix': matrix,
            'pr_curve': curve
        })
    if len(results) > 1:
        runs = { 'run_{}'.format(i): r for i,r in enumerate(results) }
        out_file = dir + '/' + args.res_file
        with open(out_file, 'w') as f:
            json.dump(runs, f, indent=4)
else:
    t = time.time()
    for n,m in enumerate(['BaseIEModelGoldEntities', 'IEModelGoldKG']):
        for i in range(args.n_exp):
            print('\n################################## RUN {} OF {} ({}) #################################\n'.format(i+1, args.n_exp, m))
            if __name__ == '__main__':
                #torch.multiprocessing.set_start_method('spawn', force=True)
                runs['run_'+str(i+1)] = experiment(
                    model = m,
                    train_data = train_data,
                    test_data = test_data,
                    lang_model = bert,
                    ner_dim = bioes.space_dim,
                    ner_scheme = bioes,
                    ned_dim = ned_dim,
                    kb = kb,
                    re_dim = len(rel2index),
                    dev = device,
                    rel2index = rel2index,
                    tokenizer = tokenizer,
                    n_epochs = conf['n_epochs'],
                    evaluate = conf['evaluate'],
                    batchsize = conf['batchsize'],
                    lr = conf['lr'],
                    negative_rel_class = conf['negative_rel_class'],
                    #save = dir + m + '_{}_{:.2f}.pth'.format(i+1,t)
                    save = dir + m + '_{}.pth'.format(i+1)
                )
        out_file = dir + '/' + args.res_file if n == 0 else dir + '/' + args.res_file.replace('results', 'results_kg')
        with open(out_file, 'w') as f:
            json.dump(runs, f, indent=4)

    


