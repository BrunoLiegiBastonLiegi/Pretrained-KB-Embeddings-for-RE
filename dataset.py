import pickle, torch, time, numpy, re
import random as rand
from tqdm import tqdm
from transformers.tokenization_utils_base import BatchEncoding
#from torch.multiprocessing import Pool, cpu_count, set_start_method, set_sharing_strategy
#set_sharing_strategy('file_system')
import matplotlib.pyplot as plt
from itertools import repeat


class IEData(torch.utils.data.Dataset):

    def __init__(self, sentences, ner_labels, re_labels, preprocess=False, ned_labels=None, tokenizer=None, ner_scheme=None, rel2index=None, save_to=None):
        self.tokenizer = tokenizer
        self.scheme = ner_scheme
        self.rel2index = rel2index
        self.samples = []
        self.pad = self.tokenizer('[PAD]', add_special_tokens=False)['input_ids'][0]
        if preprocess:
            print('> Preprocessing labels.')
            assert ner_scheme != None, 'Missing NER scheme for preprocessing.'
            #with Pool(12) as p:
            #    self.samples = list(p.starmap(self.generate_labels, tqdm(zip(sentences, ner_labels, re_labels))))
            self.samples = list(tqdm(map(self.generate_labels, sentences, ner_labels, re_labels), total=len(sentences)))
            self.samples = list(filter(lambda x: x != {}, self.samples))
        else:
            assert ned_labels != None, 'Missing NED labels.'
            for s, ner, ned, re in zip(sentences, ner_labels, ned_labels, re_labels):
                self.samples.append({
                    'sent': s,
                    'ner': ner,
                    'ned': ned,
                    're': re
                })
        print('> Done.')
        if save_to != None:
            print('> Saving to \'{}\'.'.format(save_to))
            with open(save_to, 'wb') as f:
                pickle.dump(self, f)

    def generate_labels(self, s, ner, re):
        s_tk = self.tokenizer(s, return_tensors='pt', add_special_tokens=False)['input_ids']
        #print(s)
        #print(ner)
        if s_tk.shape[1] > self.tokenizer.model_max_length:
            print(f'> Discarding the sentence \n{s}\n > exceeding maximum sequence length of the pretrained language model.')
            return {}
        names, span2span, spans, types = {}, {}, [], []
        for k, e in ner.items():
            try:
                names[e['name']] += 1
            except:
                names[e['name']] = 0
            try:
                tk = self.tokenizer(e['name'], add_special_tokens=False)['input_ids']
                span = self.find_span(s_tk.flatten().tolist(), tk, names[e['name']])
            except:
                try:
                    names[' ' + e['name']] += 1
                except:
                    names[' ' + e['name']] = 0
                tk = self.tokenizer(' ' + e['name'], add_special_tokens=False)['input_ids']
                span = self.find_span(s_tk.flatten().tolist(), tk, names[' ' + e['name']])
            span2span[k] = span
            spans.append(span)
            types.append(e['type'])
        lab = {
            'sent': self.tokenizer(s, return_tensors='pt')['input_ids'],
            'pos': (torch.tensor(spans), self.spans2matrix(spans, s_tk.shape[1])),
            'emb': torch.vstack([torch.mean(e['embedding'], dim=0) for e in ner.values()]),
            'ner': self.tag_sentence(s_tk.flatten().tolist(), types, spans),
            'ned': torch.vstack([ # probably it's a good idea to get rid of the initial position now that we specifically 
                torch.hstack((                       # have the position key
                    torch.tensor(spans[i][1]),
                    torch.mean(e['embedding'], dim=0) # need mean for multi-concept entities
                ))
                for i, e in enumerate(ner.values())
            ]),
            're': torch.vstack([  # same concern about position as with ned
                torch.tensor([
                    span2span[k[0]][1],
                    span2span[k[1]][1],
                    self.rel2index[r]
                ])
                for k, r in re.items()
            ])
            }
        return lab

    def spans2matrix(self, spans, dim):
        m = torch.zeros(len(spans), dim)
        for i, s in enumerate(spans):
            m[i][s[0]:s[1]] = 1 / (s[1]-s[0])
        return m
            
    def find_span(self, sent, ent, n=0):
        """
        Find the span of the entity in the tokenization scheme provided by the tokenizer.
        We consider only the case of the same entity occuring just once for each sentence.
        """
        match = []
        #print('///////////////////////////////////////////////////')
        #print(sent)
        #print(self.tokenizer.decode(sent))
        #print(ent)
        #print(self.tokenizer.decode(ent))
        #print('///////////////////////////////////////////////////')
        for i in range(len(sent)):
            if sent[i] == ent[0] and sent[i:i+len(ent)] == ent: 
                #match = (i, i+len(ent))
                match.append((i, i+len(ent)))
        return match[n]


    def tag_sentence(self, sent, types, spans):
        tags = torch.tensor([self.scheme.to_tensor('O', index=True) for i in range(len(sent))])
        for t,s in zip(types, spans):
            if s[1]-s[0] == 1:
                tags[s[0]] = self.scheme.to_tensor('S-' + t, index=True)
            else:
                tags[s[0]:s[1]] = self.scheme.to_tensor(*(['B-' + t] + [('I-' + t) for j in range((s[1]-s[0])-2)] + ['E-' + t]), index=True)
        return tags.view(1,-1)
                
    def collate_fn(self, batch):
        t1 = time.time()
        """
        Function to vertically stack the batches needed by the torch.Dataloader class
        """
        # alternatively I could use the huggingface tokenizer with option pad=True
        tmp = {'sent':[], 'pos': [], 'pos_matrix': [], 'emb': [], 'ner':[], 'ned':[], 're':[]} # we need to add padding in order to vstack the sents.
        max_len = 0
        for item in batch:
            #max_len = max(max_len, item['sent'][:,:-1].shape[1]) # -1 for discarding [SEP]
            max_len = max(max_len, item['sent'].shape[1])
            #tmp['sent'].append(item['sent'][:,:-1])
            tmp['sent'].append(item['sent'])
            tmp['pos'].append(item['pos'][0])
            tmp['pos_matrix'].append(item['pos'][1])
            tmp['emb'].append(item['emb'])
            tmp['ner'].append(item['ner'])
            tmp['ned'].append(item['ned'])
            tmp['re'].append(item['re'])

        sent = {'input_ids': [], 'attention_mask': []}
        sent['input_ids'] = torch.vstack(list(map(
            lambda x: torch.hstack((x, self.pad*torch.ones(1, max_len - x.shape[1]).int())),
            tmp['sent']
        )))
        sent['attention_mask'] = torch.vstack(list(map(
            lambda x: torch.hstack((torch.ones(1, x.shape[1]), torch.zeros(1, max_len - x.shape[1]))).int(),
            tmp['sent']
        )))
        tmp['sent'] = BatchEncoding(sent)
        tmp['pos'] = torch.nn.utils.rnn.pad_sequence(tmp['pos'], batch_first=True, padding_value=-1)
        m = torch.nn.utils.rnn.pad_sequence(
            list(map(
                lambda x: torch.nn.functional.pad(x, (0, max_len -2 -x.shape[-1]), "constant", 0),
                tmp['pos_matrix']
            )),
            batch_first=True)
        #print(tmp['pos_matrix'][0])
        #print('m:\n',m[0])
        M = torch.zeros(len(batch), len(batch), m.shape[1], m.shape[2])
        for i,mm in enumerate(m):
            M[i,i] = mm
        tmp['pos_matrix'] = M
        # add the [SEP] at the end
        #tmp['sent'] = torch.hstack((tmp['sent'], self.sep*torch.ones(tmp['sent'].shape[0],1).int()))
        O = self.scheme.to_tensor('O', index=True)
        # -1 because max_len counts also the [CLS]
        #print(max_len)
        #for i in tmp['ner']:
         #   print(i.shape[1] ,max_len -2 -i.shape[1])
        tmp['ner'] = torch.vstack(list(map(
            #lambda x: torch.hstack((x, O*torch.ones(1, max_len - 1 - x.shape[1]).int())),
            lambda x: torch.hstack((x, O*torch.ones(1, max_len - 2 - x.shape[1]).int())),
            tmp['ner']
        )))
        tmp['emb'] = torch.nn.utils.rnn.pad_sequence(tmp['emb'], batch_first=True, padding_value=0)
        t2 = time.time()
        #print('> Collate fn:', t2-t1)
        return tmp
            
    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


# --------------------------------------------------------------------------------------------------------------

def normalize(text):
    return re.sub(r'[^\w\s]','', text.lower()).strip()

class Stat(object):
    """
    Object for the statisical analysis of a dataset.
    """
    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.stat = None

    @property
    def entity_types(self):
        assert self.stat != None
        return { **self.stat['train']['entity_types'], **self.stat['test']['entity_types'] }
    @property
    def relation_types(self):
        assert self.stat != None
        return { **self.stat['train']['relation_types'], **self.stat['test']['relation_types'] }
    
    def scan(self):
        self.stat, self.kb, self.split, common = {}, {}, {}, {'kb2txt':{}, 'txt2kb':{}}
        self.id2type, self.edges = {}, []
        discarded_sents = []
        for s,d in zip(('train', 'test'), (self.train, self.test)):
            self.stat[s] = {
                'entity_types': {},
                'relation_types': {},
                'kb_entities': {},
                'entities': {},
            }
            self.split[s] = {
                'sent': [],
                'ents': [],
                'rels': []
            }
            if s == 'test': # this is done to preserve keys order in test/train
                for k in self.stat[s].keys():
                    self.stat['test'][k] = dict(zip(
                        self.stat['train'][k].keys(),
                        repeat(0, len(self.stat['train'][k].keys()))
                    ))
            for i,v in enumerate(d):
                print(' Scanning {} set. ({}/{})'.format(s, i, len(d)), end='\r')
                discard = False
                for e in v['entities'].values():
                    try:
                        emb_flag = e['embedding'].any() != None
                    except:
                        #e['embedding'] = torch.ones(1, 200)
                        emb_flag = False
                    e['type'] = e['type'] if e['type'] != None else 'NA'
                    if e['type'] != None and emb_flag:
                        self.id2type[e['id']] = e['type']
                        self.kb[e['id']] = torch.from_numpy(e['embedding']).float().view(1,-1)
                        e['embedding'] = self.kb[e['id']]
                        #self.kb[e['id']] = torch.tensor(e['embedding']).float().view(1, -1).mean(0).view(1, -1)
                        #e['embedding'] = torch.tensor(e['embedding']).float().view(1, -1).mean(0).view(1, -1)
                        #self.kb[e['id']] = torch.tensor(e['embedding']).float().mean(0).view(1, -1).view(1, -1)
                        #e['embedding'] = torch.tensor(e['embedding']).float().mean(0).view(1, -1).view(1, -1)
                        for k, l in zip(('entities', 'kb_entities', 'entity_types'), ('name', 'id', 'type')):
                            try:
                                self.stat[s][k][e[l]] += 1
                            except:
                                self.stat[s][k][e[l]] = 1
                        for k, l in zip(('kb2txt', 'txt2kb'), (('id', 'name'), ('name', 'id'))):
                            try:
                                if k == 'kb2txt':
                                    common[k][e[l[0]]][normalize(e[l[1]])] += 1
                                elif k == 'txt2kb':
                                    common[k][normalize(e[l[0]])][e[l[1]]] += 1
                            except:
                                try:
                                    if k == 'kb2txt':
                                        common[k][e[l[0]]][normalize(e[l[1]])] = 1
                                    elif k == 'txt2kb':
                                        common[k][normalize(e[l[0]])][e[l[1]]] = 1
                                except:
                                    if k == 'kb2txt':
                                        common[k][e[l[0]]] = {normalize(e[l[1]]): 1}
                                    elif k == 'txt2kb':
                                        common[k][normalize(e[l[0]])] = {e[l[1]]: 1}
                    else:
                        if s == 'train':
                            discard = True
                            #e['embedding'] = torch.zeros(1, list(self.kb.values())[0].shape[-1]) if not emb_flag else torch.tensor(e['embedding']).float().view(1, -1).mean(0).view(1, -1)
                        elif s == 'test':
                            e['embedding'] = torch.zeros(1, list(self.kb.values())[0].shape[-1]) if not emb_flag else torch.tensor(e['embedding']).float().view(1, -1).mean(0).view(1, -1)
                            if e['type'] == None:
                                e['type'] = 'NA' 
                            
                            
                if discard:
                    discarded_sents.append((v['sentence'], v['entities'], v['relations']))
                else:
                    for k,r in v['relations'].items():
                        self.edges.append((v['entities'][k[0]]['id'], v['entities'][k[1]]['id'], r))
                        try:
                            self.stat[s]['relation_types'][r] += 1
                        except:
                            self.stat[s]['relation_types'][r] = 1
                    self.split[s]['sent'].append(v['sentence'][0])
                    self.split[s]['ents'].append(v['entities'])
                    self.split[s]['rels'].append(v['relations'])
            print('\n')
        # This is done to complete the train dict with elements only present in the test dict
        tot = {}
        for k in self.stat['train'].keys():
            keys = list(self.stat['test'][k].keys()) + list(self.stat['train'][k].keys())
            tot[k] = dict(zip(
                keys,
                repeat(0, len(keys))
            ))
            self.stat['train'][k] = {**tot[k], **self.stat['train'][k]}
            self.stat['test'][k] = {**tot[k], **self.stat['test'][k]}
        self.stat['common'] = common
        print('> Discarded {} sentences out of {}, due to incomplete annotations.'.format(len(discarded_sents), len(self.train)+len(self.test)))
        return self.split

    def examples(self, ax1, ax2):
        print('EXAMPLES AVAILABLE')
        ax1.hist(self.stat['train']['kb_entities'].values(), bins='auto')
        ax1.set_ylabel('Text Entities')
        ax1.set_title('Examples')
        ax2.hist(self.stat['train']['entities'].values(), bins='auto')
        ax2.set_ylabel('KB Entities')

    def support(self, ax1, ax2):
        print('TEST SET SUPPORT')
        ax1.hist(self.stat['test']['kb_entities'].values(), bins='auto')
        ax1.set_title('Support')
        ax2.hist(self.stat['test']['entities'].values(), bins='auto')

    def ambiguity(self, ax1, ax2):
        print('KB TO TEXT MAPPING')
        ax1.hist([len(i) for i in self.stat['common']['kb2txt'].values()], bins='auto')
        ax1.set_title('Ambiguity')
        print('TEXT TO KB MAPPING')
        ax2.hist([len(i) for i in self.stat['common']['txt2kb'].values()], bins='auto')
    
    def gen(self):
        fig, axs = plt.subplots(2,3)
        if self.stat == None:
            self.stat = self.scan()
        self.examples(axs[0,0], axs[1,0])
        self.support(axs[0,1], axs[1,1])
        self.ambiguity(axs[0,2], axs[1,2])
        plt.show()

    def filter_rels(self, n, rels=None, total_support=False, random=False, support_range=None):
        assert self.stat != None
        if rels != None:
            assert n == len(rels)
            rels = dict(zip(rels, range(len(rels))))
        else:
            if total_support:
                r_supp = { k: v + self.stat['train']['relation_types'][k] for k,v in self.stat['test']['relation_types'].items() }
            else:
                r_supp = { k: self.stat['train']['relation_types'][k] for k in self.stat['test']['relation_types'].keys() }
            if random:
                rels = dict(rand.sample(r_supp.items(), n))
            elif support_range != None:
                rels = { k: v for k,v in r_supp.items() if v >= support_range[0] and v <= support_range[1] }
                if len(rels) > n:
                    rels = dict(sorted(rels.items(), key=lambda x: x[1], reverse=True)[:n])
            else:
                rels = dict(sorted(r_supp.items(), key=lambda x: x[1], reverse=True)[:n])
        for t in (self.train, self.test):
            for s in t:
                new_rel = {}
                for k,v in s['relations'].items():
                    if v in rels.keys():
                        new_rel[k] = v
                s['relations'] = new_rel
                        
        self.train = [s for s in self.train if len(s['relations']) > 0]
        self.test = [s for s in self.test if len(s['relations']) > 0]
        return rels, self.scan()

    def score_vs_support(self, scores):
        #d = [ [k, [v, scores[k]]] for k,v in self.stat['train']['entity_types'].items() if k in scores.keys()]
        d = [ [k, [self.stat['train']['entity_types'][k], v]] for k,v in scores.items()]
        d = sorted(d, key=lambda x: x[1][0])
        xy = numpy.array([ i[1] for i in d ])
        plt.scatter(xy[:,0], xy[:,1])
        plt.plot(xy[:,0], xy[:,1])
        for i in d:
            plt.annotate(i[0], i[1])
        plt.savefig('score_vs_support')
        plt.show()
