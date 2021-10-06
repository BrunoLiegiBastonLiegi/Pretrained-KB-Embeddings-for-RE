import numpy
import matplotlib.pyplot as plt

class Stat(object):
    
    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.stat = None

    def scan(self):
        self.stat, common = {}, {'kb2txt':{}, 'txt2kb':{}}
        discard_sents = []
        for s,d in zip(('train', 'test'), (self.train, self.test)):
            self.stat[s] = {
                'entity_types': {},
                'relation_types': {},
                'kb_entities': {},
                'entities': {},
            }
            if s == 'test': # this is done to preserve keys order in test/train
                for k in self.stat[s].keys():
                    self.stat['test'][k] = dict(zip(
                        self.stat['train'][k].keys(),
                        repeat(0, len(self.stat['train'][k].keys()))
                    ))
            for v in d:
                discard = False
                for e in v['entities'].values():
                    try:
                        emb_flag = e['embedding'].any() != None
                    except:
                        emb_flag = False
                if e['type'] != None and emb_flag:
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
                    discard = True 
                for r in v['relations'].values():
                    try:
                        self.stat[s]['relation_types'][r] += 1
                    except:
                        self.stat[s]['relation_types'][r] = 1
                if discard:
                    discarded_sents.append((v['sentence'], v['entities'], v['relations']))
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
        print('> Discarded {} sentences, due to incomplete annotations.'.format(len(discarded_sents)))
        return self.stat

    def examples(self, th=1):
        print('EXAMPLES AVAILABLE')
        kb_ex = [i for i in self.stat['train']['kb_entities'].items() if i[1]>=th]
        print('> {} KB entities ({}%) appear a number of times >= {}.'.format(len(kb_ex), int(100*len(kb_ex)/len(self.stat['train']['kb_entities'])), th))
        plt.hist(list(self.stat['train']['kb_entities'].values()), bins='auto')
        plt.show()
        ent_ex = [i for i in self.stat['train']['entities'].items() if i[1]>=th]
        print('> {} text entities ({}%) appear a number of times >= {}.'.format(len(ent_ex), int(100*len(ent_ex)/len(self.stat['train']['entities'])), th))
        return ent_ex, kb_ex

    def support(self, th=1):
        print('TEST SET SUPPORT')
        kb_supp_tot = {k:data['train']['kb_entities'][k] for k in self.stat['test']['kb_entities'].keys()}
        kb_supp = {k:data['train']['kb_entities'][k] for k in self.stat['test']['kb_entities'].keys() if data['train']['kb_entities'][k] >= th}
        kb_supp_idx = numpy.mean(list(kb_supp_tot.values()))
        print('> {} KB entities ({}%) in the test set have support >= {} in the train set.\n>> Average support: {:.2f}'.format(len(kb_supp), int(100*len(kb_supp)/len(data['test']['kb_entities'])), th, kb_supp_idx))
        ent_supp = {k:data['train']['entities'][k] for k in self.stat['test']['entities'].keys() if data['train']['entities'][k] >= th}
        ent_supp_tot = {k:data['train']['entities'][k] for k in self.stat['test']['entities'].keys()}
        ent_supp_idx = numpy.mean(list(ent_supp_tot.values()))
        print('> {} text entities ({}%) in the test set have support >= {} in the train set.\n>> Average support: {:.2f}'.format(len(ent_supp), int(100*len(ent_supp)/len(data['test']['entities'])), th, ent_supp_idx))
        kb_supp_idx /= len(kb_supp_tot)
        ent_supp_idx /= len(ent_supp_tot)
        return ent_supp_idx, kb_supp_idx

    def ambiguity(self):
        print('KB TO TEXT MAPPING')
        kb2txt = { k:v for k,v in self.stat['common']['kb2txt'].items() if len(v) > 1 }
        print('> {} KB entities ({}%) have multiple text representations.\n>> Average ambiguity: {:.2f}'.format(len(kb2txt), int(100*len(kb2txt)/len(self.stat['common']['kb2txt'])), numpy.mean(list(map(len,self.stat['common']['kb2txt'].values())))))
        print('TEXT TO KB MAPPING')
        txt2kb = { k:v for k,v in self.stat['common']['txt2kb'].items() if len(v) > 1 }
        print('> {} text entities ({}%) are ambiguous and refer to different concepts.\n>> Average ambiguity: {:.2f}'.format(len(txt2kb), int(100*len(txt2kb)/len(self.stat['common']['txt2kb'])), numpy.mean(list(map(len,self.stat['common']['txt2kb'].values())))))
    
    def gen(self):
        if self.stat == None:
            self.stat = self.scan()
        self.examples(3)
        self.support(3)
        self.ambiguity()

    def score_vs_support(self, scores):
        d = [ [k, [v, scores[k]]] for k,v in self.stat['train']['entity_types'].items()]
        d = sorted(d, key=lambda x: x[1][0])
        xy = numpy.array([ i[1] for i in d ])
        plt.scatter(xy[:,0], xy[:,1])
        plt.plot(xy[:,0], xy[:,1])
        for i in d:
            plt.annotate(i[0], i[1])
        plt.show()
