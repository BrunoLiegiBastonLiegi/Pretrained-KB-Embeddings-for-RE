import random, torch, pickle
import numpy as np
from abc import ABC, abstractmethod



# ------------------------ NER tagging schemes ----------------------------------------



class Scheme(ABC):
    
    @abstractmethod
    def __init__(self):
        pass

    @property
    def space_dim(self):
        return len(self.tag2index)
        
    def to_tensor(self, *tag, index=False):
        if index:
            return torch.tensor([ self.tag2index[i] for i in tag ]) # return only index of the class
        else:
            t = torch.zeros((len(tag),self.space_dim))
            j = 0
            for i in tag:
                t[j][self.tag2index[i]] = 1
                j += 1
            return t                                               # return 1-hot encoding tensor
        

    def to_tag(self, tensor, index=False):
        if index:
            return int(torch.argmax(tensor))
        else:
            return self.index2tag[int(torch.argmax(tensor))]

    @abstractmethod
    def transition_M(self):
        pass

    
class BIOES(Scheme):
    
    def __init__(self, entity_types):
        self.e_types = entity_types
        self.tag2index = {}
        i = 0
        for e in self.e_types:
            self.tag2index['B-' + e] = i
            i +=1
            self.tag2index['I-' + e] = i
            i +=1
            self.tag2index['E-' + e] = i
            i +=1
            self.tag2index['S-' + e] = i
            i +=1
        self.tag2index['O'] = i
        self.index2tag = {v: k for k, v in self.tag2index.items()}

    # WARNING: to be tested!
    def transition_M(self):
        p = 1./( 8*len(self.e_types) + 4*len(self.e_types) + 1)
        self.intra_transition = torch.tensor([ [0,p,p,0],
                                               [0,p,p,0],
                                               [p,0,0,p],
                                               [p,0,0,p] ])
        
        self.inter_transition = torch.tensor([ [0,0,0,0],
                                               [0,0,0,0],
                                               [p,0,0,p],
                                               [p,0,0,p] ])

        # should I consider transitions from initial tag <s> as well?
        
        boundary = torch.tensor([p,0,0,p])
        for j in range(len(self.e_types) - 1):
            boundary = torch.cat((boundary, self.intra_transition[2]))
        
        for i in range(len(self.e_types)):
            row = self.intra_transition if i == 0 else self.inter_transition
            for j in range(len(self.e_types) - 1):
                row = torch.hstack((row,self.intra_transition)) if i == j else torch.hstack((tmp,self.inter_transition))
            transition_M = row if i == 0 else torch.vstack((transition_M,row))

        transition_M = torch.vstack(transition_M, boundary)
        transition_M = torch.hstack(transition_M, torch.transpose(boundary.view(-1,1)))
            
        return transition_M



# ------------------------ Training -------------------------------------------------------




class Trainer(object):

    def __init__(self, train_data, test_data, model, tokenizer, optim, loss_f, device, save=True, wNED=1, batchsize=32):     
        self.model = model
        self.tokenizer = tokenizer
        self.optim = optim
        self.loss_f = loss_f
        self.device = device     
        self.train_set = train_data
        self.test_set = test_data
        self.save = save
        self.wNED = wNED
        self.mse = torch.nn.MSELoss(reduction='sum')

    def train(self, epochs):

        # BERT layers unfreezing
        k = 0  # counter for bert layers unfreezing
        one_3rd = int(len(self.train_set) / 3) # after 1/3 of the data we unfreeze a layer
        # Losses weights
        l_re = 0. # RE loss weight, gradually increased to 1
        l_ned = 0.
        # Losses plots
        self.loss_plots = {
            'train': {'NER':[], 'NED':[], 'RE':[]},
            'test': {'NER':[], 'NED':[], 'RE':[]}
        }
        
        for epoch in range(epochs):

            running_loss = 0.0
            ner_running_loss = 0.0
            ned_running_loss1 = 0.0
            ned_running_loss2 = 0.0
            re_running_loss = 0.0
            # shuffle training set
            random.shuffle(self.train_set)
            # set model in train mode
            self.model.train()

            for i in range(len(self.train_set)):
                #print(list(self.model.ned_lin0.parameters()))

                if k < 4:
                    if i == one_3rd:
                        self.model.unfreeze_bert_layer(k) # gradually unfreeze the last layers
                        k += 1
                    elif i == 2*one_3rd:
                        self.model.unfreeze_bert_layer(k) # gradually unfreeze the last layers
                        k += 1
                    elif i == len(self.train_set):
                        self.model.unfreeze_bert_layer(k) # gradually unfreeze the last layers
                        k += 1

                inputs = self.tokenizer(self.train_set[i][0], return_tensors="pt")
                ner_target = self.train_set[i][1]
                ned_target = self.train_set[i][2]
                re_target = self.train_set[i][3]

                # move inputs and labels to device
                if self.device != torch.device("cpu"):
                    inputs = inputs.to(self.device)
                    ner_target = ner_target.to(self.device)
                    ned_target = ned_target.to(self.device)
                    re_target = re_target.to(self.device)

                # zero the parameter gradients
                self.optim.zero_grad()

                # forward 
                ner_output, ned_output, re_output= self.model(inputs)
                # losses
                ner_loss = self.loss_f(ner_output, ner_target)
                self.loss_plots['train']['NER'].append(ner_loss)
                ned_loss1, ned_loss2 = self.NED_loss(ned_output, ned_target) if ned_output != None else (torch.tensor(1., device=self.device), torch.tensor(1., device=self.device))
                #ned_loss = torch.tensor(1., device=self.device)
                self.loss_plots['train']['NED'].append(ned_loss1)
                re_loss = self.RE_loss(re_output, re_target) if re_output != None else torch.tensor(1., device=self.device)
                self.loss_plots['train']['RE'].append(re_loss)
                if epoch == 0:
                    l_re = i / len(self.train_set)
                    l_ned = min(3*l_re, 1)
                #loss = ner_loss + l_re * re_loss + self.wNED*(l_ned * 100*ned_loss) # wNED is used for discovering the benefit of NED
                loss = ner_loss + l_re * re_loss + self.wNED*(l_re * (100*ned_loss1 + ned_loss2)) # wNED is used for discovering the benefit of NED
                # backprop
                loss.backward()
                # optimize
                self.optim.step()

                # print statistics
                ner_running_loss += ner_loss.item()
                ned_running_loss1 += ned_loss1.item()
                ned_running_loss2 += ned_loss2.item()
                re_running_loss += re_loss.item()
                running_loss += loss.item()

                if i % 500 == 499:    # print every 500 sentences
                    print('[%d, %5d] Total loss: %.3f, NER: %.3f, NED1: %.3f, NED2: %.3f, RE: %.3f' %
                          (epoch + 1, i + 1, running_loss / 500, ner_running_loss / 500, ned_running_loss1 / 500, ned_running_loss2 / 500, re_running_loss / 500))
                    running_loss = 0.0
                    ner_running_loss = 0.
                    ned_running_loss1 = 0.
                    ned_running_loss2 = 0.
                    re_running_loss = 0.
                    
            test_loss = self.test_loss()
            print('> Test Loss\n Total: %.3f, NER: %.3f, NED1: %.3f, NED2: %.3f, RE: %.3f' %
                  (test_loss[0], test_loss[1], test_loss[2], test_loss[3], test_loss[4]), '\n')

        if self.save:
            # save the model
            print('> Save model to PATH (leave blank for not saving): ')
            PATH = input()
            if PATH != '':
                torch.save(self.model.state_dict(), PATH)
                print('> Model saved to ', PATH)
            else:
                print('> Model not saved.')

        self.model.eval()
        return self.loss_plots

    def test_loss(self):
        # set model in eval mode
        self.model.eval()
        with torch.no_grad():
            loss = 0.
            test_ner_loss = torch.tensor(0., device=self.device)
            test_ned_loss1 = torch.tensor(0., device=self.device)
            test_ned_loss2 = torch.tensor(0., device=self.device)
            test_re_loss = torch.tensor(0., device=self.device)
            
            for i in self.test_set:
                inputs = self.tokenizer(i[0], return_tensors="pt")
                ner_target = i[1]
                ned_target = i[2]
                re_target = i[3]

                # move inputs and labels to device
                if self.device != torch.device("cpu"):
                    inputs = inputs.to(self.device)
                    ner_target = ner_target.to(self.device)
                    ned_target = ned_target.to(self.device)
                    re_target = re_target.to(self.device)

                ner_output, ned_output, re_output = self.model(inputs)
                ner_loss = self.loss_f(ner_output, ner_target)
                self.loss_plots['test']['NER'].append(ner_loss)
                ned_loss1, ned_loss2 = self.NED_loss(ned_output, ned_target) if ned_output != None else (torch.tensor(1., device=self.device), torch.tensor(1., device=self.device)) 
                self.loss_plots['test']['NED'].append(ned_loss1)
                re_loss = self.RE_loss(re_output, re_target) if re_output != None else torch.tensor(1., device=self.device)
                self.loss_plots['test']['RE'].append(re_loss)
                loss += ner_loss.item() + ned_loss1.item() + ned_loss2.item() + re_loss.item()
                test_ner_loss += ner_loss.item()
                test_ned_loss1 += ned_loss1.item()
                test_ned_loss2 += ned_loss2.item()
                test_re_loss += re_loss.item()

            return (loss / len(self.test_set), test_ner_loss / len(self.test_set), test_ned_loss1 / len(self.test_set), test_ned_loss2 / len(self.test_set), test_re_loss / len(self.test_set))

    def RE_loss(self, re_out, groundtruth):
        loss = 0.
        pred = []
        target = []
        
        gt = dict(zip(map(tuple, groundtruth[:,:2].tolist()), groundtruth[:,2]))
        re = dict(zip(map(tuple, re_out[0].tolist()), re_out[1]))

        for k,v in gt.items():
            try:
                p = re.pop(k)
                target.append(v)
                pred.append(p)
            except:
                pass
        if self.model.training:
            for v in re.values():
                pred.append(v)
                target.append(torch.tensor(0, device=self.device))
 
        if len(pred) > 1:
            return self.loss_f(torch.vstack(pred), torch.hstack(target))
        else:
            return torch.tensor(1., device=self.device)
        
    def NED_loss(self, ned_out, groundtruth):
        loss1, loss2 = torch.tensor(0., device=self.device), torch.tensor(0., device=self.device)
        ned_dim = self.model.ned_dim
        gt = dict(zip(groundtruth[:,0].int().tolist(), groundtruth[:,1:]))
        ned_2 = dict(zip(torch.flatten(ned_out[0]).tolist(), ned_out[1][1]))
        ned_1 = dict(zip(torch.flatten(ned_out[0]).tolist(), ned_out[1][0]))

        fake_target = torch.zeros(ned_dim, device=self.device)

        ned_2_scores = []
        ned_2_targets = []
        for k, v in gt.items():
            try:
                candidates = ned_2[k][:,1:]
                if v in candidates:
                    ned_2_scores.append(ned_2[k][:,0])
                    ind = ((ned_2[k][:,1:]-v).sum(-1)==0).nonzero()
                    if len(ind) == 1:
                        ned_2_targets.append(ind.view(1))
                    else:
                        ned_2_targets.append(ind[0].view(1)) # it happened to have two equal neighbors, strange...
                loss1 += torch.sqrt(self.mse(ned_1.pop(k), v)) # pop cause we get rid of the already calculated {entity:embedding} pair
            except:
                #loss += torch.sqrt(mse(fake_target, v))
                pass
        if self.model.training:
            for v in ned_1.values():
                loss1 += torch.sqrt(self.mse(v, fake_target))
                
        if len(ned_2_scores) > 0 :
            loss2 = self.loss_f(torch.vstack(ned_2_scores), torch.hstack(ned_2_targets))
        else:
            loss2 = torch.tensor(2.3, device=self.device)
                
        if loss1 != 0: 
            return loss1, loss2
        else:
            return torch.tensor(1., device=self.device), torch.tensor(1., device=self.device)
            


class IEData(torch.utils.data.Dataset):

    def __init__(self, sentences, ner_labels, re_labels, ned_labels=None, tokenizer=None, ner_scheme=None, rel2index=None, save_to=None):
        self.tokenizer = tokenizer
        self.scheme = ner_scheme
        self.rel2index = rel2index
        self.samples = []
        self.pad = self.tokenizer('[PAD]', add_special_tokens=False)['input_ids'][0]
        self.sep = self.tokenizer('[SEP]', add_special_tokens=False)['input_ids'][0]
        if tokenizer != None:
            print('> Preprocessing labels.')
            assert ner_scheme != None, 'Missing NER scheme for preprocessing.'
            for s, ner, re in zip(sentences, ner_labels, re_labels):
                self.samples.append(self.generate_labels(s, ner, re))
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
                pickle.dump(self.samples, f)

    def generate_labels(self, s, ner, re):
        tks = self.tokenizer([s] + list(ner.keys()), add_special_tokens=False)['input_ids']
        for e, t in zip(ner.values(), tks[1:]):
            e['span'] = self.find_span(tks[0], t)
        lab = {
            'sent': self.tokenizer(s, return_tensors='pt')['input_ids'],
            'ner': self.tag_sentence(tks[0], ner.values()),
            'ned': torch.vstack([
                torch.hstack((
                    torch.tensor(e['span'][1]),
                    torch.mean(e['embedding'], dim=0) # need mean for multi-concept entities
                ))
                for e in ner.values()
            ]),
            're': torch.vstack([
                torch.tensor([
                    ner[k[0]]['span'][1],
                    ner[k[1]]['span'][1],
                    self.rel2index[r['type']]
                ])
                for k, r in re.items()
            ])
            }
        return lab 

    def find_span(self, sent, ent):
        """
        Find the span of the entity in the tokenization scheme provided by the tokenizer.
        We consider only the case of the same entity occuring just once for each sentence.
        """
        for i in range(len(sent)):
            if sent[i] == ent[0] and sent[i:i+len(ent)] == ent: 
                match = (i, i+len(ent))
        return match

    def tag_sentence(self, sent, ents):
        tags = torch.tensor([self.scheme.to_tensor('O', index=True) for i in range(len(sent))])
        for e in ents:
            t = e['type']
            span = e['span']
            if span[1]-span[0] == 1:
                tags[span[0]] = self.scheme.to_tensor('S-' + t, index=True)
            else:
                tags[span[0]:span[1]] = self.scheme.to_tensor(*(['B-' + t] + [('I-' + t) for j in range((span[1]-span[0])-2)] + ['E-' + t]), index=True)
        return tags.view(1,-1)

    def collate_fn(self, batch):
        """
        Function to vertically stack the batches needed by the torch.Dataloader class
        """
        tmp = {'sent':[], 'ner':[], 'ned':[], 're':[]} # we need to add padding in order to vstack the sents.
        max_len = 0
        for item in batch:
            max_len = max(max_len, item['sent'][:,:-1].shape[1]) # -1 for discarding [SEP]
            tmp['sent'].append(item['sent'][:,:-1])
            tmp['ner'].append(item['ner'])
            tmp['ned'].append(item['ned'])
            tmp['re'].append(item['re'])
        tmp['sent'] = torch.vstack(list(map(
            lambda x: torch.hstack((x, self.pad*torch.ones(1, max_len - x.shape[1]).int())),
            tmp['sent']
        )))
        # add the [SEP] at the end
        tmp['sent'] = torch.hstack((tmp['sent'], self.sep*torch.ones(tmp['sent'].shape[0],1).int()))
        O = self.scheme.to_tensor('O', index=True)
        # -1 because max_len counts also the [CLS]
        tmp['ner'] = torch.vstack(list(map(
            lambda x: torch.hstack((x, O*torch.ones(1, max_len - 1 - x.shape[1]).int())),
            tmp['ner']
        )))
        return tmp
            
    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)
