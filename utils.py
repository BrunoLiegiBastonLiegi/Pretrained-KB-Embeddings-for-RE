import random, torch, pickle
import numpy as np
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader


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

    def __init__(self, train_data, test_data, model, optim, device, save=True, wNED=1, batchsize=32):     
        self.model = model
        self.optim = optim
        self.device = device     
        self.train_set = train_data
        self.test_set = test_data
        self.save = save
        self.wNED = wNED
        self.crossentropy = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss(reduction='sum')
        self.batchsize = batchsize

    def train(self, epochs):

        # BERT layers unfreezing
        k = 0  # counter for bert layers unfreezing
        one_3rd = int(len(self.train_set) / (3*self.batchsize)) # after 1/3 of the data we unfreeze a layer
        # Loss weights
        l = 0. # RE loss weight, gradually increased to 1
        
        for epoch in range(epochs):

            running_loss = 0.0
            ner_running_loss = 0.0
            ned_running_loss1 = 0.0
            ned_running_loss2 = 0.0
            re_running_loss = 0.0
            # set model in train mode
            self.model.train()
            train_loader = DataLoader(self.train_set,
                                      batch_size=self.batchsize,
                                      shuffle=True,
                                      collate_fn=self.train_set.collate_fn)
            print_step = int(len(train_loader) / 6)
            
            for i, batch in enumerate(train_loader):

                if k < 4:
                    if epoch == 0:
                        if i >= one_3rd and k == 0:
                            self.model.unfreeze_bert_layer(k) # gradually unfreeze the last layers
                            k += 1
                        elif i >= 2*one_3rd and k == 1:
                            self.model.unfreeze_bert_layer(k) # gradually unfreeze the last layers
                            k += 1
                    elif epoch == 1:
                        if k == 2:
                            self.model.unfreeze_bert_layer(k) # gradually unfreeze the last layers
                            k += 1
                        if i >= one_3rd and k == 3:
                            self.model.unfreeze_bert_layer(k) # gradually unfreeze the last layers
                            k += 1

                # zero the parameter gradients
                self.optim.zero_grad()
                ner_loss, ned_loss1, ned_loss2, re_loss = self.step(batch)

                if epoch == 0:
                    l = i / len(train_loader)
                else:
                    l = 1
                loss = ner_loss + l * re_loss + self.wNED*(l * (100*ned_loss1 + ned_loss2)) # wNED is used for discovering the benefit of NED
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

                if i % print_step == print_step - 1:    # print every print_step sentences
                    print('[%d, %5d] Total loss: %.3f, NER: %.3f, NED1: %.3f, NED2: %.3f, RE: %.3f' %
                          (epoch + 1, i*self.batchsize + 1, running_loss / print_step, ner_running_loss / print_step, ned_running_loss1 / print_step, ned_running_loss2 / print_step, re_running_loss / print_step))
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

    def step(self, batch):
        inputs = batch['sent']
        ner_target = batch['ner']
        ned_target = batch['ned']
        re_target = batch['re']

        # move inputs and labels to device
        if self.device != torch.device("cpu"):
            inputs = inputs.to(self.device)
            ner_target = ner_target.to(self.device)
            ned_target = [ t.to(self.device) for t in ned_target ]
            re_target = [ t.to(self.device) for t in re_target ]

        # forward 
        ner_out, ned_output, re_output = self.model(inputs)
        # losses
        print(ner_out.shape)
        print(torch.transpose(ner_out, 1, 2).shape)
        print(ner_target.shape)
        a
        ner_loss = self.crossentropy(
            torch.transpose(ner_out, 1, 2),
            ner_target
        )
        ned_loss1, ned_loss2 = self.NED_loss(ned_output, ned_target) if ned_output != None else (torch.tensor(2.3, device=self.device), torch.tensor(2.3, device=self.device))
        re_loss = self.RE_loss(re_output, re_target) if re_output != None else torch.tensor(2.3, device=self.device)
        return ner_loss, ned_loss1, ned_loss2, re_loss
        
    def test_loss(self):
        test_loader = DataLoader(self.test_set,
                                 batch_size=self.batchsize,
                                 shuffle=True,
                                 collate_fn=self.test_set.collate_fn)
        # set model in eval mode
        self.model.eval()
        with torch.no_grad():
            loss = 0.
            test_ner_loss = torch.tensor(0., device=self.device)
            test_ned_loss1 = torch.tensor(0., device=self.device)
            test_ned_loss2 = torch.tensor(0., device=self.device)
            test_re_loss = torch.tensor(0., device=self.device)
            
            for batch in test_loader:
                ner_loss, ned_loss1, ned_loss2, re_loss = self.step(batch)
                loss += ner_loss + re_loss + (100*ned_loss1 + ned_loss2) 
                test_ner_loss += ner_loss.item()
                test_ned_loss1 += ned_loss1.item()
                test_ned_loss2 += ned_loss2.item()
                test_re_loss += re_loss.item()
        # return to train mode
        self.model.train()

        return (loss / len(test_loader), test_ner_loss / len(test_loader), test_ned_loss1 / len(test_loader), test_ned_loss2 / len(test_loader), test_re_loss / len(test_loader))

    def RE_loss(self, re_out, groundtruth):
        loss = 0.
        for i in range(len(groundtruth)):
            g = dict(zip(
                map( tuple, groundtruth[i][:,:2].tolist() ),
                groundtruth[i][:,2]
            ))
            r = dict(zip(
                map( tuple, re_out[0][i].tolist() ),
                re_out[1][i]
            ))
            re_pred, re_target = [], []
            for k in g.keys() & r.keys():
                re_pred.append(r.pop(k))
                re_target.append(g[k])
            if self.model.training:
                for v in r.values():
                    re_pred.append(v)
                    re_target.append(torch.tensor(0, dtype=torch.int, device=self.device)) # 0 for no relation
            if len(re_pred) > 0:
                loss += self.crossentropy(torch.vstack(re_pred), torch.hstack(re_target).long())
                
        if loss != 0:
            return loss / len(groundtruth)
        else:
            return torch.tensor(2.3, device=self.device)
        
    def NED_loss(self, ned_out, groundtruth):
        loss1, loss2 = torch.tensor(0., device=self.device), torch.tensor(0., device=self.device)
        dim = self.model.ned_dim
        for i in range(len(groundtruth)):
            g = dict(zip(
                groundtruth[i][:,0].int().tolist(),
                groundtruth[i][:,1:]
            ))
            n1 = dict(zip(
                torch.flatten(ned_out[0][i]).tolist(),
                ned_out[1][i]
            ))
            n2 = dict(zip(
                torch.flatten(ned_out[0][i]).tolist(),
                ned_out[2][i]
            ))
            n2_scores, n2_targets = [], []
            for k in g.keys() & n1.keys():
                loss1 += torch.sqrt(self.mse(n1.pop(k), g[k]))
                tmp = n2.pop(k)
                candidates = tmp[:,1:]
                if g[k] in candidates:
                    ind = ((candidates-g[k]).sum(-1)==0).nonzero()
                    n2_targets.append(torch.flatten(ind)[0])
                    #n2_targets.append(ind.view(1))
                    n2_scores.append(tmp[:,0])
            if len(n2_scores) > 0 :
                loss2 += self.crossentropy(torch.vstack(n2_scores), torch.hstack(n2_targets)) 
            else:
                loss2 += torch.tensor(2.3, device=self.device)
            if self.model.training:
                loss1 += sum(map(self.mse, n1.values(), torch.zeros(len(n1), dim, device=self.device)))

        if loss1 != 0: 
            return loss1 / len(groundtruth), loss2 / len(groundtruth)
        else:
            return torch.tensor(2.3, device=self.device), torch.tensor(1., device=self.device)
            


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
