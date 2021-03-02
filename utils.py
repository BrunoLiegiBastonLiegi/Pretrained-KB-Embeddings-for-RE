import random, torch
import numpy as np
from abc import ABC, abstractmethod



# ------------------------ Preprocessing ---------------------------------------------


def split_sets(data, validation=0.1, test=0.2):
    """
    Splits the dataset in train, validation and test datasets.

    Parameters:
    data (list): list of tuples of the form (train_example, label)
    validation (float): portion of data destinated to the validation set
    test (float): portion of data destinated to the test set

    Returns:
    data (list): the final train set
    val (list): the validation set
    test (list): the test set
    """
    
    val_dim = int(validation*len(data))
    test_dim = int(test*len(data))

    random.shuffle(data)

    val = [ data.pop(random.randint(0, len(data)-1)) for i in range(val_dim) ]
    if test_dim != 0:
        test = [ data.pop(random.randint(0, len(data)-1)) for i in range(test_dim) ]

    if test_dim == 0:
        return data, val
    else:
        return data, val, test




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

    def __init__(self, train_data, test_data, model, tokenizer, optim, loss_f, device):     
        self.model = model
        self.tokenizer = tokenizer
        self.optim = optim
        self.loss_f = loss_f
        self.device = device     
        self.train_set = train_data
        self.test_set = test_data
        """
    def split_sets(self, validation, test):
        data = self.data      
        val_dim = int(validation*len(data))
        test_dim = int(test*len(data))       
        random.shuffle(data)
        val = [ data.pop(random.randint(0, len(data)-1)) for i in range(val_dim) ]
        test = [ data.pop(random.randint(0, len(data)-1)) for i in range(test_dim) ]
        return data, val, test
        """
    def train(self, epochs):

        k = 0  # counter for bert layers unfreezing
        one_3rd = int(len(self.train_set) / 3) # after 1/3 of the data we unfreeze a layer
        l = 0. # RE loss weight, gradually increased to 1
        
        for epoch in range(epochs):

            running_loss = 0.0
            ner_running_loss = 0.0
            ned_running_loss = 0.0
            re_running_loss = 0.0
            # shuffle training set
            random.shuffle(self.train_set)

            for i in range(len(self.train_set)):

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

                # forward + backward + optimize
                ner_output, ned_output, re_output= self.model(inputs)
                ner_loss = self.loss_f(ner_output, ner_target)
                ned_loss = self.NED_loss(ned_output, ned_target) if ned_output != None else torch.tensor(0., device=self.device)
                re_loss = self.RE_loss(re_output, re_target) if re_output != None else torch.tensor(0., device=self.device)
                if epoch == 0:
                    l = i / len(self.train_set)
                loss = ner_loss + l * (re_loss + ned_loss)
                #loss = ner_loss + re_loss
                loss.backward()
                self.optim.step()

                # print statistics
                ner_running_loss += ner_loss.item()
                ned_running_loss += ned_loss.item()
                re_running_loss += re_loss.item()
                running_loss += loss.item()

                if i % 500 == 499:    # print every 500 sentences
                    print('[%d, %5d] Total loss: %.3f, NER: %.3f, NED: %.3f, RE: %.3f' %
                          (epoch + 1, i + 1, running_loss / 500, ner_running_loss / 500, ned_running_loss / 500, re_running_loss / 500))
                    running_loss = 0.0
                    ner_running_loss = 0.
                    ned_running_loss = 0.
                    re_running_loss = 0.
                    
            test_loss = self.test_loss()
            print('> Test Loss\n Total: %.3f, NER: %.3f, NED: %.3f, RE: %.3f' %
                  (test_loss[0], test_loss[1], test_loss[2], test_loss[3]), '\n')
            
        # save the model
        print('> Save model to PATH (leave blank for not saving): ')
        PATH = input()
        if PATH != '':
            torch.save(self.model.state_dict(), PATH)
            print('> Model saved to ', PATH)
        else:
            print('> Model not saved.')

    def test_loss(self):
        with torch.no_grad():
            loss = 0.
            test_ner_loss = torch.tensor(0., device=self.device)
            test_ned_loss = torch.tensor(0., device=self.device)
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
                ned_loss = self.NED_loss(ned_output, ned_target) if ned_output != None else torch.tensor(0., device=self.device)
                re_loss = self.RE_loss(re_output, re_target) if re_output != None else torch.tensor(0., device=self.device)
                loss += ner_loss.item() + ned_loss.item() + re_loss.item()
                test_ner_loss += ner_loss.item()
                test_ned_loss += ned_loss.item()
                test_re_loss += re_loss.item()

            return (loss / len(self.test_set), test_ner_loss / len(self.test_set), test_ned_loss / len(self.test_set), test_re_loss / len(self.test_set))

    def RE_loss(self, re_out, groundtruth):
        loss = 0.
        fake_target = []
        for i in range(len(re_out[0])):
            tg = None
            for j in groundtruth:
                if re_out[0][i][0] == j[0] and re_out[0][i][1] == j[1]:
                    tg = j[2] 
            if tg != None:
                fake_target.append(tg)
            else:
                fake_target.append(torch.tensor(0, device=self.device)) # 0 for NO_RELATION

        return self.loss_f(re_out[1], torch.stack(fake_target, dim=0))

    def NED_loss(self, ned_out, groundtruth):
        loss = torch.tensor(0., device=self.device)
        mse = torch.nn.MSELoss(reduction='sum')
        ned_dim = self.model.ned_dim
        gt = dict(zip(groundtruth[:,0].int().tolist(), groundtruth[:,1:]))
        #print(gt)
        ned = dict(zip(torch.flatten(ned_out[0]).tolist(), ned_out[1]))
        #print(ned)
        """"
        gt = dict(zip(
            [ str(i.tolist()) for i in groundtruth[:, :-ned_dim] ],
            groundtruth[:, -ned_dim:] ))
        ned = dict(zip(
            [ str(i.tolist()) for i in ned_out[:, :-ned_dim] ],
            ned_out[:, -ned_dim:] ))
        """
        fake_target = torch.zeros(ned_dim, device=self.device)
        
        # maybe it would be better to take for good also entity predictions that contain all
        # the groundtruth tokens + something else, for example:
        # GT: [1127, 897]  PRED: [6725, 1127, 897]
        # they are not the same, but the prediction was mostly correct
        # the easiest way would be to just stick with the previous approach of considering
        # the last token only
        for k, v in gt.items():
            try:
                loss += torch.sqrt(mse(ned.pop(k), v)) # pop cause we get rid of the already calculated {entity:embedding} pair
            except:
                loss += torch.sqrt(mse(fake_target, v))
        for v in ned.values():
            loss += torch.sqrt(mse(v, fake_target))
  
        return loss
            
