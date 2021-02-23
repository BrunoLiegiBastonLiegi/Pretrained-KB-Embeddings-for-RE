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
        #if validation != 0. or test != 0.:
            #self.train_set, self.val_set, self.test_set = self.split_sets(validation, test)
        #else:
            #self.train_set = self.data
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
        l = 0 # RE loss weight, gradually increased to 1
        
        for epoch in range(epochs):

            running_loss = 0.0
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
                re_target = self.train_set[i][2]

                # move inputs and labels to device
                if self.device != torch.device("cpu"):
                    inputs = inputs.to(self.device)
                    ner_target = ner_target.to(self.device)
                    re_target = re_target.to(self.device)
    
                # zero the parameter gradients
                self.optim.zero_grad()

                # forward + backward + optimize
                ner_output, re_output = self.model(inputs)
                ner_loss = self.loss_f(ner_output, ner_target)
                re_loss = self.RE_loss(re_output, re_target) if re_output != None else 0
                if epoch == 0:
                    l = i / len(self.train_set)
                loss = ner_loss + l * re_loss
                #loss = ner_loss + re_loss
                loss.backward()
                self.optim.step()

                # print statistics
                running_loss += loss.item()

                if i % 500 == 499:    # print every 500 sentences
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 500))
                    running_loss = 0.0

            try:
                test_loss = self.test_loss()
                print('> Test Loss: %3f' % test_loss, '\n')
            except:
                pass
            
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
            for i in self.test_set:
            
                inputs = self.tokenizer(i[0], return_tensors="pt")
                ner_target = i[1]
                re_target = i[2]

                # move inputs and labels to device
                if self.device != torch.device("cpu"):
                    inputs = inputs.to(self.device)
                    ner_target = ner_target.to(self.device)
                    re_target = re_target.to(self.device)

                ner_output, re_output = self.model(inputs)
                ner_loss = self.loss_f(ner_output, ner_target)
                re_loss = self.RE_loss(re_output, re_target) if re_output != None else 0
                loss += ner_loss.item() + re_loss

            return loss / len(self.test_set)

    def RE_loss(self, re_out, groundtruth):
        loss = 0.
        fake_target = []
        for i in range(len(re_out[1])):
            tg = None
            for j in groundtruth:
                if re_out[1][i][0] == j[0] and re_out[1][i][1] == j[1]:
                    tg = j[2] 
            if tg != None:
                fake_target.append(tg)
            else:
                fake_target.append(torch.tensor(0, device=self.device)) # 0 for NO_RELATION

        return self.loss_f(re_out[0], torch.stack(fake_target, dim=0)) 

