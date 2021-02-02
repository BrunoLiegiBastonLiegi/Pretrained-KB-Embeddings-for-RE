import random, torch
from abc import ABC, abstractmethod



# ------------------------ Preprocessing ---------------------------------------------


def split_sets(data, validation=0.2, test=0.1):
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



# ------------------------ Training -------------------------------------------------------




class Trainer(object):

    def __init__(self, data, model, tokenizer, optim, loss_f, device, validation=0., test=0.):     
        self.data = data
        self.model = model
        self.tokenizer = tokenizer
        self.optim = optim
        self.loss_f = loss_f
        self.device = device     
        if validation != 0. or test != 0.:
            self.train_set, self.val_set, self.test_set = self.split_sets(validation, test)
        else:
            self.train_set = self.data

    def split_sets(self, validation, test):
        data = self.data      
        val_dim = int(validation*len(data))
        test_dim = int(test*len(data))       
        random.shuffle(data)
        val = [ data.pop(random.randint(0, len(data)-1)) for i in range(val_dim) ]
        test = [ data.pop(random.randint(0, len(data)-1)) for i in range(test_dim) ]
        return data, val, test

    def train(self, epochs):

        k = 0  # counter for bert layers unfreezing
        one_3rd = int(len(self.train_set) / 3)
        
        for epoch in range(epochs):

            running_loss = 0.0

            # shuffle training set
            random.shuffle(self.train_set)

            for i in range(len(self.train_set)):

                if k < 4:
                    if i == one_3rd:
                        k += 1
                        self.model.unfreeze_bert_layer(k) # gradually unfreeze the last  layers
                    elif i == 2*one_3rd:
                        k += 1
                        self.model.unfreeze_bert_layer(k) # gradually unfreeze the last  layers
                    elif i == len(self.train_set):
                        k += 1
                        self.model.unfreeze_bert_layer(k) # gradually unfreeze the last  layers

                inputs = self.tokenizer(self.train_set[i][0], return_tensors="pt")
                target = self.train_set[i][1]

                # move inputs and labels to device
                if self.device != torch.device("cpu"):
                    inputs = inputs.to(self.device)
                    target = target.to(self.device)
    
                # zero the parameter gradients
                self.optim.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.loss_f(outputs, target)
                loss.backward()
                self.optim.step()

                # print statistics
                running_loss += loss.item()

                if i % 500 == 499:    # print every 500 sentences
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 500))
                    running_loss = 0.0

            try:
                val_loss = self.validation_loss().item()
                print('> Validation Loss: %3f' % val_loss, '\n')
            except:
                pass
            
        # save the model
        print('> Save model to PATH (leave blank for not saving): ')
        PATH = input()
        if PATH != '':
            torch.save(self.model.state_dict(), PATH)
        else:
            print('> Model not saved.')

    def validation_loss(self):
        with torch.no_grad():
            loss = 0.
            for i in self.val_set:
            
                inputs = self.tokenizer(i[0], return_tensors="pt")
                target = i[1]

                # move inputs and labels to device
                if self.device != torch.device("cpu"):
                    inputs = inputs.to(self.device)
                    target = target.to(self.device)

                outputs = self.model(inputs)
                loss += self.loss_f(outputs, target)

            return loss / len(self.val_set)






# ------------------------ F1 score -------------------------------------


class F1(object):

    def __init__(self, prediction, groundtruth):
        self.pred = prediction
        self.gt = groundtruth

    def __call__(self, prediction, groundtruth):

        TP, FP, FN = 0, 0, 0

        for i in range(len(groundtruth)):
            print(i,'-->',self.subtract(prediction,groundtruth))
            #j = 0
            #while groundtruth[i][j] == 'O' and groundtruth[i][j] == prediction[i][j]:
             #   j += 1
            #if groundtruth[i][j] != 'O':
             #   while groundtruth[i][j] == prediction[i][j]:
             #       j +=1

    def subtract(self, prediction, groundtruth):
        sub = []
        for i in range(len(groundtruth)):
            if groundtruth[i] == prediction[i]:
                if groundtruth[i] != 'O':
                    sub.append(1)
                else:
                    sub.append(0)
            else:
                sub.append(-1)
        return sub
