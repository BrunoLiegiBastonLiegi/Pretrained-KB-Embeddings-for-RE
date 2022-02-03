from abc import ABC, abstractmethod
import torch



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
