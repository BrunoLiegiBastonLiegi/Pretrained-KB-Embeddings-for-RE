import random
import torch
import math

from transformers import AutoTokenizer, AutoModel
from itertools import product


class Pipeline(torch.nn.Module):

    def __init__(self, bert, ner_dim, ner_scheme, re_dim):
        super().__init__()

        self.scheme = ner_scheme
        self.sm = torch.nn.Softmax(dim=1)
        
        # BERT
        self.pretrained_tokenizer = AutoTokenizer.from_pretrained(bert)
        self.pretrained_model = AutoModel.from_pretrained(bert)
        self.bert_dim = 768  # BERT encoding dimension
        for param in self.pretrained_model.base_model.parameters():  
                param.requires_grad = False                              # freezing the BERT encoder
    
        # NER
        self.ner_dim = ner_dim  # dimension of NER tagging scheme
        self.ner_lin = torch.nn.Linear(self.bert_dim, self.ner_dim)
        #self.ner_rnn = torch.nn.LSTM(self.ner_dim, self.ner_dim, bidirectional=False)
        
        # NED
        #self.ned_dim = 300  # dimension of the KB graph embedding space
        #self.ned_lin = torch.nn.Linear(self.bert_dim + self.ner_dim, self.ned_dim)

        # Head-Tail
        self.ht_dim = 128  # dimension of head/tail embedding
        #self.h_lin = torch.nn.Linear(self.bert_dim + self.ner_dim + self.ned_dim, self.ht_dim)
        #self.t_lin = torch.nn.Linear(self.bert_dim + self.ner_dim + self.ned_dim, self.ht_dim)
        self.h_lin = torch.nn.Linear(self.bert_dim + self.ner_dim, self.ht_dim)
        self.t_lin = torch.nn.Linear(self.bert_dim + self.ner_dim, self.ht_dim)

        # RE
        self.re_dim = re_dim  # dimension of RE classification space
        self.re_bil = torch.nn.Bilinear(self.ht_dim, self.ht_dim, self.re_dim)
        self.re_lin = torch.nn.Linear(2*self.ht_dim, self.re_dim, bias=False)  # we need only one bias, we can decide to
                                                                               # switch off either the linear or bilinear bias
        

    def forward(self, x):
        inputs = x['input_ids'][0][1:-1] # CONSIDER MAKING THIS A self.inputs !!!!
        x = self.BERT(x)
        #print('### BERT encoding:\n', x.shape)
        ner = self.NER(x)                                           # this is the output of the linear layer, should we use this as
        x = torch.cat((x, self.sm(ner)), 1)                         # as embedding or rather the softmax of this?
        #x = torch.cat((x, ner), 1)
        #print('### NER encoding:\n', x.shape)
        
        # remove non-entity tokens, before this we need to merge multi-token entities
        x, inputs = self.Entity_filter(x, inputs)
        if len(x) < 2:
            return (ner, None)
        #print('### Entities found:\n', x.shape)
        #ned = self.NED(x)
        #x = torch.cat((x, ned), 1)
        #print('### NED encoding:\n', x.shape)
        re = self.RE(x, inputs)
        #print('### RE encoding:\n', re.shape)
        #return ner, ned, re
        return (ner, re)


        

    def BERT(self, x):
        #inputs = self.pretrained_tokenizer(x, return_tensors="pt", padding=True, truncation=True, max_length=128)
        return self.pretrained_model(**x).last_hidden_state[0][1:-1]
        #return self.pretrained_model(**inputs).last_hidden_state[0][1:-1]     # [0] explaination:
                                                                   # The output of model(**x) is of shape (a,b,c) with a = batchsize,
                                                                   # which in our case equals 1 since we pass a single sentence,
                                                                   # b = max_lenght of the sentences in the batch and c = encoding_dim.
                                                                   # In case we want to use batchsize > 1 we need to change also the
                                                                   # other modules of the pipeline, for example using convolutional
                                                                   # layers in place of linear ones.
                                                                   # [1:-1] : we want to get rid of [CLS] and [SEP] tokens

    def NER(self, x):
        #x = self.ner_lin(x)
        #x, _ = self.ner_rnn(x.view(len(x),1,-1))
        #return x.view(x.size()[0], x.size()[2])
        return self.ner_lin(x)

    def Entity_filter(self, x, inputs):
        encodings = []
        relative_input = []
        for i in range(x.size()[0]): # consider doing this with map() for some speed up
            #if int(torch.argmax(xx[-self.ner_dim:])) != self.ner_dim - 1 :
             #   tmp.append(xx)
            if self.scheme.to_tag(x[i][-self.ner_dim:])[0] == 'E': # keep only End entity tokens
                encodings.append(x[i])
                relative_input.append(inputs[i])
        if len(encodings) !=0:
            return (torch.stack(encodings, dim=0), torch.stack(relative_input, dim=0))
        else:
            return ([], [])

    def NED(self, x):
        return self.ned_lin(x) # it's missing the dot product in the graph embedding space, the idea would be to find the closest
                               # concepts in embedding space and then return the closest in a greedy approach, or the closest ones
                               # with beam search. We also need to decide if it's better to use the predicted graph embedding of
                               # the concept or to map the predicted embedding to the corresponding true graph embedding and then 
                               # use this for RE.

    def HeadTail(self, x, inputs):
        h = self.h_lin(x)
        t = self.t_lin(x)
        # Building candidate pairs
        #head = torch.stack([ h for i in range(x.shape[0])], dim=1).view(x.shape[0]**2, h.shape[1])    # Combining all possible heads
        #tail = torch.stack([ t for i in range(x.shape[0])]).view(x.shape[0]**2, t.shape[1])           # with every possible tail
        ht = list(product(h,t))
        relative_inputs = list(product(inputs,inputs))
        # removing self-relations pairs
        for i in range(x.shape[0]):
            ht.pop(i*x.shape[0])                # we don't need i*x.shape[0] + i cause pop shortens the
            relative_inputs.pop(i*x.shape[0])   # list at each iteration, thus we have the +i for free
        ht = tuple(map(torch.stack, zip(*ht)))
        return (ht[0], ht[1], relative_inputs)

    def Biaffine(self, x, y):
        return self.re_bil(x,y) + self.re_lin(torch.cat((x,y), dim=1))

    def RE(self, x, inputs):
        x, y, relative_inputs = self.HeadTail(x, inputs)
        bi = self.Biaffine(x,y)
        #return { k : v for k, v in zip(relative_inputs,  bi) }
        #return list(zip(bi, relative_inputs))
        return (bi, relative_inputs)
    
    def unfreeze_bert_layer(self, i):
        for param in self.pretrained_model.base_model.encoder.layer[11-i].parameters():
                param.requires_grad = True


