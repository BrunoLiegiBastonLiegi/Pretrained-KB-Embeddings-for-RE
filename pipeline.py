import random, torch, math, itertools

from transformers import AutoTokenizer, AutoModel
from itertools import product
from sklearn.neighbors import NearestNeighbors


class Pipeline(torch.nn.Module):

    def __init__(self, bert, ner_dim, ner_scheme, ned_dim, KB, re_dim):
        super().__init__()

        self.sm = torch.nn.Softmax(dim=1)
        
        # BERT
        self.pretrained_tokenizer = AutoTokenizer.from_pretrained(bert)
        self.pretrained_model = AutoModel.from_pretrained(bert)
        self.bert_dim = 768  # BERT encoding dimension
        for param in self.pretrained_model.base_model.parameters():  
                param.requires_grad = False                              # freezing the BERT encoder
    
        # NER # think about adding transition matrix for improvement
        self.scheme = ner_scheme
        self.ner_dim = ner_dim  # dimension of NER tagging scheme
        self.ner_lin = torch.nn.Linear(self.bert_dim, self.ner_dim)
        
        # NED
        self.KB = KB
        self.KB_embs = list(KB.values())
        self.n_neighbors = 10
        self.nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='auto')
        self.nbrs.fit(torch.vstack(self.KB_embs))
        self.ned_dim = ned_dim  # dimension of the KB graph embedding space
        hdim = self.bert_dim + self.ner_dim 
        #self.dropout = torch.nn.Dropout(p=0.1)
        self.relu = torch.nn.ReLU()
        self.ned_lin1 = torch.nn.Linear(self.bert_dim + self.ner_dim, hdim)
        self.ned_lin2 = torch.nn.Linear(hdim, hdim)
        self.ned_lin3 = torch.nn.Linear(hdim, hdim)
        self.ned_lin = torch.nn.Linear(hdim, self.ned_dim)
        self.ned_lin0 = torch.nn.Linear(2*self.ned_dim, 1)
        
        # Head-Tail
        self.ht_dim = 32#128  # dimension of head/tail embedding # apparently no difference between 64 and 128, but 32 seems to lead to better scores
        self.h_lin = torch.nn.Linear(self.bert_dim + self.ner_dim + self.ned_dim, self.ht_dim)
        self.t_lin = torch.nn.Linear(self.bert_dim + self.ner_dim + self.ned_dim, self.ht_dim)
        #self.h_lin = torch.nn.Linear(self.bert_dim + self.ner_dim, self.ht_dim)
        #self.t_lin = torch.nn.Linear(self.bert_dim + self.ner_dim, self.ht_dim)

        # RE
        self.re_dim = re_dim  # dimension of RE classification space
        self.re_bil = torch.nn.Bilinear(self.ht_dim, self.ht_dim, self.re_dim)
        self.re_lin = torch.nn.Linear(2*self.ht_dim, self.re_dim, bias=False)  # we need only one bias, we can decide to
                                                                               # switch off either the linear or bilinear bias
        

    def forward(self, x):
        #inputs = x['input_ids'][0][1:-1] # CONSIDER MAKING THIS A self.inputs !!!!
        inputs = torch.tensor(range(1, len(x['input_ids'][0][1:-1]) + 1))
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        x = self.BERT(x)
        ner = self.NER(x)                                           # this is the output of the linear layer, should we use this as
        x = torch.cat((x, self.sm(ner)), 1)                         # as embedding or rather the softmax of this?
        #x = torch.cat((x, ner), 1)
        # detach the context
        ctx = x[0]
        x = x[1:]
        ner = ner[1:]
        # remove non-entity tokens and merge multi-token entities
        x, inputs = self.Entity_filter(x, inputs, filt='merge')
        if len(x) == 0:
            return (ner, None, None)
        ned = self.NED(x, ctx)
        #ned = self.NED(x)
        if len(x) < 2:
            ned = (inputs, ned)
            return (ner, ned, None)
            #return (ner, None, None)
        x = torch.cat((
            x,
            torch.sum(ned[1][:,:,0].view(-1, self.n_neighbors, 1)*ned[1][:,:,1:], dim=1)
            ), dim=1)
        #x = torch.cat((x, ned), 1)
        ned = (inputs, ned)
        re = self.RE(x, inputs)
        return ner, ned, re
        #return ner, None, re


        

    def BERT(self, x):
        #inputs = self.pretrained_tokenizer(x, return_tensors="pt", padding=True, truncation=True, max_length=128)
        #return self.pretrained_model(**x).last_hidden_state[0][1:-1]
        return self.pretrained_model(**x).last_hidden_state[0][:-1]
        #return self.pretrained_model(**inputs).last_hidden_state[0][1:-1]     # [0] explaination:
                                                                   # The output of model(**x) is of shape (a,b,c) with a = batchsize,
                                                                   # which in our case equals 1 since we pass a single sentence,
                                                                   # b = max_lenght of the sentences in the batch and c = encoding_dim.
                                                                   # In case we want to use batchsize > 1 we need to change also the
                                                                   # other modules of the pipeline, for example using convolutional
                                                                   # layers in place of linear ones.
                                                                   # [1:-1] : we want to get rid of [CLS] and [SEP] tokens

    def NER(self, x):
        return self.ner_lin(x)

    def Entity_filter(self, x, inputs, filt='E'): # filt can be 'E'/'B' for last/first token filtering or 'merge' for merging
        #for i in range(x.size()[0]): # consider doing this with map() for some speed up
         #       tag = self.scheme.to_tag(x[i][-self.ner_dim:])[0]
          #      print(tag)
        encodings = []
        relative_input = []
        if filt != 'merge':
            for i in range(x.size()[0]): # consider doing this with map() for some speed up
                tag = self.scheme.to_tag(x[i][-self.ner_dim:])[0]
                if tag == 'E' or tag == 'S': # keep only End/Single entity tokens
                    encodings.append(x[i])
                    relative_input.append(inputs[i])
        else:
            i = 0
            while i < x.size()[0]:
                tag = self.scheme.to_tag(x[i][-self.ner_dim:])[0]
                if tag == 'B':
                    tensor_mean = [ x[i] ]
                    #input_mean = [ inputs[i].view(1) ]
                    # maybe its's better not just to stop for 'E' but also for 'O', in case we don't find any
                    while tag != 'E':
                        i += 1
                        if i >= x.size()[0]:
                            break
                        tag = self.scheme.to_tag(x[i][-self.ner_dim:])[0]
                        tensor_mean.append(x[i])
                        #input_mean.append(inputs[i].view(1))
                    tensor_mean = torch.stack(tensor_mean, dim=0)
                    #input_mean = torch.cat(input_mean)
                    encodings.append(torch.mean(tensor_mean, 0))
                    #relative_input.append(input_mean)
                    if i < x.size()[0]:
                        relative_input.append(inputs[i].view(1))
                    else:
                        relative_input.append(inputs[i-1].view(1)) # problem if B is the last token
                elif tag == 'S':
                    encodings.append(x[i])
                    relative_input.append(inputs[i].view(1))
                    i += 1
                # could be a good idea to consider also 'E' tags alone without an initial 'B', so elif tag == 'E':
                else:
                    i += 1
                        
        if len(encodings) !=0:
            """
            if filt == 'merge':
                max_len = max([len(i) for i in relative_input])
                relative_input = [torch.nn.functional.pad(i, pad=(0, max_len - i.numel()), mode='constant', value=-1) for i in relative_input]
            """
            return (torch.stack(encodings, dim=0), torch.stack(relative_input, dim=0))
        else:
            return ([], [])

    def NED(self, x, ctx):
        #ctx = torch.vstack([ctx for i in range(len(x))])
        x = torch.vstack((ctx, x))
        #x = relu(self.ned_bil(ctx, x) + self.ned_lin0(torch.cat((ctx,x), dim=1)))
        x = self.relu(self.ned_lin1(x))
        x = self.relu(self.ned_lin2(x))
        x = self.relu(self.ned_lin3(x))
        x = self.ned_lin(x)
        ctx, x = x[0], x[1:]
        ned_1 = x  # predicted graph embeddings
        
        #print(ned_1)
        _, indices = self.nbrs.kneighbors(x.detach().cpu())
        x = torch.vstack([self.KB_embs[i] for i in indices.flatten()]).view(-1, self.n_neighbors, self.ned_dim)
        x = x.cuda()
        x.requires_grad = True
        candidates = x # selected candidates in the KB
        #print(x.shape)
        x = torch.vstack(list(itertools.starmap(lambda x,y: x*y, zip(x, ned_1))))  # candidates*original_prediction (ned_1) product
        #print(x.shape)
        x = torch.vstack(list(map(lambda t: torch.hstack((t, ctx.squeeze(0))), x)))  # concatenation of context ctx
        #print(x.shape)                                                        # might be better to do the element-wise product
        x = self.sm(self.ned_lin0(x).view(-1, self.n_neighbors, 1))
        #print(x.shape)
        #print('LIN OUT\n',x)
        #cat = torch.distributions.Categorical(x.view(-1,1,x.shape[1])) # sample depending on the score
        #x = torch.nn.functional.one_hot(cat.sample(), num_classes=self.n_neighbors).view(-1,self.n_neighbors,1) 
        #print('ONE_HOT\n',x)
        #print(x.shape)
        #print(candidates.shape)
        #x = x*candidates                               
        #print('FINAL CANDIDATE\n',x)
        #x = torch.sum(x, dim=1)                         # the sum is just to get rid of the (0,0,...,0) tensors obtained with the last product
        #print('SUM\n',x)
        #ned_2 = torch.vstack([ candidates[i][j] for i,j in enumerate(indices)]) # predicted true pre-trained embeddings
        ned_2 = torch.hstack((x.view(-1,1), candidates.view(-1, self.ned_dim))).view(-1, self.n_neighbors, self.ned_dim + 1)
        #print(ned_2)
        #ctx = torch.vstack([x[0] for i in range(len(x)-1)])
        #x = x[1:]
        #x = self.ned_bil(ctx, x) + self.ned_lin0(torch.cat((ctx,x), dim=1))
        
        #x = relu(self.ned_lin4(x))
        #x = relu(self.ned_lin5(x))
        #x = relu(self.ned_lin6(x))
        #x = relu(self.ned_lin7(x))
        #x = relu(self.ned_lin8(x))
        #x = relu(self.ned_lin9(x))
        #x = relu(self.ned_lin10(x))
        #x = relu(self.ned_lin11(x))
        #x = relu(self.ned_lin12(x))
        
        #return x
        return ned_1, ned_2
        
    def HeadTail(self, x, inputs):
        h = self.h_lin(x)
        t = self.t_lin(x)
        # Building candidate pairs
        # Combining all possible heads
        # with every possible tail
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
        return (torch.tensor(relative_inputs).cuda(), bi)
    
    def unfreeze_bert_layer(self, i):
        for param in self.pretrained_model.base_model.encoder.layer[11-i].parameters():
                param.requires_grad = True


