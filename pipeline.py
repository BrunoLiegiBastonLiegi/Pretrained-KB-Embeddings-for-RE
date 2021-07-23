import random, torch, math, itertools, time, faiss

from transformers import AutoTokenizer, AutoModel
from itertools import product, repeat, chain
#from sklearn.neighbors import NearestNeighbors
from torch.nn.utils.rnn import pad_sequence
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from torch.multiprocessing import Pool, set_start_method, set_sharing_strategy
set_sharing_strategy('file_system')

class Pipeline(torch.nn.Module):

    def __init__(self, bert, ner_dim, ner_scheme, ned_dim, KB, re_dim):
        super().__init__()

        self.sm = torch.nn.Softmax(dim=2)
        self.dropout = torch.nn.Dropout(p=0.1)

        # BERT
        self.pretrained_tokenizer = AutoTokenizer.from_pretrained(bert)
        self.pretrained_model = AutoModel.from_pretrained(bert)
        self.bert_dim = self.pretrained_model.pooler.dense.out_features#768  # BERT encoding dimension
        for param in self.pretrained_model.base_model.parameters():  
                param.requires_grad = False                              # freezing the BERT encoder
    
        # NER # think about adding transition matrix for improvement
        self.scheme = ner_scheme
        # B, E and S indices for each entity type
        self.B = [ self.scheme.to_tensor('B-' + t, index=True).cuda() for t in self.scheme.e_types]
        self.E = [ self.scheme.to_tensor('E-' + t, index=True).cuda() for t in self.scheme.e_types]
        self.S = [ self.scheme.to_tensor('S-' + t, index=True).cuda() for t in self.scheme.e_types]
        self.ner_dim = ner_dim  # dimension of NER tagging scheme
        self.ner_lin0 = torch.nn.Linear(self.bert_dim, self.bert_dim)
        self.ner_lin = torch.nn.Linear(self.bert_dim, self.ner_dim)
        
        # NED
        self.KB = KB
        self.KB_embs = torch.vstack(list(KB.values()))
        #self.NN = faiss.IndexFlatL2(ned_dim)
        quantizer = faiss.IndexFlatL2(ned_dim)  # the other index
        self.NN = faiss.IndexIVFFlat(quantizer, ned_dim, 100)
        assert not self.NN.is_trained
        self.NN.train(self.KB_embs.numpy())
        assert self.NN.is_trained
        self.NN.add(self.KB_embs.numpy())
        self.NN.nprobe = 1
        self.n_neighbors = 10
        #self.nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='auto')
        #self.nbrs.fit(torch.vstack(self.KB_embs))
        self.ned_dim = ned_dim  # dimension of the KB graph embedding space
        nhead = 8
        hdim = int((self.bert_dim + self.ner_dim)/nhead)*nhead
        self.ned_lin1 = torch.nn.Linear(self.bert_dim + self.ner_dim, hdim)
        # nhead must be a divisor of hdim, pay attention!!!
        ned_transformer_layer = torch.nn.TransformerEncoderLayer(d_model=hdim, nhead=8)
        self.ned_transformer = torch.nn.TransformerEncoder(ned_transformer_layer, num_layers=6)
        self.relu = torch.nn.ReLU()
        #self.ned_lin1 = torch.nn.Linear(self.bert_dim + self.ner_dim, hdim)
        #self.ned_lin2 = torch.nn.Linear(hdim, hdim)
        #self.ned_lin3 = torch.nn.Linear(hdim, hdim)
        self.ned_lin = torch.nn.Linear(hdim, self.ned_dim)
        self.ned_dist_lin1 = torch.nn.Linear(self.ned_dim, self.ned_dim)
        self.ned_dist_lin2 = torch.nn.Linear(self.ned_dim, 1)
        self.ned_ctx_lin1 = torch.nn.Linear(self.ned_dim, self.ned_dim)
        self.ned_ctx_lin2 = torch.nn.Linear(self.ned_dim, 1)
        
        # Head-Tail
        self.ht_dim = 128  # dimension of head/tail embedding # apparently no difference between 64 and 128, but 32 seems to lead to better scores
        self.h_lin0 = torch.nn.Linear(self.bert_dim + self.ner_dim + self.ned_dim, self.bert_dim + self.ner_dim + self.ned_dim)
        self.h_lin = torch.nn.Linear(self.bert_dim + self.ner_dim + self.ned_dim, self.ht_dim)
        self.t_lin0 = torch.nn.Linear(self.bert_dim + self.ner_dim + self.ned_dim, self.bert_dim + self.ner_dim + self.ned_dim)
        self.t_lin = torch.nn.Linear(self.bert_dim + self.ner_dim + self.ned_dim, self.ht_dim)
        
        # RE
        self.re_dim = re_dim  # dimension of RE classification space
        self.re_bil = torch.nn.Bilinear(self.ht_dim, self.ht_dim, self.re_dim)
        self.re_lin = torch.nn.Linear(2*self.ht_dim, self.re_dim, bias=False)  # we need only one bias, we can decide to
                                                                               # switch off either the linear or bilinear bias
        

    def forward(self, x):
        inputs = torch.vstack([torch.tensor(range(1, len(x['input_ids'][0][1:-1]) + 1)) for i in range(x['input_ids'].shape[0])])
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        t1 = time.time()
        x = self.BERT(x)
        t2 = time.time()
        #print('> BERT:', t2-t1)
        t1 = time.time()
        ner = self.NER(x)
        t2 = time.time()
        #print('> NER:', t2-t1)
        x = torch.cat((x, self.sm(ner)), dim=-1)
        # detach the context
        ctx = x[:,0].unsqueeze(1)
        x = x[:,1:]
        ner = ner[:,1:]
        # remove non-entity tokens and merge multi-token entities
        t1 = time.time()
        #with Pool(6) as p:
            #x, inputs = list(zip(*p.starmap(self.Entity_filter_slow, zip(x,inputs))))
            #x, inputs = list(zip(*p.map(self.filter_worker, x)))
        x, inputs = self.Entity_filter(x)
        t2 = time.time()
        #print('> Filter:', t2-t1)
        # pad to max number of entities in batch sample
        #x, inputs = self.PAD(x), self.PAD(inputs, pad=-1)
        t1 = time.time()
        x, inputs = self.PAD(x, inputs)
        t2 = time.time()
        #print('> PAD:', t2-t1)
        if x.shape[1] == 0:
            return (ner, None, None)
        t1 = time.time()
        ned = self.NED(x, ctx)
        t2 = time.time()
        #print('> NED:', t2-t1)
        if x.shape[1] < 2:
            ned = (inputs, ned[0], ned[1])
            return (ner, ned, None)
        x = torch.cat((
            x,
            (self.sm(ned[1][:,:,:,0].view(x.shape[0], -1, self.n_neighbors, 1))*ned[1][:,:,:,1:]).sum(2)
            ), dim=-1)
        ned = (inputs, ned[0], ned[1])
        t1 = time.time()
        re = self.RE(x, inputs)
        t2 = time.time()
        #print('> RE:', t2-t1)
        return ner, ned, re

    def BERT(self, x):
        return self.pretrained_model(**x).last_hidden_state[:,:-1]     
                                                                   
    def NER(self, x):
        x = self.dropout(self.relu(self.ner_lin0(x)))
        return self.ner_lin(x)
    
    def filter_worker(self, x):
        amax = torch.argmax(x[:,-self.ner_dim:], dim=-1)
        ind = []
        for e,b,s in zip(self.B, self.E, self.S):
            B, E, S = (amax == b).nonzero(), (amax == e).nonzero(), (amax == s).nonzero()
            ind += zip(repeat('B', len(B)), B)
            ind += zip(repeat('E', len(E)), E)
            ind += zip(repeat('S', len(S)), S)
        ind = sorted(ind, key=lambda x: x[1])
        i = 0
        entities, positions = [], []
        print('entering loop')
        while i < len(ind):
            if ind[i][0] == 'B':
                start = ind[i][0]
                while ind[i][0] != 'E' and i < len(ind):
                    i += 1
                entities.append(torch.mean(
                    x[start:ind[i][1]+1],
                dim=0))
                positions.append(ind[i][1]+1)
            elif ind[i][0] == 'S':
                entities.append(x[ind[i][1]])
                positions.append(ind[i][1]+1)
                i += 1
            elif ind[i][0] == 'E':
                i += 1
        print('out of loop')
        return torch.vstack(entities), torch.vstack(positions)
                    

    def Entity_filter_fast(self, x):
        amax = torch.argmax(x[:,:,-self.ner_dim:], dim=-1)
        #indB = torch.vstack([(amax == i).nonzero() for i in self.B])
        #indE = torch.vstack([(amax == i).nonzero() for i in self.E])
        #indS = torch.vstack([(amax == i).nonzero() for i in self.S])
        indB, indE, indS = [], [], []
        for e,b,s in zip(self.B, self.E, self.S):
            indB.append((amax == b).nonzero())
            indE.append((amax == e).nonzero())
            indS.append((amax == s).nonzero())
        indB, indE, indS = torch.vstack(indB), torch.vstack(indE), torch.vstack(indS)
        #l = min(indB.shape[0], indE.shape[0])
        indB = sorted(indB, key=lambda x: min(x[0],x[1]))#[:l]
        indE = sorted(indE, key=lambda x: min(x[0],x[1]))#[:l]
        entities = dict(zip(range(x.shape[0]), repeat([], x.shape[0])))
        positions = dict(zip(range(x.shape[0]), repeat([], x.shape[0])))#dict(entities)
        j = 0
        for b in indB:
            while j < len(indE):
                if indE[j][0] == b[0]:
                    if indE[j][1] > b[1]:
                        entities[b[0].item()].append(torch.mean(
                            x[b[0], b[1]:indE[j][1]+1, :],
                        dim=0))
                        positions[b[0].item()].append(indE[j][1]+1)
                        j += 1
                        break
                    else:
                        j += 1
                else:
                    break
        for s in indS:
            entities[s[0].item()].append(x[s[0], s[1], :])
            positions[s[0].item()].append(s[1]+1)
        for k,e,i in zip(entities.keys(), entities.values(), positions.values()):
            entities[k], positions[k] = torch.vstack(e), torch.vstack(i)
        return list(entities.values()), list(positions.values())


    def Entity_filter(self, x):
        amax = torch.argmax(x[:,:,-self.ner_dim:], dim=-1)
        inputs, entities = torch.zeros(1, 2).int().cuda(), torch.zeros(1, x.shape[-1]).int().cuda()
        for t in self.scheme.e_types: # there is the problem of overlapping entities of different types
            indB = (amax == self.scheme.to_tensor('B-' + t, index=True).cuda()).nonzero()
            indE = (amax == self.scheme.to_tensor('E-' + t, index=True).cuda()).nonzero()
            inputs = torch.vstack([
                inputs,
                (amax == self.scheme.to_tensor('S-' + t, index=True).cuda()).nonzero()
            ]) # Store S-entities
            entities = torch.vstack([
                entities,
                x[amax == self.scheme.to_tensor('S-' + t, index=True).cuda()]
            ])# and relative inputs
            end = [-1, -1] # initial end value for entering the loop
            for i in indB: # there is the problem of overlaping S and B-E entities
                start = i
                if start[0] != end[0]:
                    end = [-1, -1]
                if start[1] > end[1]: # check that the Bs don't overlap
                    ind = (indE[:,0] == i[0]).nonzero()
                    tmp = indE[ind].squeeze(1)
                    if len(tmp) > 0:
                        tmp = (tmp[:,1] > i[1]).nonzero()
                        if len(tmp) > 0:
                            end = indE[ind[tmp[0]]].view(2)
                        else:
                            end = torch.hstack(( i[0], torch.tensor(x.shape[1]).cuda() ))
                    else:
                        end = torch.hstack(( i[0], torch.tensor(x.shape[1]).cuda() ))
                    entities = torch.vstack([
                        entities,
                        torch.mean(x[start[0].item(), start[1].item():end[1].item()], dim=0)
                    ])
                    inputs = torch.vstack([inputs, end])
                    
        inputs, entities = inputs[1:], entities[1:] # get rid of the first torch.zeros initalization
        l_ents = []
        l_inp = []
        for i in range(x.shape[0]):
            inds = (inputs[:,0] == i).nonzero()
            l_inp.append(inputs[inds].squeeze(1)[:,1].view(-1,1) + 1) # the +1 is needed for inputs matching
            l_ents.append(entities[inds].squeeze(1))                  # the labels
        return l_ents, l_inp
            
                
    # I don't particularly like this implementation of the filter
    # I'd like to find the time to change it
    def Entity_filter_slow(self, x, inputs):
        # awesome one-liner to get the indices of all the non predicted-O tokens
        #list(map(lambda y: list(filter(lambda x: x[1]!=self.scheme.to_tensor('O', index=True), enumerate(y))), torch.argmax(x[:,:,-self.ner_dim:])))
        entities = []
        relative_input = []
        i = 0
        while i < x.shape[0]:
            tag = self.scheme.to_tag(x[i][-self.ner_dim:])[0]
            if tag == 'B':
                tensor_mean = [ x[i] ]
                while tag != 'E':
                    i += 1
                    if i >= x.shape[0]:
                        break
                    tag = self.scheme.to_tag(x[i][-self.ner_dim:])[0]
                    tensor_mean.append(x[i])
                tensor_mean = torch.stack(tensor_mean, dim=0)
                entities.append(torch.mean(tensor_mean, 0))
                if i < x.shape[0]:
                    relative_input.append(inputs[i].view(1))
                else:
                    relative_input.append(inputs[i-1].view(1)) # problem if B is the last token
                i += 1
            elif tag == 'S':
                entities.append(x[i])
                relative_input.append(inputs[i].view(1))
                i += 1
            else:
                i += 1
                        
        if len(entities) !=0:
            return (torch.stack(entities, dim=0), torch.stack(relative_input, dim=0))
        else:
            return ([], [])

    def PAD(self, x, inputs):
        return (
            pad_sequence(x, batch_first=True, padding_value=0.),
            pad_sequence(inputs, batch_first=True, padding_value=-1)
        )

    def PAD_slow(self, l, pad=0):
        max_len = 0
        dim = 0
        for item in l:
            max_len = max(max_len, len(item))
            if item != [] and dim == 0:
                dim = item.shape[1]
        p = []
        for item in l:
            if item != []:
                p.append(torch.vstack((
                    item,
                    pad*torch.ones(max_len - len(item), dim, dtype=torch.int).cuda()
                )).unsqueeze(0))
            else:
                p.append(pad*torch.ones(max_len, dim, dtype=torch.int).unsqueeze(0).cuda())
        return torch.vstack(p)
    
    def NED(self, x, ctx):
        x = torch.cat((ctx, x), dim=1)
        x = self.dropout(self.relu(self.ned_lin1(x)))
        x = self.ned_transformer(x)
        x = self.ned_lin(x)
        ctx, x = x[:,0].unsqueeze(1), x[:,1:]
        ned_1 = x  # predicted graph embeddings
        
        #_, indices = zip(*map(self.NN.search, ned_1.detach().cpu().numpy(), repeat(self.n_neighbors, ned_1.shape[0])))
        _, indices = self.NN.search(
            ned_1.reshape(-1,self.ned_dim).detach().cpu().numpy(),
            self.n_neighbors
        )
        #_, indices = zip(*map(self.KB.search, ned_1.detach().cpu().numpy(), repeat(self.n_neighbors, ned_1.shape[0])))
        #indices, _ = zip(*chain(*map(
        #    self.KB.search,
        #    ned_1.reshape(-1, self.ned_dim).detach().cpu(),
        #    repeat(self.n_neighbors, ned_1.shape[0]*ned_1.shape[1])
        #)))
        #print(list(map(self.KB.get_object, indices)))
        #candidates = torch.tensor(list(map(self.KB.get_object, indices))).view(x.shape[0], -1, self.n_neighbors, self.ned_dim)
        indices = torch.tensor(indices)
        indices[indices==-1] = 0# random.randint(0, len(self.KB_embs)-1)
        candidates = torch.index_select(
            self.KB_embs,
            0,
            indices.flatten()
        ).view(x.shape[0], -1, self.n_neighbors, self.ned_dim)
        candidates = candidates.cuda()
        candidates.requires_grad = True
        x = (candidates - x.unsqueeze(2))
        ctx = 1*(ctx.unsqueeze(2) * candidates)
        x = self.dropout(self.relu(self.ned_dist_lin1(x)))
        x = self.ned_dist_lin2(x)
        ctx = self.dropout(self.relu(self.ned_ctx_lin1(ctx)))
        ctx = self.ned_ctx_lin2(ctx)
        x = x + ctx
        ned_2 = torch.cat((x,candidates), dim=-1) # scores in first position candidates after score

        return ned_1, ned_2
        
    def HeadTail(self, x, inputs):
        h, t = self.dropout(self.relu(self.h_lin0(x))), self.dropout(self.relu(self.t_lin0(x)))
        h, t = self.h_lin(x), self.t_lin(x)
        # Building candidate pairs
        # Combining all possible heads
        # with every possible tail
        h, t = map(
            lambda y: torch.vstack(y).view(h.shape[0], -1, h.shape[-1]),
            tuple(zip(*map(self.pairs, h, t)))
        )
        inputs = torch.cat(tuple(map(
            lambda y: torch.vstack(y).view(inputs.shape[0], -1, 1),
            tuple(zip(*map(self.pairs, inputs, inputs)))
        )), dim=-1)
        return h, t, inputs

    def pairs(self, l1, l2):
        p = list(product(l1,l2))
        # removing self-relation pairs
        for i in range(len(l1)):
            p.pop(i*len(l1))
        return tuple(map(torch.stack, zip(*p)))

    def Biaffine(self, x, y):
        return self.re_bil(x,y) + self.re_lin(torch.cat((x,y), dim=-1))

    def RE(self, x, inputs):
        x, y, inputs = self.HeadTail(x, inputs)
        bi = self.Biaffine(x,y)
        return inputs, bi
    
    def unfreeze_bert_layer(self, i):
        print('> Unfreezing BERT layer ', len(self.pretrained_model.encoder.layer)-1-i)
        for param in self.pretrained_model.encoder.layer[len(self.pretrained_model.encoder.layer)-1-i].parameters():
                param.requires_grad = True


