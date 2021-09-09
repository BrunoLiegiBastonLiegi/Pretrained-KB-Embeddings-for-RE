import torch, faiss, numpy
from transformers import AutoTokenizer, AutoModel
from itertools import product
from torch.nn.utils.rnn import pad_sequence

# --------------------------------------------------------------------------------------------------------------------

class PretrainedLanguageModel(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = AutoModel.from_pretrained(model)
        self.dim = self.model.pooler.dense.out_features      # encoding dimension
        for param in self.model.base_model.parameters():     # freezing the encoder
                param.requires_grad = False                              

    def forward(self, x):
        return self.model(**x).last_hidden_state[:,:-1]

    def unfreeze_layer(self, i):
        print('> Unfreezing encoder layer ', len(self.model.encoder.layer)-1-i)
        for param in self.model.encoder.layer[len(self.model.encoder.layer)-1-i].parameters():
                param.requires_grad = True

# --------------------------------------------------------------------------------------------------------------------

class NERModule(torch.nn.Module):

    def __init__(self, n_layers, in_dim, out_dim, activation=torch.nn.ReLU(), dropout=0.1):
        super().__init__()
        layers = []
        for i in range(n_layers-1):
            layers.append(torch.nn.Linear(in_dim, in_dim))
            layers.append(torch.nn.Dropout(dropout))
            layers.append(activation)
        layers.append(torch.nn.Linear(in_dim, out_dim))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
# --------------------------------------------------------------------------------------------------------------------
    
class NEDModule(torch.nn.Module):

    def __init__(self, n_layers, in_dim, out_dim, KB, activation=torch.nn.ReLU(), dropout=0.1, device=None):
        super().__init__()
        self.dev = device 
        
        # Stuff for the NN search
        self.KB_embs = torch.vstack(list(KB.values()))
        quantizer = faiss.IndexFlatL2(out_dim) 
        self.NN = faiss.IndexIVFFlat(quantizer, out_dim, 100)
        assert not self.NN.is_trained
        self.NN.train(self.KB_embs.numpy())
        assert self.NN.is_trained
        self.NN.add(self.KB_embs.numpy())
        self.NN.nprobe = 1
        self.n_neighbors = 10
        
        # Mapper
        self.out_dim = out_dim
        nhead = 8
        h_dim = int((in_dim)/nhead)*nhead
        transformer_layer = torch.nn.TransformerEncoderLayer(d_model=h_dim, nhead=nhead)
        # nhead must be a divisor of h_dim, pay attention!!!
        drop = torch.nn.Dropout(dropout)
        self.mapper = torch.nn.Sequential(
            torch.nn.Linear(in_dim, h_dim),
            drop,
            activation,
            torch.nn.TransformerEncoder(transformer_layer, num_layers=n_layers),
            torch.nn.Linear(h_dim, out_dim)
        ) # mapping to the graph embedding space

        # Scores
        self.distance_score = torch.nn.Sequential(
            torch.nn.Linear(out_dim, out_dim),
            drop,
            activation,
            torch.nn.Linear(out_dim, 1)
        )
        self.context_score = torch.nn.Sequential(
            torch.nn.Linear(out_dim, out_dim),
            drop,
            activation,
            torch.nn.Linear(out_dim, 1)
        )

    def forward(self, x, ctx):
        if self.dev == None: # infer device if not provided
            self.dev = x.device
        x = torch.cat((ctx, x), dim=1)
        x = self.mapper(x)
        ctx, x = x[:,0].unsqueeze(1), x[:,1:] # separate the context

        # Find the candidates
        prediction = x.clone()
        with torch.cuda.amp.autocast(enabled=False):
            _, indices = self.NN.search(
                prediction.float().reshape(-1,self.out_dim).detach().cpu().numpy(),
                self.n_neighbors
            )
        indices = torch.tensor(indices)
        indices[indices==-1] = 0
        candidates = torch.index_select(
            self.KB_embs,
            0,
            indices.flatten()
        ).view(x.shape[0], -1, self.n_neighbors, self.out_dim)
        candidates = candidates.to(self.dev)
        candidates.requires_grad = True

        # Assign score to each candidate
        x = (candidates - x.unsqueeze(2)) # distance from prediction
        ctx = ctx.unsqueeze(2) * candidates # attention to context
        x = self.distance_score(x)
        ctx = self.context_score(ctx)
        candidates = torch.cat((x,candidates), dim=-1) # scores in first position candidates after score
        return prediction, candidates
        
# --------------------------------------------------------------------------------------------------------------------

class REModule(torch.nn.Module):

    def __init__(self, in_dim, out_dim, h_dim, activation=torch.nn.ReLU(), dropout=0.1):
        super().__init__()
        
        # Head - Tail decomposition
        drop = torch.nn.Dropout(dropout)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(in_dim, h_dim),
            drop,
            activation,
            torch.nn.Linear(h_dim, h_dim),
        )
        self.tail = torch.nn.Sequential(
            torch.nn.Linear(in_dim, h_dim),
            drop,
            activation,
            torch.nn.Linear(h_dim, h_dim),
        )
        # Biaffine attention layers
        self.bil = torch.nn.Bilinear(h_dim, h_dim, out_dim)
        self.lin = torch.nn.Linear(2*h_dim, out_dim, bias=False)  # we need only one bias, we can decide to
                                                                  # switch off either the linear or bilinear bias                                                                      
    def forward(self, x, positions):
        head, tail = self.head(x), self.tail(x)
        # Building candidate pairs by combining all possible heads
        # with every possible tail
        head, tail = map(
            lambda y: torch.vstack(y).view(head.shape[0], -1, head.shape[-1]),
            tuple(zip(*map(self.pairs, head, tail)))
        )
        positions = torch.cat(tuple(map(
            lambda y: torch.vstack(y).view(positions.shape[0], -1, 1),
            tuple(zip(*map(self.pairs, positions, positions)))
        )), dim=-1)
        return positions, self.Biaffine(head, tail)

    def pairs(self, l1, l2):
        p = list(product(l1,l2))
        # removing self-relation pairs
        for i in range(len(l1)):
            p.pop(i*len(l1))
        return tuple(map(torch.stack, zip(*p)))

    def Biaffine(self, x, y):
        return self.bil(x,y) + self.lin(torch.cat((x,y), dim=-1))

# --------------------------------------------------------------------------------------------------------------------


class BaseIEModel(torch.nn.Module):
    """
    End-to-End model for NER and RE.
    """

    def __init__(self, language_model, ner_dim, ner_scheme, re_dim, device=torch.device('cpu')):
        super().__init__()

        # Misc
        self.sm = torch.nn.Softmax(dim=2)
        self.dev = device
        self.entity_lim = 10 # maximum number of entities admitted after entity filter

        # Pretrained Language Model
        self.lang_model = PretrainedLanguageModel(language_model)
        
        # NER
        self.ner_dim = ner_dim
        self.ner_scheme = ner_scheme
        self.NER = NERModule(
            n_layers = 2,
            in_dim = self.lang_model.dim,
            out_dim = ner_dim,
        )

        # RE
        self.RE = REModule(
            in_dim = self.lang_model.dim + ner_dim,
            out_dim = re_dim,
            h_dim = 512
        )
        
        # Move itself to device
        self.to(device)

    def forward(self, x):
        x = self.lang_model(x)
        ner = self.NER(x)
        x = torch.cat((x, self.sm(ner)), dim=-1)
        ctx, x, ner = x[:,0].unsqueeze(1), x[:,1:], ner[:,1:]
        x, positions = self.Entity_filter(x)
        x, positions = self.PAD(x, positions)
        if x.shape[1] > self.entity_lim:
            x, positions = x[:, :self.entity_lim, :], positions[:, :self.entity_lim, :]
        if x.shape[1] < 2:
            return [ner, None]
        re = self.RE(x, positions)
        return [ner, re]
        
    def Entity_filter(self, x):
        amax = torch.argmax(x[:,:,-self.ner_dim:], dim=-1)
        inputs, entities = torch.zeros(1, 2).int().to(self.dev), torch.zeros(1, x.shape[-1]).int().to(self.dev)
        for t in self.ner_scheme.e_types: # there is the problem of overlapping entities of different types
            indB = (amax == self.ner_scheme.to_tensor('B-' + t, index=True).to(self.dev)).nonzero()
            indE = (amax == self.ner_scheme.to_tensor('E-' + t, index=True).to(self.dev)).nonzero()
            inputs = torch.vstack([
                inputs,
                (amax == self.ner_scheme.to_tensor('S-' + t, index=True).to(self.dev)).nonzero()
            ]) # Store S-entities
            entities = torch.vstack([
                entities,
                x[amax == self.ner_scheme.to_tensor('S-' + t, index=True).to(self.dev)]
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
                            end = torch.hstack(( i[0], torch.tensor(x.shape[1]).to(self.dev) ))
                    else:
                        end = torch.hstack(( i[0], torch.tensor(x.shape[1]).to(self.dev) ))
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

    def PAD(self, x, inputs):
        return (
            pad_sequence(x, batch_first=True, padding_value=0.),
            pad_sequence(inputs, batch_first=True, padding_value=-1)
        )

    def prepare_inputs_targets(self, batch):
        inputs = [batch['sent'].to(self.dev)]
        targets = [
            batch['ner'].to(self.dev),
            list(map(lambda x: x.to(self.dev), batch['re']))
        ]
        return inputs, targets

    def ner_loss(self, prediction, target):
        return torch.nn.functional.cross_entropy(
            torch.transpose(prediction, 1, 2),
            target
        )

    def re_loss(self, prediction, target, no_rel_idx=0, random_re_err=1.):
        loss = 0.
        for i in range(len(target)):
            g = dict(zip(
                map( tuple, target[i][:,:2].tolist() ),
                target[i][:,2]
            ))
            p = dict(zip(
                map( tuple, prediction[0][i].tolist() ),
                prediction[1][i]
            ))
            re_pred, re_target = [], []
            for k in g.keys() & p.keys():
                re_pred.append(p.pop(k))
                re_target.append(g.pop(k))
            loss += random_re_err*len(g)
            if self.training:
                for k,v in p.items():
                    if -1 not in k:
                        re_pred.append(v)
                        re_target.append(torch.tensor(no_rel_idx, dtype=torch.int, device=self.dev))
            if len(re_pred) > 0:
                loss += torch.nn.functional.cross_entropy(torch.vstack(re_pred), torch.hstack(re_target).long())
        return loss / len(target)

    def loss(self, predictions, targets, **kwargs):
        args = {'no_rel_idx': None, 'random_re_err': None}
        return {
            'ner': self.ner_loss(predictions[0], targets[0]),
            'ned': torch.tensor([0., 0.], device=self.dev),
            're': self.re_loss(
                predictions[1],
                targets[1],
                **{k: kwargs[k] for k in args.keys() & kwargs.keys()}
            ) if predictions[1] != None else torch.tensor(kwargs['random_re_err'], device=self.dev)
        }
            

# --------------------------------------------------------------------------------------------------------------------


class BaseIEModelGoldEntities(BaseIEModel):

    def __init__(self, language_model, re_dim, device=torch.device('cpu')):
        super(BaseIEModel, self).__init__()

        # Misc
        self.sm = torch.nn.Softmax(dim=2)
        self.dev = device
    
        # Pretrained Language Model
        self.lang_model = PretrainedLanguageModel(language_model)

        # RE
        self.re_dim = re_dim
        self.RE = REModule(
            in_dim = self.lang_model.dim,
            out_dim = re_dim,
            h_dim = 512
        )

        # Move itself to device
        self.to(device)

    def forward(self, x, entities):
        x = self.lang_model(x)
        ctx = x[:,0].unsqueeze(1)
        x = x[:,1:]
        # get the entities
        x, positions = self.get_entities(x, entities)
        x, positions = self.PAD(x, positions)
        re = self.RE(x, positions)
        return [re]

    def get_entities(self, x, entities):
        ner, positions = [], []
        for i,e in enumerate(entities):
            ner_tmp, pos_tmp = [], []
            for p in e:
                ner_tmp.append(torch.mean(x[i][p[0]:p[1]], dim=0))
                pos_tmp.append(p[1])
            ner.append(torch.vstack(ner_tmp).to(self.dev))
            positions.append(torch.vstack(pos_tmp).to(self.dev))
        return ner, positions

    def prepare_inputs_targets(self, batch):
        inputs = [
            batch['sent'].to(self.dev),
            list(map(lambda x: x.to(self.dev), batch['pos']))
        ]
        targets = [list(map(lambda x: x.to(self.dev), batch['re']))]
        return inputs, targets

    def loss(self, predictions, targets, **kwargs):
        args = {'no_rel_idx': None, 'random_re_err': None}
        return {
            'ner': torch.tensor(0., device=self.dev),
            'ned': torch.tensor([0., 0.], device=self.dev),
            're': self.re_loss(predictions[0], targets[0], **{k: kwargs[k] for k in args.keys() & kwargs.keys()})
        }
            
# --------------------------------------------------------------------------------------------------------------------


class IEModel(BaseIEModel):
    """
    End-to-End model for NER, NED and RE.
    """
    def __init__(self, language_model, ner_dim, ner_scheme, ned_dim, KB, re_dim, device=torch.device('cpu')):
        super(BaseIEModel, self).__init__()
        
        # Misc
        self.sm = torch.nn.Softmax(dim=2)
        self.dev = device
        self.entity_lim = 10 # maximum number of entities admitted after entity filter

        # Pretrained Language Model
        self.lang_model = PretrainedLanguageModel(language_model)
        
        # NER
        self.ner_dim = ner_dim
        self.ner_scheme = ner_scheme
        self.NER = NERModule(
            n_layers = 2,
            in_dim = self.lang_model.dim,
            out_dim = ner_dim,
        )

        # NED
        self.ned_dim = ned_dim
        self.NED = NEDModule(
            n_layers = 6,
            in_dim = self.lang_model.dim + ner_dim,
            out_dim = ned_dim,
            KB = KB,
            device = device
        )
        
        # RE
        self.RE = REModule(
            in_dim = self.lang_model.dim + ned_dim + ner_dim,
            out_dim = re_dim,
            h_dim = 512
        )

        # Move itself to device
        self.to(device)

    def forward(self, x):
        x = self.lang_model(x)
        ner = self.NER(x)
        x = torch.cat((x, self.sm(ner)), dim=-1)
        ctx, x, ner = x[:,0].unsqueeze(1), x[:,1:], ner[:,1:]
        x, positions = self.Entity_filter(x)
        x, positions = self.PAD(x, positions)
        if x.shape[1] > self.entity_lim:
            x, positions = x[:, :self.entity_lim, :], positions[:, :self.entity_lim, :]
        if x.shape[1] == 0:
            return [ner, None, None]
        ned = self.NED(x, ctx)
        if x.shape[1] < 2:
            ned = (positions, ned[0], ned[1])
            return [ner, ned, None]
        x = torch.cat((
            x,
            (self.sm(ned[1][:,:,:,0].view(x.shape[0], -1, self.NED.n_neighbors, 1))*ned[1][:,:,:,1:]).sum(2)
            ), dim=-1)
        ned = (positions, ned[0], ned[1])
        re = self.RE(x, positions)
        return [ner, ned, re]

    def prepare_inputs_targets(self, batch):
        inputs = [batch['sent'].to(self.dev)]
        targets = [
            batch['ner'].to(self.dev),
            list(map(lambda x: x.to(self.dev), batch['ned'])),
            list(map(lambda x: x.to(self.dev), batch['re']))
        ]
        return inputs, targets

    def ned_loss(self, predictions, targets, random_ned_err=[1.,1.]):
        loss1, loss2 = 0., 0.
        for i in range(len(targets)):
            t = dict(zip(
                targets[i][:,0].int().tolist(),
                targets[i][:,1:]
            ))
            n1 = dict(zip(
                torch.flatten(predictions[0][i]).tolist(),
                predictions[1][i]
            ))
            n2 = dict(zip(
                torch.flatten(predictions[0][i]).tolist(),
                predictions[2][i]
            ))
            n2_scores, n2_targets = [], []
            n1_pred, n1_target = [], []
            for k in t.keys() & n1.keys():
                gt_tmp = t.pop(k)
                n1_pred.append(n1.pop(k))
                n1_target.append(gt_tmp)
                p_tmp = n2.pop(k)
                candidates = p_tmp[:,1:]
                ind = ((candidates-gt_tmp).sum(-1)==0)
                if ind.any():
                    ind = ind.nonzero()
                    n2_targets.append(torch.flatten(ind)[0])
                    n2_scores.append(p_tmp[:,0])
            if len(n1_pred) > 0:
                loss1 += torch.sqrt(torch.nn.functional.mse_loss(torch.vstack(n1_pred), torch.vstack(n1_target)))
            loss1 += random_ned_err[0]*len(t) 
            if len(n2_scores) > 0 :
                loss2 += torch.nn.functional.cross_entropy(torch.vstack(n2_scores), torch.hstack(n2_targets)) 
            else:
                loss2 += random_ned_err[1]
        return loss1 / len(targets), loss2 / len(targets)

    def loss(self, predictions, targets, **kwargs):
        ned_args = {'random_ned_err': None}
        re_args = {'no_rel_idx': None, 'random_re_err': None}
        return {
            'ner': self.ner_loss(predictions[0], targets[0]),
            'ned': self.ned_loss(
                predictions[1],
                targets[1], 
                **{k: kwargs[k] for k in ned_args.keys() & kwargs.keys()}
            ) if predictions[1] != None else torch.tensor(kwargs['random_ned_err'], device=self.dev),
            're': self.re_loss(
                predictions[2],
                targets[2],
                **{k: kwargs[k] for k in re_args.keys() & kwargs.keys()}
            ) if predictions[2] != None else torch.tensor(kwargs['random_re_err'], device=self.dev)
        }
        
# --------------------------------------------------------------------------------------------------------------------


class IEModelGoldEntities(BaseIEModelGoldEntities, IEModel):
    """
    End-to-End model for NED and RE, provided gold entities.
    """
    def __init__(self, language_model, ned_dim, KB, re_dim, device=torch.device('cpu')):
        super(BaseIEModel, self).__init__()

        # Misc
        self.sm = torch.nn.Softmax(dim=2)
        self.dev = device

        # Pretrained Language Model
        self.lang_model = PretrainedLanguageModel(language_model)

        # NED
        self.ned_dim = ned_dim
        self.NED = NEDModule(
            n_layers = 6,
            in_dim = self.lang_model.dim,
            out_dim = ned_dim,
            KB = KB,
            device = device
        )
        
        # RE
        self.RE = REModule(
            in_dim = self.lang_model.dim + ned_dim,
            out_dim = re_dim,
            h_dim = 512
        )

        # Move itself to device
        self.to(device)

    def forward(self, x, entities):
        x = self.lang_model(x)
        ctx = x[:,0].unsqueeze(1)
        x = x[:,1:]
        # get the entities
        x, positions = self.get_entities(x, entities)
        x, positions = self.PAD(x, positions)
        ned = self.NED(x, ctx)
        if x.shape[1] < 2:
            ned = (positions, ned[0], ned[1])
            return ner, ned, None
        x = torch.cat((
            x,
            (self.sm(ned[1][:,:,:,0].view(x.shape[0], -1, self.NED.n_neighbors, 1))*ned[1][:,:,:,1:]).sum(2)
            ), dim=-1)
        ned = (positions, ned[0], ned[1])
        re = self.RE(x, positions)
        return [ned, re]

    def prepare_inputs_targets(self, batch):
        inputs = [
            batch['sent'].to(self.dev),
            list(map(lambda x: x.to(self.dev), batch['pos']))
        ]
        targets = [
            list(map(lambda x: x.to(self.dev), batch['ned'])),
            list(map(lambda x: x.to(self.dev), batch['re']))
        ]
        return inputs, targets
    
    def loss(self, predictions, targets, **kwargs):
        ned_args = {'random_ned1_err': None, 'random_ned2_err': None}
        re_args = {'no_rel_idx': None, 'random_re_err': None}
        return {
            'ner': torch.tensor(0., device=self.dev),
            'ned': self.ned_loss(predictions[0], targets[0], **{k: kwargs[k] for k in ned_args.keys() & kwargs.keys()}),
            're': self.re_loss(predictions[1], targets[1], **{k: kwargs[k] for k in re_args.keys() & kwargs.keys()})
        }

# --------------------------------------------------------------------------------------------------------------------


class IEModelGoldKG(BaseIEModelGoldEntities):

    def __init__(self, language_model, ned_dim, re_dim, device=torch.device('cpu')):
        super(BaseIEModel, self).__init__()

        # Misc
        self.sm = torch.nn.Softmax(dim=2)
        self.dev = device

        # Pretrained Language Model
        self.lang_model = PretrainedLanguageModel(language_model)

        # RE
        self.RE = REModule(
            in_dim = self.lang_model.dim + ned_dim,
            out_dim = re_dim,
            h_dim = 512
        )

        # Move itself to device
        self.to(device)

    def forward(self, x, entities, embeddings):
        x = self.lang_model(x)
        x = x[:,1:]
        # get the entities
        x, positions = self.get_entities(x, entities, embeddings)
        x, positions = self.PAD(x, positions)
        re = self.RE(x, positions)
        return [re]

    def get_entities(self, x, entities, embeddings):
        ner_ned, positions = [], []
        for i, (ent, emb) in enumerate(zip(entities, embeddings)):
            ner_ned_tmp, pos_tmp = [], []
            for p, e in zip(ent, emb):
                ner_ned_tmp.append(torch.cat((
                    torch.mean(x[i][p[0]:p[1]], dim=0),
                    e
                    ), dim = -1))
                pos_tmp.append(p[1])
            ner_ned.append(torch.vstack(ner_ned_tmp).to(self.dev))
            positions.append(torch.vstack(pos_tmp).to(self.dev))
        return ner_ned, positions

    def prepare_inputs_targets(self, batch):
        inputs = [
            batch['sent'].to(self.dev),
            list(map(lambda x: x.to(self.dev), batch['pos'])),
            list(map(lambda x: x.to(self.dev), batch['emb']))
        ]
        targets = [list(map(lambda x: x.to(self.dev), batch['re']))]
        return inputs, targets

    def loss(self, predictions, targets, **kwargs):
        re_args = {'no_rel_idx': None, 'random_re_err': None}
        return {
            'ner': torch.tensor(0., device=self.dev),
            'ned': torch.tensor([0., 0.], device=self.dev),
            're': self.re_loss(predictions[0], targets[0], **{k: kwargs[k] for k in re_args.keys() & kwargs.keys()})
        }
