import torch, random, time, faiss
from torch.utils.data import DataLoader



class Trainer(object):

    def __init__(self, train_data, test_data, model, optim, rel2index, device, gold_entities=False, save=True, wNED=1, batchsize=32, tokenizer=None):
        self.model = model
        self.optim = optim
        self.rel2index = rel2index
        self.device = device     
        self.train_set = train_data
        self.test_set = test_data
        self.gold = gold_entities
        self.save = save
        self.wNED = wNED
        self.crossentropy = torch.nn.CrossEntropyLoss()
        self.cosine = torch.nn.CosineEmbeddingLoss()
        self.mse = torch.nn.MSELoss(reduction='mean')
        self.batchsize = batchsize
        #self.random_ner_err, self.random_ned_err, self.random_re_err = self.task_rand_err()
        self.tokenizer=tokenizer

    def task_rand_err(self, samples=100):
        try:
            p, t = torch.randn(samples, self.model.ner_dim), torch.randint(0, self.model.ner_dim, (samples,))
            ner_err = self.crossentropy(p,t)
        except:
            ner_err = 0
        p, t = torch.randn(samples, self.model.re_dim), torch.randint(0, self.model.re_dim, (samples,))
        re_err = self.crossentropy(p,t)
        rand_ned = self.mean_emb_dist()
        return ner_err, rand_ned, re_err   
        
    def mean_emb_dist(self):
        """
        samples = random.choices(self.train_set, k=10)
        samples = torch.vstack([ s['ned'][:,1:] for s in samples ]).to(self.device)
        mean = 0
        
        for i, s in enumerate(samples):
            for t in samples:
                if (s-t).sum(-1) != 0:
                    mean += self.mse(s,t)
                    #mean += self.cosine(s.view(1,-1), t.view(1,-1), torch.ones(1, device=self.device))
                print('> Building estimate of random graph-embedding error. ({}%)'.format(int(i/len(samples)*100)), end='\r')
        print('\n> Done.')
        #return torch.sqrt(mean) / (samples.shape[0]-1)**2
        return mean / (samples.shape[0]-1)**2
        """
        kg = torch.vstack(list(map(lambda x: x['ned'][:,1], self.train_set)))
        quantizer = faiss.IndexFlatL2(kg[0].shape[-1])
        nn = faiss.IndexIVFFlat(quantizer, kg[0].shape[-1], 100)
        assert not nn.is_trained
        nn.train(kg.numpy())
        assert nn.is_trained
        nn.add(kg.numpy())
        nn.nprobe = 1
        sample = random.choices(kg, k=100).view(100,-1)
        d, _ = list(zip(*list(map(nn.search, sample, repeat(10,100)))))
        return numpy.vstack(d).mean()

    def train(self, epochs):

        # Set model in train mode
        self.model.train()
        # Gradient scaler for automatic mixed precision
        scaler = torch.cuda.amp.GradScaler()

        train_loader = DataLoader(self.train_set,
                                      batch_size=self.batchsize,
                                      shuffle=True,
                                      collate_fn=self.train_set.collate_fn)
        # BERT layers unfreezing
        k = 0  # counter for bert layers unfreezing
        one_3rd = int(len(train_loader) / 3) # after 1/3 of the data we unfreeze a layer
        print_step = int(len(train_loader) / 6)
        # loss plots
        plots = {
            'train':{
                'ner':[],
                'ned1':[],
                'ned2':[],
                're':[]
            },
            'test':{
                'ner':[],
                'ned1':[],
                'ned2':[],
                're':[]
            } 
        }
        # Loss weights
        l = 0. if not self.gold else 1.# RE loss weight, gradually increased to 1
        
        for epoch in range(epochs):

            running_loss = 0.0
            ner_running_loss = 0.0
            ned_running_loss1 = 0.0
            ned_running_loss2 = 0.0
            re_running_loss = 0.0
            avg_iter_time = 0.
            step_t1 = None
            # set model in train mode
            self.model.train()
            
            print_step = int(len(train_loader) / 6)
            
            for i, batch in enumerate(train_loader):
                if step_t1 == None:
                    step_t1 = time.time()
                it_t1 = time.time()
                if k < 4:
                    if epoch == 0:
                        if i >= one_3rd and k == 0:
                            self.model.lang_model.unfreeze_layer(k) # gradually unfreeze the last layers
                            k += 1
                        elif i >= 2*one_3rd and k == 1:
                            self.model.lang_model.unfreeze_layer(k) # gradually unfreeze the last layers
                            k += 1
                    elif epoch == 1:
                        if k == 2:
                            self.model.lang_model.unfreeze_layer(k) # gradually unfreeze the last layers
                            k += 1
                        if i >= one_3rd and k == 3:
                            self.model.lang_model.unfreeze_layer(k) # gradually unfreeze the last layers
                            k += 1

                # zero the parameter gradients
                self.optim.zero_grad()
                #t1 = time.time()
                with torch.cuda.amp.autocast():
                    # forward
                    losses = self.step(batch)
                    #t2 = time.time()
                    #print('> Forward:', t2-t1)
                    # unfreeze NED and RE training
                    if epoch == 1 and not self.gold:
                        l = i / len(train_loader)
                    loss = losses['ner'] + l * losses['re'] + self.wNED*(l * (losses['ned'][0] + losses['ned'][1])) # wNED is used for discovering the benefit of NED
                #t1 = time.time()
                # save train losses
                #for v, j in zip(plots['train'].values(), [ner_loss, ned_loss1, ned_loss2, re_loss]):
                #    try:
                #        v.append(j.item())
                #    except:
                #        v.append(j)
                # backprop
                scaler.scale(loss).backward()
                #loss.backward()
                #t2 = time.time()
                #print('> BackProp:', t2-t1)
                # optimize
                #self.optim.step()
                scaler.step(self.optim)
                # Updates the scale for next iteration.
                scaler.update()

                # print statistics
                ner_running_loss += losses['ner'] #ner_loss.item()
                ned_running_loss1 += losses['ned'][0] #ned_loss1.item()
                ned_running_loss2 += losses['ned'][1] #ned_loss2.item()
                re_running_loss += losses['re'] #re_loss.item()
                running_loss += loss.item()
                it_t2 = time.time()
                avg_iter_time += it_t2 - it_t1

                if i % print_step == print_step - 1:    # print every print_step sentences
                    step_t2 = time.time()
                    print('[%d, %5d] Total loss: %.3f, NER: %.3f, NED1: %.3f, NED2: %.3f, RE: %.3f \t Total time: %.2f (%.2f it/s)' %
                          (epoch + 1, i*self.batchsize + 1, running_loss / print_step, ner_running_loss / print_step, ned_running_loss1 / print_step, ned_running_loss2 / print_step, re_running_loss / print_step, step_t2-step_t1, 1/(avg_iter_time / print_step)))
                    running_loss = 0.
                    ner_running_loss = 0.
                    ned_running_loss1 = 0.
                    ned_running_loss2 = 0.
                    re_running_loss = 0.
                    avg_iter_time = 0.
                    step_t1 = None
                    
            test_loss = self.test_loss()
            #for v, j in zip(plots['test'].values(), test_loss[1:]):
            #        v.append(j.item())
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
        return plots

    def step(self, batch):
        #inputs = batch['sent']
        #if self.gold:
         #   entities = batch['pos']
        #print(self.tokenizer.decode(inputs['input_ids'][0]))
        #ner_target = batch['ner']
        #print(ner_target[0])
        #ned_target = batch['ned']
        #re_target = batch['re']
        inputs, targets = self.model.prepare_inputs_targets(batch)

        # move inputs and labels to device
        #if self.device != torch.device("cpu"):
        #    inputs = inputs.to(self.device)
        #    ner_target = ner_target.to(self.device)
        #    ned_target = [ t.to(self.device) for t in ned_target ]
        #    re_target = [ t.to(self.device) for t in re_target ]

        # forward
        #if self.gold:
        #   ned_output, re_output = self.model(inputs, entities)
        #   ner_loss = torch.tensor(0., device=self.device)
        #else:
        #    ner_out, ned_output, re_output = self.model(inputs)
        outs = self.model(*inputs)
        return self.model.loss(
            outs,
            targets,
            no_rel_idx = self.rel2index['NO_RELATION'],
            random_ned_err = [1., 1.],
            random_re_err = 1.#self.random_re_err
        )
            # losses
        #    ner_loss = self.crossentropy(
        #        torch.transpose(ner_out, 1, 2),
        #        ner_target
        #    )
        #ned_loss1, ned_loss2 = self.NED_loss(ned_output, ned_target) if ned_output != None else (self.random_ned_err, torch.tensor(2.3, device=self.device))
        #re_loss = self.RE_loss(re_output, re_target) if re_output != None else torch.tensor(2.3, device=self.device)
        #return ner_loss, ned_loss1, ned_loss2, re_loss
        
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
                losses = self.step(batch)
                loss += losses['ner'] + losses['re'] + (1*losses['ned'][0] + losses['ned'][1]) 
                test_ner_loss += losses['ner'] #ner_loss.item()
                test_ned_loss1 += losses['ned'][0] #ned_loss1.item()
                test_ned_loss2 += losses['ned'][1] #ned_loss2.item()
                test_re_loss += losses['re'] #re_loss.item()
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
                re_target.append(g.pop(k))
            if len(g) > 0:
                loss += torch.tensor(2.3, device=self.device)
            if self.model.training:
                for k,v in r.items():
                    if -1 not in k:
                        re_pred.append(v)
                        try:
                            re_target.append(torch.tensor(self.rel2index['NO_RELATION'], dtype=torch.int, device=self.device))
                        except:
                            re_target.append(torch.tensor(self.rel2index['no_relation'], dtype=torch.int, device=self.device))
            if len(re_pred) > 0:
                loss += self.crossentropy(torch.vstack(re_pred), torch.hstack(re_target).long())
                
        return loss / len(groundtruth)
        
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
            n1_pred, n1_target = [], []
            for k in g.keys() & n1.keys():
                gt_tmp = g.pop(k)
                #loss1 += torch.sqrt(self.mse(n1.pop(k), gt_tmp))
                n1_pred.append(n1.pop(k))
                n1_target.append(gt_tmp)
                p_tmp = n2.pop(k)
                candidates = p_tmp[:,1:]
                ind = ((candidates-gt_tmp).sum(-1)==0)
                if ind.any():
                    ind = ind.nonzero()
                    n2_targets.append(torch.flatten(ind)[0])
                    n2_scores.append(p_tmp[:,0])
            loss1 += torch.sqrt(self.mse(torch.vstack(n1_pred), torch.vstack(n1_target)))
            #loss1 += self.cosine(torch.vstack(n1_pred), torch.vstack(n1_target), torch.ones(len(n1_pred), device=self.device))
            #loss1 += self.random_ned_err*len(g) 
            if len(n2_scores) > 0 :
                loss2 += self.crossentropy(torch.vstack(n2_scores), torch.hstack(n2_targets)) 
            else:
                loss2 += torch.tensor(2.3, device=self.device)
            """
            if self.model.training:
                try:
                    n1.pop(-1)
                except:
                    pass
                #loss1 += sum(map(self.mse, n1.values(), torch.zeros(len(n1), dim, device=self.device)))
                if len(n1) > 0:
                    loss1 += self.cosine(torch.vstack(list(n1.values())), torch.zeros(len(n1), dim, device=self.device), torch.ones(len(n1)))
            """
        return loss1 / len(groundtruth), loss2 / len(groundtruth)
        
