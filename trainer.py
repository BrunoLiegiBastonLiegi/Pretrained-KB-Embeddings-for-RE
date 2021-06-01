import torch
from torch.utils.data import DataLoader



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

        # set model in train mode
        self.model.train()

        train_loader = DataLoader(self.train_set,
                                      batch_size=self.batchsize,
                                      shuffle=True,
                                      collate_fn=self.train_set.collate_fn)
        # BERT layers unfreezing
        k = 0  # counter for bert layers unfreezing
        one_3rd = int(len(train_loader) / 3) # after 1/3 of the data we unfreeze a layer
        print_step = int(len(train_loader) / 6)
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

                if epoch == 1:
                    l = i / len(train_loader) 
                loss = ner_loss + l * re_loss + self.wNED*(l * (1*ned_loss1 + ned_loss2)) # wNED is used for discovering the benefit of NED
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
        ner_loss = self.crossentropy(
            torch.transpose(ner_out, 1, 2),
            ner_target
        )
        ned_loss1, ned_loss2 = self.NED_loss(ned_output, ned_target) if ned_output != None else (torch.tensor(1, device=self.device), torch.tensor(2.3, device=self.device))
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
                loss += ner_loss + re_loss + (1*ned_loss1 + ned_loss2) 
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
                re_target.append(g.pop(k))
            if len(g) > 0:
                loss += torch.tensor(2.3, device=self.device)
            if self.model.training:
                for k,v in r.items():
                    if -1 not in k:
                        re_pred.append(v)
                        re_target.append(torch.tensor(0, dtype=torch.int, device=self.device)) # 0 for no relation
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
            for k in g.keys() & n1.keys():
                gt_tmp = g.pop(k)
                loss1 += torch.sqrt(self.mse(n1.pop(k), gt_tmp))
                p_tmp = n2.pop(k)
                candidates = p_tmp[:,1:]
                if gt_tmp in candidates:
                    ind = ((candidates-gt_tmp).sum(-1)==0).nonzero()
                    n2_targets.append(torch.flatten(ind)[0])
                    #n2_targets.append(ind.view(1))
                    n2_scores.append(p_tmp[:,0])
            loss1 += 0.1*len(g)
            if len(n2_scores) > 0 :
                loss2 += self.crossentropy(torch.vstack(n2_scores), torch.hstack(n2_targets)) 
            else:
                loss2 += torch.tensor(2.3, device=self.device)
            if self.model.training:
                try:
                    n1.pop(-1)
                except:
                    pass
                loss1 += sum(map(self.mse, n1.values(), torch.zeros(len(n1), dim, device=self.device)))

        return loss1 / len(groundtruth), loss2 / len(groundtruth)
        
