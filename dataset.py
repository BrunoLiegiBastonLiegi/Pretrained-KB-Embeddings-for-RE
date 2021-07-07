import pickle, torch
from transformers.tokenization_utils_base import BatchEncoding


class IEData(torch.utils.data.Dataset):

    def __init__(self, sentences, ner_labels, re_labels, preprocess=False, ned_labels=None, tokenizer=None, ner_scheme=None, rel2index=None, save_to=None):
        self.tokenizer = tokenizer
        self.scheme = ner_scheme
        self.rel2index = rel2index
        self.samples = []
        self.pad = self.tokenizer('[PAD]', add_special_tokens=False)['input_ids'][0]
        if preprocess:
            print('> Preprocessing labels.')
            assert ner_scheme != None, 'Missing NER scheme for preprocessing.'
            for i, s, ner, re in zip(range(len(sentences)), sentences, ner_labels, re_labels):
                self.samples.append(self.generate_labels(s, ner, re))
                print('{} / {} ({}%)'.format(i, len(sentences), int(i/len(sentences)*100)), end='\r')
        else:
            assert ned_labels != None, 'Missing NED labels.'
            for s, ner, ned, re in zip(sentences, ner_labels, ned_labels, re_labels):
                self.samples.append({
                    'sent': s,
                    'ner': ner,
                    'ned': ned,
                    're': re
                })
        print('\n> Done.')
        if save_to != None:
            print('> Saving to \'{}\'.'.format(save_to))
            with open(save_to, 'wb') as f:
                pickle.dump(self.samples, f)

    def generate_labels(self, s, ner, re):
        s_tk = self.tokenizer(s, return_tensors='pt', add_special_tokens=False)['input_ids']
        names, span2span, spans, types = {}, {}, [], []
        for k, e in ner.items():
            try:
                names[e['name']] += 1
            except:
                names[e['name']] = 0
            tk = self.tokenizer(e['name'], add_special_tokens=False)['input_ids']
            print(s_tk)
            print(tk)
            span = self.find_span(s_tk.flatten().tolist(), tk, names[e['name']]) 
            span2span[k] = span
            spans.append(span)
            types.append(e['type'])
        lab = {
            'sent': self.tokenizer(s, return_tensors='pt')['input_ids'],
            'ner': self.tag_sentence(s_tk.flatten().tolist(), types, spans),
            'ned': torch.vstack([
                torch.hstack((
                    torch.tensor(spans[i][1]),
                    torch.mean(e['embedding'], dim=0) # need mean for multi-concept entities
                ))
                for i, e in enumerate(ner.values())
            ]),
            're': torch.vstack([
                torch.tensor([
                    span2span[k[0]][1],
                    span2span[k[1]][1],
                    self.rel2index[r]
                ])
                for k, r in re.items()
            ])
            }
        return lab
            

    def find_span(self, sent, ent, n=0):
        """
        Find the span of the entity in the tokenization scheme provided by the tokenizer.
        We consider only the case of the same entity occuring just once for each sentence.
        """
        match = []
        #print(sent)
        #print(self.tokenizer.decode(sent))
        #print(ent)
        #print(self.tokenizer.decode(ent))
        for i in range(len(sent)):
            if sent[i] == ent[0] and sent[i:i+len(ent)] == ent: 
                #match = (i, i+len(ent))
                match.append((i, i+len(ent)))
        return match[n]

    def tag_sentence(self, sent, types, spans):
        tags = torch.tensor([self.scheme.to_tensor('O', index=True) for i in range(len(sent))])
        for t,s in zip(types, spans):
            if s[1]-s[0] == 1:
                tags[s[0]] = self.scheme.to_tensor('S-' + t, index=True)
            else:
                tags[s[0]:s[1]] = self.scheme.to_tensor(*(['B-' + t] + [('I-' + t) for j in range((s[1]-s[0])-2)] + ['E-' + t]), index=True)
        return tags.view(1,-1)
                
    def collate_fn(self, batch):
        """
        Function to vertically stack the batches needed by the torch.Dataloader class
        """
        # alternatively I could use the huggingface tokenizer with option pad=True
        tmp = {'sent':[], 'ner':[], 'ned':[], 're':[]} # we need to add padding in order to vstack the sents.
        max_len = 0
        for item in batch:
            #max_len = max(max_len, item['sent'][:,:-1].shape[1]) # -1 for discarding [SEP]
            max_len = max(max_len, item['sent'].shape[1])
            #tmp['sent'].append(item['sent'][:,:-1])
            tmp['sent'].append(item['sent'])
            tmp['ner'].append(item['ner'])
            tmp['ned'].append(item['ned'])
            tmp['re'].append(item['re'])

        sent = {'input_ids': [], 'attention_mask': []}
        sent['input_ids'] = torch.vstack(list(map(
            lambda x: torch.hstack((x, self.pad*torch.ones(1, max_len - x.shape[1]).int())),
            tmp['sent']
        )))
        sent['attention_mask'] = torch.vstack(list(map(
            lambda x: torch.hstack((torch.ones(1, x.shape[1]), torch.zeros(1, max_len - x.shape[1]))).int(),
            tmp['sent']
        )))
        tmp['sent'] = BatchEncoding(sent)
        # add the [SEP] at the end
        #tmp['sent'] = torch.hstack((tmp['sent'], self.sep*torch.ones(tmp['sent'].shape[0],1).int()))
        O = self.scheme.to_tensor('O', index=True)
        # -1 because max_len counts also the [CLS]
        #print(max_len)
        #for i in tmp['ner']:
         #   print(i.shape[1] ,max_len -2 -i.shape[1])
        tmp['ner'] = torch.vstack(list(map(
            #lambda x: torch.hstack((x, O*torch.ones(1, max_len - 1 - x.shape[1]).int())),
            lambda x: torch.hstack((x, O*torch.ones(1, max_len - 2 - x.shape[1]).int())),
            tmp['ner']
        )))
        return tmp
            
    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)
