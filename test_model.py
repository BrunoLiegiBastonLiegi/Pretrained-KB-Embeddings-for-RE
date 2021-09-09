from transformers import AutoTokenizer
from model import BaseIEModel, BaseIEModelGoldEntities, IEModel
from ner_schemes import BIOES
import torch

bioes = BIOES(['a','b','c'])
bert = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(bert)
m = BaseIEModelGoldEntities(bert, 2)
#m = IEModel()

s='questa e una frase di prove.'
i = tokenizer(s, return_tensors='pt')
print(m(i, [torch.tensor([[1,2],[3,5]])]))


