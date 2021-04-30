import pickle, sys
import numpy as np

files = sys.argv[1:]

f1 = {}
for i in {'NER', 'NED', 'RE'}:
    f1[i] = {'micro': [], 'macro': []}
for i in files:
    with open(i, 'rb') as f:
        r = pickle.load(f)
    for j in {'NER', 'NED'}:
        f1[j]['micro'].append(r[j]['micro avg']['f1-score']) 
        f1[j]['macro'].append(r[j]['macro avg']['f1-score'])
    f1['RE']['micro'].append(r['RE']['accuracy'])                 # for some reason in RE we don;t find micro avg but accuracy
    f1['RE']['macro'].append(r['RE']['macro avg']['f1-score'])    # maybe cause it's a binary class. task

# Calculate mean and variance
print('\t Micro F1 \t\t Macro F1\n')
mean = {}
var = {}
for i in {'NER', 'NED', 'RE'}:
    mean[i] = {'micro': np.mean(f1[i]['micro']), 'macro': np.mean(f1[i]['macro'])}
    var[i] = {'micro': np.var(f1[i]['micro']), 'macro': np.var(f1[i]['macro'])}
    print(i, '\t', '%.4f +- %.4f \t %.4f +- %.4f' % (mean[i]['micro'],var[i]['micro'], mean[i]['macro'], var[i]['macro']))
