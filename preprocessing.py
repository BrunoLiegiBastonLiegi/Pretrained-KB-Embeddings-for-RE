import random


def split_sets(data, validation=0.2, test=0.1):
    
    val_dim = int(validation*len(data))
    test_dim = int(test*len(data))

    val = [ data.pop(random.randint(0, len(data)-1)) for i in range(val_dim) ]
    if test_dim != 0:
        test = [ data.pop(random.randint(0, len(data)-1)) for i in range(test_dim) ]

    if test_dim == 0:
        return data, val
    else:
        return data, val, test


