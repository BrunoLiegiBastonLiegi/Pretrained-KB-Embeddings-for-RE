import ngtpy, pickle

# Create index
emb_dim = 200
ngtpy.create('kb_index', emb_dim, distance_type='Cosine')
index = ngtpy.Index('kb_index')

# Fill the index
ids = {}
for i in ('fewrel_train.pkl', 'fewrel_val.pkl'):
    with open(i, 'rb') as f:
        d = pickle.load(f)
    print('> Filling the index')
    for s in d:
        for e in s['entities'].values():
            try:
                flag = e['embedding'].any() != None
            except:
                flag = False
            if flag:
                try:
                    ids[e['id']] += 1
                except:
                    ids[e['id']] = 0
                    index.insert(e['embedding'])

# Optimize
optimizer = ngtpy.Optimizer(log_disabled = True)
optimizer.set_processing_modes(search_parameter_optimization=True, prefetch_parameter_optimization=True, accuracy_table_generation=True)
optimizer.optimize_number_of_edges_for_anng('kb_index')

index.build_index() # build index
index.save() # save the index
