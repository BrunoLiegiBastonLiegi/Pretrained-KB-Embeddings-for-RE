import ngtpy, gc, pickle

# Create index
emb_dim = 200
ngtpy.create('wikidata_index', emb_dim)
index = ngtpy.Index('wikidata_index')

# Fill the index
for n, i in enumerate(('id2emb_part_1.pkl', 'id2emb_part_2.pkl', 'id2emb_part_3.pkl', 'id2emb_part_4.pkl')):
    print('> Loading graph embeddings part {} from file {}.'.format(n+1, i))
    with open(i, 'rb') as f:
        id2emb = pickle.load(f)
    print('> Filling the index')
    for v in id2emb.values():
        index.insert(v)
    del id2emb
    gc.collect()

index.build_index() # build index
index.save() # save the index

