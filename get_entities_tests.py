import torch
import numba
from numba import cuda

# numba

@cuda.jit
def get_numba(x: torch.Tensor, entities: list):
    ner, positions = [], []
    for xx, e in zip(cuda.as_cuda_array(x), entities):
        numba_e = cuda.as_cuda_array(e)
        ner.append(
            numpy.vstack([ numpy.mean(xx[ee[0]:ee[1]], axis=0) for ee in numba_e ])
        )
        positions.append(numba_e[:,-1])
    return ner, positions

# torchscript

@torch.jit.script
def get_torchscript(x: torch.Tensor, entities: torch.Tensor):
    ner, positions = [], []
    for i in range(x.shape[0]):
        tmp = []
        for j in range(entities.shape[1]):
            if entities[i][j][0] != -1:
                tmp.append(x[i][entities[i][j][0]:entities[i][j][1]].mean(0))
        ner.append(torch.vstack(tmp))
        positions.append(entities[i][:,-1])
    return ner, positions




if __name__ == '__main__':

    x = torch.randn(2, 20, 768).cuda()
    #entities = [ torch.tensor([[1,4],[10,17]]).cuda(), torch.tensor([[2,5],[7,9],[15,19]]).cuda() ]
    entities = torch.tensor([
        [[1,4],[10,17],[-1,-1]],
        [[2,5],[7,9],[15,19]]
    ]).cuda()

    #print(get_numba(x, entities))
    
    print(get_torchscript.code)
    print(get_torchscript(x, entities))

    #traced_get = torch.jit.trace(get_torchscript, (x, entities))
    #print(traced_get.code)
    #print(traced_get(x, entities))
