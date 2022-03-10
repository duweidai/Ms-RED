import torch
import numpy as np
from torch import einsum
from torch import Tensor
from scipy.ndimage import distance_transform_edt as distance
from scipy.spatial.distance import directed_hausdorff

from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union

# switch between representations
def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)

    return res

def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, _, _ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))

    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res


def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    assert one_hot(torch.Tensor(seg), axis=0)
    C: int = len(seg)

    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            # print('negmask:', negmask)
            # print('distance(negmask):', distance(negmask))
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
            # print('res[c]', res[c])
    return res


def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

    # Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

class SurfaceLoss():
    def __init__(self):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = [1]   #这里忽略背景类  https://github.com/LIVIAETS/surface-loss/issues/3

    # probs: bcwh, dist_maps: bcwh
    def __call__(self, probs: Tensor, dist_maps: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        #print('pc', pc)
        #print('dc', dc)

        multipled = einsum("bcwh,bcwh->bcwh", pc, dc)

        loss = multipled.mean()

        return loss


if __name__ == "__main__":
    data = torch.tensor([[[[0, 0, 0, 0, 0, 0, 0], 
                          [0, 1, 1, 0, 0, 0, 0],
                          [0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0]]], 
                          
                          [[[0, 0, 0, 0, 0, 0, 0], 
                          [0, 1, 1, 1, 0, 0, 0],
                          [0, 1, 1, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0]]]])   # [1, 4, 7]
                          
    b, c, h, w = data.shape
    bd_map = torch.zeros([b, 2, h, w])                      
    for i in range(data.shape[0]):
        data_2 = data[i]
        data_3 = class2one_hot(data_2, 2)  # [1, 2, 4, 7]
        data_4 = data_3[0].numpy()
        data_5 = one_hot2dist(data_4)       
        bd_map[i] = torch.tensor(data_5)
        
    logits = torch.tensor([[[[0, 0, 0, 0, 0, 0, 0], 
                             [0, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]]],
                             
                             [[[0, 0, 0, 0, 0, 0, 0], 
                             [0, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]]]])  # [1,4,7]
    
    pd_map = torch.zeros([b, 2, h, w])
    
    for i in range(logits.shape[0]):
        pd_1 = logits[i]
        pd_2 = class2one_hot(pd_1, 2)
        pd_map[i] = pd_2
                  
    Loss = SurfaceLoss()
    res = Loss(pd_map, bd_map, None)
    print('loss:', res)