
from typing import List, Tuple, Dict, Iterator
import numpy as np
import numpy.random as rnd
from itertools import product


def bin2bip(x):
    '''
    conversion binary [0,1] -> bipolar [-1,+1] 
    '''
    return 2. * x - 1.


def bip2bin(x):
    '''
    conversion bipolar [-1,+1] -> binary [0,1]
    '''
    return (x + 1.) / 2.


def bip2str(s: List[int]) -> str:
    """
    0 0 1 1 0 0 1     -> --++--+
    -1 -1 1 1 -1 1 -1 -> --++-+-
    """
    symbols = {+1: '+',
               -1: '-',
               0: '-'}
    return ' '.join([symbols[x] for x in s])


def int_to_binary_vector(x: int, n: int) -> np.ndarray:
    """
    Given an integer x, returns x as a binary vector of length n. Pads 0 to the right.
    """
    return np.array([(x >> i) & 1 for i in range(n)])


def yield_binary_masks(n) -> Iterator[np.ndarray]:
    for mask in product((0, 1), repeat=n):
        yield np.array(list(mask)).astype(bool)


def binary_matrix(n) -> np.ndarray:
    """
    Returns a matrix of all binary vectors of length n.
    Output shape: [2**n, n]

    Example:
    binary_matrix(3):
    [[0, 0, 0],
     [1, 0, 0],
     [0, 1, 0],
     [1, 1, 0],
     [0, 0, 1],
     [1, 0, 1],
     [0, 1, 1],
     [1, 1, 1]]
    """
    if n < 1:
        return None

    Z = np.array([[0],
                  [1]])

    for i in range(1,n):
        z = np.zeros((2**i, 1))
        o = np.ones ((2**i, 1))
        Z = np.vstack((np.hstack((Z,z)),
                       np.hstack((Z,o))))

    return Z


def bipolar_matrix(n):
    """
    Same a above as bipolar matrix.
    """
    return bin2bip(binary_matrix(n))


def signum(x):
    """
    Array-wise sign function.
    [2, 3, -4, -5] -> [1, 1, -1, -1]
    """
    return np.where(x >= 0, +1, -1)

#
# functions to manipulates data 
#
def binarize_wrt_mean(x):
    return np.where(x > np.mean(x), 1, 0)


def distort_bip_vector(x, p=0.2):
    return - x * bin2bip(rnd.binomial(1, p, len(x)))


if __name__ == '__main__':
    pass
