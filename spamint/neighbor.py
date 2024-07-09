# 本文件均为求出仅与Neighbor spot/one-hop neighbor cell有关的值
# 包括距离矩阵、Affinity矩阵、KNN等
# 这种做法的好处是矩阵都是稀疏的，节约内存

import pandas as pd
import numpy as np
from scipy.special import digamma
from scipy.spatial import distance_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.spatial import KDTree
from scipy import sparse
import scanpy as sc
from sklearn.metrics import mean_squared_error
import time
import logging

import umap
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from . import utils
import pdb
import multiprocessing
import cProfile

def neighbor_cell_distance_matrix(
        sc_coord: pd.DataFrame,
        spots_nn_lst: dict[str, list[str]],
        spot_cell_dict: dict[str, list[str]]):
    '''
    类似`distance_matrix(sc_coord, sc_coord)`，但返回sparse matrix

    只计算每个cell与自己neighbor spot中细胞的距离，其他留0
    '''
    count = len(sc_coord.index)
    ret = lil_matrix( (count, count) )
    for spot in spots_nn_lst:
        nn_spots = spots_nn_lst[spot]
        nn_cells = []
        for sp in nn_spots:
            nn_cells += spot_cell_dict[sp]
    
    return ret.tocsr()
