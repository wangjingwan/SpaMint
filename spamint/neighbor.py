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

def neighbor_distance_matrix(sc_coord, sc_nn_list):
    '''
    类似distance_matrix，但返回sparse matrix

    只计算每个cell与自己neighbor spot中细胞的距离，其他留0

    `sc_coord`: `pd.DataFrame` @ cell x (x,y)
    `sc_nn_list`: `dict[str, list[str]]` cell x cell
    '''
    pass
