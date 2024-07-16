from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from . import optimizers
from . import utils
from . import cell_selection
from . import preprocess as pp
from .cell_selection import CellSelectionSolver
from .gradient_descent import GradientDescentSolver

import time
import sys
import logging
logger = logging.getLogger()
if logger.level != logging.DEBUG:
    logger.setLevel(level = logging.DEBUG)
    logformatter = logging.Formatter('[%(asctime)s/%(levelname)s] %(message)s')
    logfile = logging.FileHandler("spexmod.log", mode='a')
    logfile.setFormatter(logformatter)
    logfile.setLevel(logging.DEBUG)
    logger.addHandler(logfile)
    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(logformatter)
    console.setLevel(logging.DEBUG)
    logger.addHandler(console)
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

import pandas as pd
import numpy as np
import os
import cProfile
import pdb
from typing import Dict, List
'''
@author: Jingwan WANG
Created on 2022/11/09
See also CHANGELOG.txt
        
st_tp: choose among either visum, st or slide-seq
'''

class SpaMint:
    # 简单数据成员
    save_path: str # 保存输出的地方
    cell_type_key: str # sc_meta中表明细胞类型的那列
    st_tp: str # 测序方法 只有某几种取值
    nprocess: int # 涉及多进程操作时的进程数

    # DataFrame成员
    weight: pd.DataFrame # spot x cellType; 每种细胞在spot中的比重(st_decon)
    st_exp: pd.DataFrame # spot x gene 每个基因在Spot中表达程度
    st_coord: pd.DataFrame # spot x (x,y) Spot的坐标
    sc_ref: pd.DataFrame # cell x gene
    sc_exp: pd.DataFrame # cell x gene 每个基因在细胞中表达程度
    sc_meta: pd.DataFrame # cell x unknown 细胞的基本信息 与cell_type_key配合使用
    lr_df: pd.DataFrame # list(L,R,score) 计算Affinity需要的LR配对表
    svg: pd.DataFrame

    # TODO: 计算时产生的中间量成员
    spots_nn_lst: Dict[str, List[str]] # 每个spot的邻近spot 防止重复计算
    st_aff_profile_df: pd.DataFrame # spot x spot x aff_gene
    sc_agg_meta: pd.DataFrame # cell x [...] 主要指明cell位于的spot
    lr_df_align: pd.DataFrame
    alter_sc_exp: pd.DataFrame

    def __init__(self, save_path = None, st_adata = None, weight = None, 
                 sc_ref = None, sc_adata = None, cell_type_key = 'celltype', lr_df = None, 
                 st_tp = 'st'
                 ):
        self.save_path = save_path +'/'
        self.st_adata = st_adata
        self.weight = weight
        self.sc_ref = sc_ref
        self.sc_adata = sc_adata
        self.cell_type_key = cell_type_key
        self.lr_df = lr_df
        self.st_tp = st_tp
        self._prep()
        logger.debug("SpaMint object created.")


    def _check_input(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        utils.check_st_tp(self.st_tp)
        self.st_adata, self.sc_adata, self.sc_ref, self.weight = utils.check_index_str(self.st_adata, self.sc_adata, self.sc_ref, self.weight)
        self.st_adata, self.weight = utils.check_spots(self.st_adata,self.weight)
        self.st_coord = utils.check_st_coord(self.st_adata)
        self.lr_df = utils.align_lr_gene(self)
        utils.check_st_sc_pair(self.st_adata, self.sc_adata)
        self.sc_adata,self.sc_ref = utils.check_sc(self.sc_adata, self.sc_ref)
        utils.check_decon_type(self.weight, self.sc_adata, self.cell_type_key)
        logger.debug('Parameters checked!')


    def _prep(self):
        ######### init ############
        # 1. check input and parameters
        self._check_input()
        # 2. creat obj
        self.st_exp = self.st_adata.to_df()
        self.sc_exp = self.sc_adata.to_df()
        self.sc_meta = self.sc_adata.obs.copy()
        del self.sc_adata
        # 3. generate obj
        # TODO redundant with feature selection in init
        self.svg = optimizers.get_hvg(self.st_adata)
        del self.st_adata
        logger.debug('Getting svg genes')
        # 4. remove no neighbor spots
        spots_nn_lst = optimizers.findSpotKNN(self.st_coord,self.st_tp)
        # TODO 可能会没有empty spots需要del
        empty_spots = [k for k, v in spots_nn_lst.items() if not v]
        if empty_spots:
            # empty_spots is not empty
            #remove from spots_nn_lst is key have empty value, which means spot have no neighbor
            spots_nn_lst = {k: v for k, v in spots_nn_lst.items() if v}
            #remove empty spots from st_exp
            self.st_exp = self.st_exp.drop(empty_spots)
            self.st_coord = self.st_coord.drop(empty_spots)
            self.weight = self.weight.drop(empty_spots)
        self.spots_nn_lst = spots_nn_lst
        logger.debug("Calculating spots affinity profile data")
        self.st_aff_profile_df = optimizers.cal_aff_profile(self.st_exp, spots_nn_lst, self.lr_df)


    def select_cells(self, use_sc_orig = True, p = 0.1, mean_num_per_spot = 10, mode = 'strict', max_rep = 3, repeat_penalty = 10):
        solver = CellSelectionSolver(self, use_sc_orig, p, mean_num_per_spot, mode, max_rep, repeat_penalty)
        self.select_cells_solver = solver
        solver.solve()
        result = solver.result
        self.lr_df_align = solver.lr_df_align

        # 6. save result
        self.sc_agg_meta = result
        return result


    def gradient_descent(self, alpha, beta, gamma, delta, eta, 
                init_sc_embed = False,
                iteration = 20, k = 2, W_HVG = 2,
                left_range = 1, right_range = 2, steps = 1, dim = 2):
        solver = GradientDescentSolver(self, alpha, beta, gamma, delta, eta, 
                init_sc_embed, iteration, k, W_HVG, left_range, right_range, steps, dim)
        self.gradient_descent_solver = solver
        self.alter_sc_exp, self.sc_agg_meta = solver.gradient_descent()
        self.sc_knn = solver.sc_knn
        return self.alter_sc_exp, self.sc_agg_meta