import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from scipy.spatial import KDTree
from scipy.special import digamma
from scipy.spatial import distance_matrix
from scipy.sparse import lil_matrix
from scipy import sparse
import scanpy as sc
from sklearn.metrics import mean_squared_error

from loess.loess_1d import loess_1d
REALMIN = np.finfo(float).tiny
from . import optimizers
from . import utils
from . import preprocess as pp
import pdb
import cProfile
import multiprocessing

class GradientDescentSolver:
    st_exp: pd.DataFrame # spot x gene 每个基因在Spot中表达程度
    st_coord: pd.DataFrame # spot x (x,y) Spot的坐标
    sc_ref: pd.DataFrame # cell x gene
    sc_exp: pd.DataFrame # cell x gene 每个基因在细胞中表达程度
    sc_meta: pd.DataFrame # cell x unknown 细胞的基本信息 与cell_type_key配合使用
    lr_df: pd.DataFrame # id x (L,R,score) 计算Affinity需要的LR配对表

    # 从cell select继承
    sc_agg_meta: pd.DataFrame
    lr_df_align: pd.DataFrame

    # 原先的传参（略）

    # 中间产物
    
    def __init__(self, spex, alpha, beta, gamma, delta, eta, 
                init_sc_embed = False,
                iteration = 20, k = 2, W_HVG = 2,
                left_range = 1, right_range = 2, steps = 1, dim = 2):
        self.st_exp = spex.st_exp
        self.st_coord = spex.st_coord
        self.sc_exp = spex.sc_exp
        self.sc_ref = spex.sc_ref
        self.lr_df = spex.lr_df
        self.st_tp = spex.st_tp
        self.sc_meta = spex.sc_meta
        self.cell_type_key = spex.cell_type_key
        self.save_path = spex.save_path
        self.spots_nn_lst = spex.spots_nn_lst
        self.st_aff_profile_df = spex.st_aff_profile_df
        self.lr_df_align = spex.lr_df_align
        self.sc_agg_meta = spex.sc_agg_meta
        self.alter_sc_exp = self.sc_exp.loc[self.sc_agg_meta['sc_id']]
        self.alter_sc_exp.index = self.sc_agg_meta.index
        self.spot_cell_dict = self.sc_agg_meta.groupby('spot').apply(optimizers.apply_spot_cell).to_dict()
        self.svg = spex.svg

        self.ALPHA = alpha
        self.BETA = beta
        self.GAMMA = gamma
        self.DELTA = delta
        self.ETA = eta

        self.init_sc_embed = init_sc_embed
        self.iteration = iteration
        self.K = k
        self.W_HVG = W_HVG
        self.left_range = left_range
        self.right_range = right_range
        self.steps = steps
        self.dim = dim

    def run_gradient(self):
        # 1. First term
        self.term1_df,self.loss1 = optimizers.cal_term1(self.alter_sc_exp,self.sc_agg_meta,self.st_exp,self.svg,self.W_HVG)
        print('First-term calculation done!')

        # 2. Second term
        self.term2_df,self.loss2 = optimizers.cal_term2(self.alter_sc_exp,self.sc_ref)
        print('Second-term calculation done!')

        # 3. Third term, closer cells have larger affinity
        if not (self.st_tp == 'slide-seq' and hasattr(self, 'sc_knn')):
            # if slide-seq and already have found sc_knn
            # dont do it again
            self.sc_dist = pd.DataFrame(distance_matrix(self.sc_coord,self.sc_coord), index = self.alter_sc_exp.index, columns = self.alter_sc_exp.index)
            # 3.2 get c' = N(c)
            self.sc_knn = optimizers.findCellKNN(self.st_coord,self.st_tp,self.sc_agg_meta,self.sc_coord,self.K)
            utils.check_empty_dict(self.sc_knn)

        # 3.3 get the paring genes (g') of gene g for each cells
        self.rl_agg = optimizers.generate_LR_agg(self.alter_sc_exp,self.lr_df)
        # 3.4 get the affinity
        self.aff = optimizers.calcNeighborAffinityMat(self.spots_nn_lst, self.sc_agg_meta, self.lr_df, self.alter_sc_exp)
        #np.fill_diagonal(self.aff,0)
        #不要转为DataFrame吧？不然不是相当于没稀疏吗
        self.aff = pd.DataFrame(self.aff.toarray(), index = self.sc_agg_meta.index, columns=self.sc_agg_meta.index)
        self.sc_knn = optimizers.findCellAffinityKNN(self.spots_nn_lst, self.sc_agg_meta, self.aff, self.K)
        # 3.5 Calculate the derivative
        self.term3_df,self.loss3 = optimizers.cal_term3(self.alter_sc_exp,self.sc_knn,self.aff,self.sc_dist,self.rl_agg)
        print('Third term calculation done!')

        # 4. Fourth term, towards spot-spot affinity profile
        # self.rl_agg_align = optimizers.generate_LR_agg(self.alter_sc_exp,self.lr_df_align)
        self.term4_df,self.loss4 = optimizers.cal_term4(self.st_exp,self.sc_knn,self.st_aff_profile_df,self.alter_sc_exp,
                                                        self.sc_agg_meta,self.spot_cell_dict,self.lr_df_align)
        print('Fourth term calculation done!')
        
        # 5. Fifth term, norm2 regulization
        self.term5_df,self.loss5 = optimizers.cal_term5(self.alter_sc_exp)
        

    def init_grad(self):
        if isinstance(self.init_sc_embed, pd.DataFrame):
            self.sc_coord = utils.check_sc_coord(self.init_sc_embed)
            print('Using user provided init sc_coord.')
        else:
            print('Init sc_coord by affinity embedding...')
            # TODO 减少aff计算次数；使用sparse array
            self.sc_coord,_,_,_ = optimizers.aff_embedding(self.alter_sc_exp,self.st_coord,self.sc_agg_meta,self.lr_df,
                            self.save_path,self.left_range,self.right_range,self.steps,self.dim,verbose = False)
        self.run_gradient()
        # v5 calculte the initial loss of each term to balance their force.
        adj2,adj3,adj4,adj5 = optimizers.loss_adj(self.loss1,self.loss2,self.loss3,self.loss4,self.loss5)
        self.ALPHA,self.BETA,self.GAMMA,self.DELTA = self.ALPHA*adj2,self.BETA*adj3,self.GAMMA*adj4,self.DELTA*adj5
        #self.sc_agg_meta[['UMAP1','UMAP2']] = self.sc_coord
        print('Hyperparameters adjusted.')


    def gradient_descent(self):
        if not isinstance(self.sc_ref, np.ndarray):
            self.sc_ref = np.array(self.sc_ref.loc[self.sc_agg_meta['sc_id']])
        print('Running v12 now...')
        res_col = ['loss1','loss2','loss3','loss4','loss5','total']
        result = pd.DataFrame(columns=res_col)
        if self.st_tp == 'slide-seq':
            # cell coord as spot coord
            self.init_sc_embed = self.st_coord.loc[self.sc_agg_meta['spot']]
            self.init_sc_embed.index = self.sc_agg_meta.index
        self.init_grad()
        # loss = self.loss1 + self.ALPHA*self.loss2 + self.BETA*self.loss3 + self.GAMMA*self.loss4 + self.DELTA*self.loss5
        # tmp = pd.DataFrame(np.array([[self.loss1,self.ALPHA*self.loss2,self.BETA*self.loss3,self.GAMMA*self.loss4,self.DELTA*self.loss5,loss]]),columns = res_col, index = [0])
        # result = pd.concat((result,tmp),axis=0)
        ######### init done ############
        for ite in range(self.iteration):
            print(f'-----Start iteration {ite} -----')
            # TODO 减少aff计算次数；使用sparse array
            if self.st_tp != 'slide-seq':
                self.sc_coord,_,_,_ = optimizers.aff_embedding(self.alter_sc_exp,self.st_coord,self.sc_agg_meta,self.lr_df,
                                self.save_path,self.left_range,self.right_range,self.steps,self.dim)
            self.run_gradient()
            gradient = self.term1_df - self.ALPHA*self.term2_df + self.BETA*self.term3_df + self.GAMMA*self.term4_df + self.DELTA*self.term5_df
            self.alter_sc_exp = self.alter_sc_exp - self.ETA * gradient
            # TODO check
            self.alter_sc_exp[self.alter_sc_exp<0] = 0
            # v2 added

            print(f'---{ite} self.loss4 {self.loss4} self.GAMMA {self.GAMMA} self.GAMMA*self.loss4 {self.GAMMA*self.loss4}')
            loss = self.loss1 + self.ALPHA*self.loss2 + self.BETA*self.loss3 + self.GAMMA*self.loss4 + self.DELTA*self.loss5
            tmp = pd.DataFrame(np.array([[self.loss1,self.ALPHA*self.loss2,self.BETA*self.loss3,self.GAMMA*self.loss4,self.DELTA*self.loss5,loss]]),columns = res_col, index = [ite])
            result = pd.concat((result,tmp),axis=0)
            print(f'---In iteration {ite}, the loss is:loss1:{self.loss1:.5f},loss2:{self.loss2:.5f},loss3:{self.loss3:.5f},', end="")
            print(f'loss4:{self.loss4:.5f},loss5:{self.loss5:.5f}.')
            print(f'---In iteration {ite}, the loss is:loss1:{self.loss1:.5f},loss2:{self.ALPHA*self.loss2:.5f},loss3:{self.BETA*self.loss3:.5f},', end="")
            print(f'loss4:{self.GAMMA*self.loss4:.5f},loss5:{self.DELTA*self.loss5:.5f}.')
            print(f'The total loss after iteration {ite} is {loss:.5f}.')

        ### v5 add because spatalk demo
        # TODO check
        self.alter_sc_exp[self.alter_sc_exp < 1] = 0  
        self.alter_sc_exp.to_csv(f'{self.save_path}/alter_sc_exp.tsv',sep = '\t',header=True,index=True)
        result.to_csv(f'{self.save_path}/loss.tsv',sep = '\t',header=True,index=True)
        self.result = result
        ### v6 add for center shift  
        if self.st_tp != 'slide-seq':
            self.sc_coord,_,_,_ = optimizers.aff_embedding(self.alter_sc_exp,self.st_coord,self.sc_agg_meta,self.lr_df,
                                self.save_path,self.left_range,self.right_range,self.steps,self.dim)
            _, sc_spot_center = optimizers.sc_prep(self.st_coord, self.sc_agg_meta)
            self.sc_agg_meta[['st_x','st_y']] = sc_spot_center
            self.sc_agg_meta = optimizers.center_shift_embedding(self.sc_coord, self.sc_agg_meta, max_dist = 1)
        else:
            # v10
            self.sc_agg_meta[['st_x','st_y']] = self.sc_coord
            self.sc_agg_meta[['adj_spex_UMAP1','adj_spex_UMAP2']] = self.sc_coord
        return self.alter_sc_exp, self.sc_agg_meta
