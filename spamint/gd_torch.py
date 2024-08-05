import torch
import numpy as np
import pandas as pd
from typing import Dict, List
import logging as logger
import pdb
from sklearn.preprocessing import LabelEncoder

from . import optimizers

def test_grad_loss(grad1, loss1, grad2, loss2):
    grad = grad1 - grad2
    loss = loss1 - loss2
    if loss <= 1e-8 and (grad <= 1e-6).all().all():
        logger.debug('Grad test passed')
    else:
        logger.debug('Grad test failed!')
        pdb.set_trace()

class GradientDescentSolver:
    # 通过__getattr__继承了SpaMint所有成员，此外还有：

    # 原先的传参（略）

    # 中间产物
    spot_cell_dict: Dict[str, List[str]]
    
    def __init__(self, spex, alpha, beta, gamma, delta, eta, 
                init_sc_embed = False,
                iteration = 20, k = 2, W_HVG = 2,
                left_range = 1, right_range = 2, steps = 1, dim = 2):
        self.spex = spex

        self.alter_sc_exp = self.sc_exp.loc[self.sc_agg_meta['sc_id']]
        self.alter_sc_exp.index = self.sc_agg_meta.index
        self.spot_cell_dict = self.sc_agg_meta.groupby('spot')\
            .apply(lambda x: x.index.tolist()).to_dict()

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


    def __getattr__(self, k):
        try:
            return getattr(self.spex, k)
        except AttributeError:
            raise AttributeError(f"GradientDescent has no attr {k}")
        

    # 将DataFrame全部Tensor化 column和index另外存储
    def preprocess(self):
        if not isinstance(self.sc_ref, torch.Tensor):
            self.sc_ref_tensor = torch.from_numpy(np.array(self.sc_ref.loc[self.sc_agg_meta['sc_id']]))
        else:
            self.sc_ref_tensor = torch.from_numpy(self.sc_ref)
        # 对sc_exp: 全是数值 保存列的map即可
        self.sc_exp_columns_map = {k: v for v, k in enumerate(self.alter_sc_exp.columns)}
        self.sc_exp_tensor = torch.from_numpy(self.alter_sc_exp.to_numpy())
        # 对st_exp: 同前
        self.st_exp_columns_map = {k: v for v, k in enumerate(self.st_exp.columns)}
        self.st_exp_tensor = torch.from_numpy(self.st_exp.to_numpy())
        # 对sc_agg_meta: 有字符串 需要用LabelEncoder


    def term1(self):
        # add for merfish
        alter_sc_exp = self.alter_sc_exp[self.st_exp.columns].copy()
        # 1.1 Aggregate exp of chosen cells for each spot 
        alter_sc_exp['spot'] = self.sc_agg_meta['spot']
        sc_spot_sum = alter_sc_exp.groupby('spot').sum()
        del alter_sc_exp['spot']
        sc_spot_sum = sc_spot_sum.loc[self.st_exp.index]
        # 1.2 equalize sc_spot_sum by dividing cell number each spot
        cell_n_spot = self.sc_agg_meta.groupby('spot').count().loc[self.st_exp.index]
        div_sc_spot = sc_spot_sum.div(cell_n_spot.iloc[:,0].values, axis=0)
        st_exp_tensor = torch.from_numpy(self.st_exp.to_numpy().astype(float))
        div_sc_spot_tensor = torch.from_numpy(div_sc_spot.to_numpy().astype(float))
        criterion = torch.nn.MSELoss()
        div_sc_spot_tensor.requires_grad_()
        loss = criterion(div_sc_spot_tensor, st_exp_tensor)
        loss.backward()
        grad = div_sc_spot_tensor.grad

        term1_df = pd.DataFrame(grad.numpy(), index=div_sc_spot.index, columns=div_sc_spot.columns)
        term1_df *= (div_sc_spot.shape[0] * div_sc_spot.shape[1])

        # 1.3 add weight of hvg
        hvg = list(self.svg)
        nvg = list(set(self.st_exp.columns).difference(set(hvg)))
        weight_nvg = pd.DataFrame(np.ones((self.st_exp.shape[0],len(nvg))),columns = nvg,  index = self.st_exp.index)
        weight_hvg = pd.DataFrame(np.ones((self.st_exp.shape[0],len(hvg)))*self.W_HVG,columns = hvg,  index = self.st_exp.index)
        weight = pd.concat((weight_hvg,weight_nvg),axis = 1)
        weight = weight[self.st_exp.columns]
        term1_df *= weight

        # 1.4 Broadcast gradient for each cell
        term1_df = term1_df.loc[self.sc_agg_meta['spot']]
        term1_df.index = alter_sc_exp.index
        # add for merfish
        term1_df = optimizers.complete_other_genes(self.alter_sc_exp, term1_df)
        return term1_df, float(loss)

    def term2(self):
        sc_ref_tensor = torch.from_numpy(self.sc_ref)
        sc_exp_tensor = torch.from_numpy(self.alter_sc_exp.values)
        term2 = sc_ref_tensor - torch.digamma(sc_exp_tensor + 1)
        term2_df = pd.DataFrame(term2.numpy(),index = self.alter_sc_exp.index,columns=self.alter_sc_exp.columns)
        criterion = torch.nn.MSELoss()
        loss = criterion(sc_ref_tensor, sc_exp_tensor)
        return term2_df, float(loss)

    def term3(self):
        return term3_df, float(loss)
    
    def term4(self):
        return grad, loss
    
    def term5(self):
        return grad, loss

    def run_gradient(self):
        # 1. First term
        '''
        logger.debug('Term1 (original) start')
        self._term1_df,self._loss1 = optimizers.cal_term1(
            self.alter_sc_exp,self.sc_agg_meta,self.st_exp,self.svg,self.W_HVG)
        logger.debug('Term1 (torch) start')
        self.term1_df, self.loss1 = self.term1()
        logger.debug('First-term calculation done!')
        test_grad_loss(self._term1_df, self._loss1, self.term1_df, self.loss1)
        
        # 2. Second term
        logger.debug('Term2 (original) start')
        self._term2_df,self._loss2 = optimizers.cal_term2(self.alter_sc_exp,self.sc_ref)
        logger.debug('Term2 (torch) start')
        self.term2_df, self.loss2 = self.term2()
        logger.debug('Second-term calculation done!')
        test_grad_loss(self._term2_df, self._loss2, self.term2_df, self.loss2)
        '''

        # 3. Third term, closer cells have larger affinity
        # 需要先求出aff矩阵与distance矩阵（都是只需要求neighbor之间的）
        term3_knn = False
        if not (self.st_tp == 'slide-seq' and hasattr(self, 'sc_knn')):
            # if slide-seq and already have found sc_knn
            # dont do it again
            self.sc_dist = pd.DataFrame(distance_matrix(self.sc_coord,self.sc_coord), index = self.alter_sc_exp.index, columns = self.alter_sc_exp.index)
            # 3.2 get c' = N(c)
            self.sc_knn = optimizers.findCellKNN(self.spots_nn_lst,self.st_tp,self.sc_agg_meta,self.spot_cell_dict,self.sc_coord,self.K)
            utils.check_empty_dict(self.sc_knn)
            term3_knn = True

        # 3.3 get the paring genes (g') of gene g for each cells
        self.rl_agg = optimizers.generate_LR_agg(self.alter_sc_exp,self.lr_df)
        # 3.4 get the affinity
        #self.aff = optimizers.calcNeighborAffinityMat(self.spots_nn_lst, self.spot_cell_dict, self.lr_df, self.alter_sc_exp)
        self.aff = optimizers.calcNeighborAffinityMat(self.spots_nn_lst, self.spot_cell_dict, self.lr_df, self.alter_sc_exp)
        #np.fill_diagonal(self.aff,0)
        #不要转为DataFrame吧？不然不是相当于没稀疏吗
        self.aff = pd.DataFrame(self.aff.toarray(), index = self.sc_agg_meta.index, columns=self.sc_agg_meta.index)
        # 这里很慢
        if not term3_knn:
            self.sc_knn = optimizers.findCellAffinityKNN(self.spots_nn_lst, self.spot_cell_dict, self.aff, self.K)
        # 3.5 Calculate the derivative
        self.term3_df,self.loss3 = optimizers.cal_term3(self.alter_sc_exp,self.sc_knn,self.aff,self.sc_dist,self.rl_agg)
        logger.debug('Third term calculation done!')

        '''
        # 4. Fourth term, towards spot-spot affinity profile
        # self.rl_agg_align = optimizers.generate_LR_agg(self.alter_sc_exp,self.lr_df_align)
        self.term4_df,self.loss4 = optimizers.cal_term4(self.st_exp,self.sc_knn,self.st_aff_profile_df,self.alter_sc_exp,
                                                        self.sc_agg_meta,self.spot_cell_dict,self.lr_df_align)
        logger.debug('Fourth term calculation done!')
        
        # 5. Fifth term, norm2 regulization
        self.term5_df,self.loss5 = optimizers.cal_term5(self.alter_sc_exp)
        '''


    def init_grad(self):
        if isinstance(self.init_sc_embed, pd.DataFrame):
            self.sc_coord = utils.check_sc_coord(self.init_sc_embed)
            self.sc_coord = self.sc_coord.to_numpy()
            logger.debug('Using user provided init sc_coord.')
        else:
            logger.debug('Init sc_coord by affinity embedding...')
            # TODO 减少aff计算次数；使用sparse array
            self.sc_coord,_,_,_ = optimizers.aff_embedding(
                self.spots_nn_lst, self.spot_cell_dict,
                self.alter_sc_exp,self.st_coord,self.sc_agg_meta,self.lr_df,
                self.save_path,self.left_range,self.right_range,self.steps,self.dim,verbose = False)
        self.run_gradient()
        # v5 calculte the initial loss of each term to balance their force.
        adj2,adj3,adj4,adj5 = optimizers.loss_adj(self.loss1,self.loss2,self.loss3,self.loss4,self.loss5)
        self.ALPHA,self.BETA,self.GAMMA,self.DELTA = self.ALPHA*adj2,self.BETA*adj3,self.GAMMA*adj4,self.DELTA*adj5
        self.sc_agg_meta[['UMAP1','UMAP2']] = self.sc_coord
        logger.debug('Hyperparameters adjusted.')


    def gradient_descent(self):
        self.preprocess()
        logger.debug('Running v13 now...')
        res_col = ['loss1','loss2','loss3','loss4','loss5','total']
        #if self.st_tp != 'slide-seq':
        # 调试需要
        # 看起来每次都会需要进行aff embedding 为了term3
        # 合理安排一下可以节约一次affinity mat的计算
        if not hasattr(self, 'sc_coord'):
                self.sc_coord,_,_,_ = optimizers.aff_embedding(
                    self.spots_nn_lst, self.spot_cell_dict,
                    self.alter_sc_exp,self.st_coord,self.sc_agg_meta,self.lr_df,
                    self.save_path,self.left_range,self.right_range,self.steps,self.dim, verbose=True)
        self.run_gradient()
        pass

    def ps(self):
        if not isinstance(self.sc_ref, np.ndarray):
            self.sc_ref = np.array(self.sc_ref.loc[self.sc_agg_meta['sc_id']])
        logger.debug('Running v12 now...')
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
            logger.debug(f'-----Start iteration {ite} -----')
            # TODO 减少aff计算次数；使用sparse array
            if self.st_tp != 'slide-seq':
                self.sc_coord,_,_,_ = optimizers.aff_embedding(
                    self.spots_nn_lst, self.spot_cell_dict,
                    self.alter_sc_exp,self.st_coord,self.sc_agg_meta,self.lr_df,
                    self.save_path,self.left_range,self.right_range,self.steps,self.dim)
            self.run_gradient()
            gradient = self.term1_df - self.ALPHA*self.term2_df + self.BETA*self.term3_df + self.GAMMA*self.term4_df + self.DELTA*self.term5_df
            self.alter_sc_exp = self.alter_sc_exp - self.ETA * gradient
            # TODO check
            self.alter_sc_exp[self.alter_sc_exp<0] = 0
            # v2 added

            logger.debug(f'---{ite} self.loss4 {self.loss4} self.GAMMA {self.GAMMA} self.GAMMA*self.loss4 {self.GAMMA*self.loss4}')
            loss = self.loss1 + self.ALPHA*self.loss2 + self.BETA*self.loss3 + self.GAMMA*self.loss4 + self.DELTA*self.loss5
            tmp = pd.DataFrame(np.array([[self.loss1,self.ALPHA*self.loss2,self.BETA*self.loss3,self.GAMMA*self.loss4,self.DELTA*self.loss5,loss]]),columns = res_col, index = [ite])
            result = pd.concat((result,tmp),axis=0)
            logger.debug(f'---In iteration {ite}, the loss is:loss1:{self.loss1:.5f},loss2:{self.loss2:.5f},loss3:{self.loss3:.5f},\
                loss4:{self.loss4:.5f},loss5:{self.loss5:.5f}.')
            logger.debug(f'---In iteration {ite}, the loss is:loss1:{self.loss1:.5f},loss2:{self.ALPHA*self.loss2:.5f},loss3:{self.BETA*self.loss3:.5f},\
                loss4:{self.GAMMA*self.loss4:.5f},loss5:{self.DELTA*self.loss5:.5f}.')
            logger.debug(f'The total loss after iteration {ite} is {loss:.5f}.')

        ### v5 add because spatalk demo
        # TODO check
        self.alter_sc_exp[self.alter_sc_exp < 1] = 0  
        self.alter_sc_exp.to_csv(f'{self.save_path}/alter_sc_exp.tsv',sep = '\t',header=True,index=True)
        result.to_csv(f'{self.save_path}/loss.tsv',sep = '\t',header=True,index=True)
        self.result = result
        ### v6 add for center shift  
        if self.st_tp != 'slide-seq':
            self.sc_coord,_,_,_ = optimizers.aff_embedding(
                self.spots_nn_lst, self.spot_cell_dict,
                self.alter_sc_exp,self.st_coord,self.sc_agg_meta,self.lr_df,
                self.save_path,self.left_range,self.right_range,self.steps,self.dim)
            _, sc_spot_center = optimizers.sc_prep(self.st_coord, self.sc_agg_meta)
            self.sc_agg_meta[['st_x','st_y']] = sc_spot_center.values
            self.sc_agg_meta = optimizers.center_shift_embedding(self.sc_coord, self.sc_agg_meta, max_dist = 1)
        else:
            # v10
            self.sc_agg_meta[['st_x','st_y']] = self.sc_coord
            self.sc_agg_meta[['adj_spex_UMAP1','adj_spex_UMAP2']] = self.sc_coord
        return self.alter_sc_exp, self.sc_agg_meta
