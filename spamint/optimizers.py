import pandas as pd
import numpy as np
from scipy.special import digamma
from scipy.spatial import distance_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
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


def loss_adj(loss1,loss2,loss3,loss4,loss5):
    adj2 = loss1/loss2
    adj3 = loss1/loss3
    adj4 = loss1/loss4
    adj5 = loss1/loss5
    return adj2,adj3,adj4,adj5


def cal_term1_old(alter_sc_exp,sc_meta,st_exp):
    '''
    1. First term, towards spot expression
    '''  
    # 1.1 Aggregate exp of chosen cells for each spot 
    alter_sc_exp['spot'] = sc_meta['spot']
    sc_spot_sum = alter_sc_exp.groupby('spot').sum()
    del alter_sc_exp['spot']
    sc_spot_sum = sc_spot_sum.loc[st_exp.index]
    # 1.2 equalize sc_spot_sum by dividing cell number each spot
    cell_n_spot = sc_meta.groupby('spot').count().loc[st_exp.index]
    div_sc_spot = sc_spot_sum.div(cell_n_spot.iloc[:,0].values, axis=0)
    # 1.2 Calculate gradient
    term1_df = 2 * (div_sc_spot - st_exp)
    # 1.3 Broadcast gradient for each cell
    term1_df = term1_df.loc[sc_meta['spot']]
    term1_df.index = alter_sc_exp.index
    loss_1 = mean_squared_error(st_exp,div_sc_spot)
    return term1_df,loss_1


def cal_term1(sc_exp,sc_meta,st_exp,hvg,W_HVG):
    '''
    1. First term, towards spot expression
    '''  
    # add for merfish
    alter_sc_exp = sc_exp[st_exp.columns].copy()
    # 1.1 Aggregate exp of chosen cells for each spot 
    alter_sc_exp['spot'] = sc_meta['spot']
    sc_spot_sum = alter_sc_exp.groupby('spot').sum()
    del alter_sc_exp['spot']
    sc_spot_sum = sc_spot_sum.loc[st_exp.index]
    # 1.2 equalize sc_spot_sum by dividing cell number each spot
    cell_n_spot = sc_meta.groupby('spot').count().loc[st_exp.index]
    div_sc_spot = sc_spot_sum.div(cell_n_spot.iloc[:,0].values, axis=0)
    # 1.2 Calculate gradient
    term1_df = 2 * (div_sc_spot - st_exp)
    # v5 add weight on hvg
    # 1.3 add weight
    hvg = list(hvg)
    nvg = list(set(st_exp.columns).difference(set(hvg)))
    weight_nvg = pd.DataFrame(np.ones((st_exp.shape[0],len(nvg))),columns = nvg,  index = st_exp.index)
    weight_hvg = pd.DataFrame(np.ones((st_exp.shape[0],len(hvg)))*W_HVG,columns = hvg,  index = st_exp.index)
    weight = pd.concat((weight_hvg,weight_nvg),axis = 1)
    weight = weight[st_exp.columns]
    term1_df *= weight
    # 1.4 Broadcast gradient for each cell
    term1_df = term1_df.loc[sc_meta['spot']]
    term1_df.index = alter_sc_exp.index
    loss_1 = mean_squared_error(st_exp,div_sc_spot)
    # add for merfish
    term1_df = complete_other_genes(sc_exp, term1_df)
    return term1_df,loss_1


def cal_term2(alter_sc_exp,sc_distribution):
    '''
    2. Second term, towards sc cell-type specific expression
    '''
    # 2.1 poisson distribution
    term2 = sc_distribution - digamma(alter_sc_exp + 1).values
    term2_df = pd.DataFrame(term2,index = alter_sc_exp.index,columns=alter_sc_exp.columns)
    loss_2 = mean_squared_error(sc_distribution, alter_sc_exp)
    return term2_df,loss_2     


def findSpotKNN_old(st_coord, st_tp): 
    # TODO
    # write on 1107 to replace findSpotNeighbor in the future
    # no need for slide-seq exceptions
    # for sc usage, moderate k? == further research
    # 如果这么写就无法检测离群点了会不会有影响？==> 会影响到SC的nn，因为SC还是考虑离群点了
    coordinates = st_coord.values
    if st_tp != 'slide-seq':
        k = 6
    else:
        k = 6
    kdtree = KDTree(coordinates)
    distances, indices = kdtree.query(coordinates, k+1)
    knn_dict = {}
    spots_id = st_coord.index.tolist()
    for i, nearest_indices in enumerate(indices):
        point = nearest_indices[0]
        knn = nearest_indices[1:].tolist()
        knn_dict[spots_id[point]] = [spots_id[i] for i in knn]
    return knn_dict


def findSpotKNN(st_coord, st_tp):
    # TODO
    # write on 1107 to replace findSpotNeighbor in the future
    # no need for slide-seq exceptions
    # for sc usage, moderate k? == further research
    thred = 95
    total_sum = 0
    coordinates = st_coord.values
    if st_tp != 'slide-seq':
        k = 4
    else:
        k = 6
    kdtree = KDTree(coordinates)
    distances, indices = kdtree.query(coordinates, k=k + 1)
    # Autocalculate the outlier threshold based on the distances
    threshold = np.percentile(distances[:, 1:], thred)
    indices = pd.DataFrame(indices,index = st_coord.index)
    distances = pd.DataFrame(distances,index = st_coord.index)
    # print(threshold)
    knn_dict = {}
    spots_id = st_coord.index.tolist()
    for key in spots_id:
        nearest_neighbors = indices.loc[key, 1:]
        nearest_distances = distances.loc[key, 1:]
        # keep nn within threshold
        filtered_neighbors = nearest_neighbors[nearest_distances <= threshold]
        str_idx = [spots_id[index] for index in filtered_neighbors]
        knn_dict[key] = str_idx
        total_sum += len(str_idx)
    print(f'By setting k as {k}, each spot has average {total_sum/st_coord.shape[0]} neighbors.')
    return knn_dict


def findSpotNeighbor(st_coord,st_tp):
    # old
    all_x = np.sort(list(set(st_coord.iloc[:,0])))
    delta_x = all_x[1] - all_x[0]

    if st_tp == 'visum':
        n_thred = 2*delta_x
        print(f'Visum format, setting threshold as {n_thred}')
    else:
        n_thred = delta_x + 0.001

    st_dist = pd.DataFrame(distance_matrix(st_coord,st_coord),columns = st_coord.index, index = st_coord.index)
    st_dist[(st_dist < n_thred)&(st_dist >0)] = 1
    st_dist[st_dist != 1] = 0
    return st_dist

def findCellKNN_fn(spot,spots_nn_lst,spot_cell_dict,sc_coord,k):
    sc_knn = {}
    # 1. Find neighboring cells from adjacent spot
    st_neighbors = spots_nn_lst[spot]
    sc_neighbors = []
    for x in st_neighbors: sc_neighbors += spot_cell_dict[x]
    sc_myself = spot_cell_dict[spot]

    sc_coord_neighbor = sc_coord[[int(x) for x in sc_neighbors]]
    sc_coord_myself = sc_coord[[int(x) for x in sc_myself]]

    # 1.3 对这些细胞造KDTree，然后在自己含有的细胞中分别求knn
    sc_kdtree = KDTree(data=sc_coord_neighbor)
    for sc_i in range(len(sc_myself)):
        # 这种写法好像有问题，不符合原功能
        _, sc_idxes = sc_kdtree.query(sc_coord_myself[sc_i], k=k)
        sc_idxes = sc_idxes.tolist()
        sc_knn[sc_myself[sc_i]] = [str(x) for x in sc_idxes]
    return sc_knn


def findCellKNN(spots_nn_lst, st_tp, sc_meta, spot_cell_dict, sc_coord, k): 
    '''
    st_tp = 'visum'
    k = 2
    sc_coord = obj_spex.sc_coord
    '''
    if st_tp == 'slide-seq':
        sc_knn = findCellKNN_slide(sc_meta,sc_coord)
    else:
        sc_knn = {}
        for key in sc_meta.index.tolist():
            sc_knn[key] = []

        pool = multiprocessing.Pool(8)
        arglist = [(x,spots_nn_lst,spot_cell_dict,sc_coord,k)
                   for x in spots_nn_lst]
        ret_list = pool.starmap(findCellKNN_fn, arglist)
        for ret in ret_list:
          for cell in ret:
            sc_knn[cell].extend(ret[cell])

        # remove no neighbor cells
        empty_keys = [k for k, v in sc_knn.items() if not v]
        for k in empty_keys:
            del sc_knn[k]
        '''
        # 1.1 assign ST coord to cells
        _, sc_coord_st = sc_prep(st_coord, sc_meta)
        # 1.2 find neighboring cells from adjacent spot
        sc_nn = findSpotNeighbor(sc_coord_st,st_tp)
        # sc_nn = findSpotKNN(sc_coord_st,st_tp)
        # 1.3 calculate real cell-cell distance
        # TODO: 不要用全部的distance
        sc_dist = pd.DataFrame(distance_matrix(sc_coord,sc_coord),columns = sc_meta.index, index = sc_meta.index)

        # 2. One hop cross-spot neighbor within a threshold
        # 2.1 Find threshold, i.e., sc_centroid distance
        sc_coord = pd.DataFrame(sc_coord,columns = ['x','y'])
        sc_coord['spot'] = sc_meta['spot'].values
        sc_centroid = sc_coord.groupby('spot').mean()
        sc_centroid, sc_centroid_cells = sc_prep(sc_centroid, sc_meta)
        sc_centroid_cells_dist = pd.DataFrame(distance_matrix(sc_centroid_cells,sc_centroid_cells),columns = sc_meta.index, index = sc_meta.index)  
        # 2.2 Find one hop neighbor between spots
        sc_dist_st_nn = sc_dist * sc_nn.values
        sc_dist_st_nn = sc_dist_st_nn[sc_dist_st_nn != 0]
        # 2.2.1 Apply cell-centroid threshold
        sc_dist_st_nn = sc_dist_st_nn[sc_centroid_cells_dist > sc_dist_st_nn]
        # 2.2.2 Find k-nearest neighbor in each neighbor
        sc_dist_st_nn['spot'] = sc_meta['spot'].values
        # sc_knn = sc_dist_st_nn.groupby('spot').apply(apply_nsmallest, k = k, nn_dict = sc_knn)
        # sc_knn = sc_knn[sc_knn.keys()[0]]
        pool = multiprocessing.Pool(8)
        ret_list = pool.starmap(apply_nsmallest, [(x, k) for _, x in sc_dist_st_nn.groupby('spot')])
        for ret in ret_list:
          for cell in ret:
            sc_knn[cell].extend(ret[cell])
        '''
    return sc_knn


def findCellKNN_slide(sc_meta,sc_coord):
    # k=4+1 include self
    k = 7
    # drop 95 out of 100
    thred = 95
    sum = 0
    sc_knn = {}
    for key in sc_meta.index.tolist():
        sc_knn[key] = []
    idx_lst = sc_coord.index
    kdtree = KDTree(sc_coord)
    distances, indices = kdtree.query(sc_coord, k=k)
    threshold = np.percentile(distances[:, 1:], thred)
    # print(threshold)
    indices = pd.DataFrame(indices,index = sc_meta.index)
    distances = pd.DataFrame(distances,index = sc_meta.index)
    for key in sc_meta.index.tolist():
        # remove self
        nearest_neighbors = indices.loc[key, 1:]
        nearest_distances = distances.loc[key, 1:]
        # keep nn within threshold
        filtered_neighbors = nearest_neighbors[nearest_distances <= threshold]
        str_idx = [idx_lst[index] for index in filtered_neighbors]
        sc_knn[key] = str_idx
        sum += len(str_idx)
    print(f'Running slide-seq data, each cell has average {sum/sc_meta.shape[0]} neighbor.')
    return sc_knn


def apply_nsmallest(x, k, nn_dict=None):
    x = x.dropna(axis=1, how='all')
    ret = {}
    for cell in x.columns:
        #print(cell,x[cell].nsmallest(k).index.tolist())
        if cell != 'spot':
            #nn_dict[cell].extend(x[cell].nsmallest(k).index.tolist())
            ret[cell] = x[cell].nsmallest(k).index.tolist()
    #return nn_dict
    return ret


def complete_other_genes(alter_sc_exp, term_LR_df):
    '''
    Complete non-LR genes as zero for term3,4
    '''
    term_df = pd.DataFrame(np.zeros(alter_sc_exp.shape),columns=alter_sc_exp.columns,index = alter_sc_exp.index)
    term_df.update(term_LR_df)
    return term_df

# 求出邻近Spot中和细胞Affinity最大的几个细胞 返回dict
def findCellAffinityKNN(spots_nn_lst,spot_cell_dict,aff,k):
    sc_knn = {}

    pool = multiprocessing.Pool(8)
    arglist = [(spot,spots_nn_lst,spot_cell_dict,aff,k)
               for spot in spots_nn_lst]
    ret_list = pool.starmap(findCellAffinityKNNFn, arglist)
    for ret in ret_list:
      for cell in ret:
        sc_knn[cell] = ret[cell]

    # remove no neighbor cells
    empty_keys = [k for k, v in sc_knn.items() if not v]
    for k in empty_keys:
        del sc_knn[k]

    return sc_knn


def findCellAffinityKNNFn(spot, spots_nn_lst, spot_cell_dict, aff, k):
    sc_knn = {}

    st_neighbors = spots_nn_lst[spot]
    sc_neighbors = []
    for x in st_neighbors: sc_neighbors += spot_cell_dict[x]
    sc_myself = spot_cell_dict[spot]

    # 对这些细胞分别在sc_neighbor选出aff最大的k个
    for sc in sc_myself:
        sc_knn[sc] = aff.loc[sc][sc_neighbors].nlargest(k).index.tolist()

    return sc_knn


def cal_term3(alter_sc_exp,sc_knn,aff,sc_dist,rl_agg):
    # v3 added norm_aff and norm rl_cp to regulize the values
    # v5 Updated the calculation of term3 with [ind], accelerated.
    # v5 Scale both data to a fixed range every time
    MIN = 0
    MAX = 100
    norm_aff = np.sqrt(aff/2)
    term3_LR = pd.DataFrame()
    #'''
    #sc_dist_re = sc_dist.copy()
    # ind: the neighboring indicator matrix
    # Row: cell; Col: the neighbor of this cell
    ind = pd.DataFrame(False, columns = norm_aff.columns,index = norm_aff.index)
    for idx, cp in sc_knn.items():
        ind.loc[idx,cp] = True
    # n_cp: the neighboring cells of each cell (row-wise summation)
    n_cp = ind.sum(axis = 1)
    cp_aff_df = norm_aff[ind]
    cp_dist_df = sc_dist[ind]
    '''
    lil_cp_aff_df = lil_matrix(norm_aff.shape)
    lil_cp_dist_df = lil_matrix(sc_dist.shape)
    n_cp = pd.Series()
    for idx, cp in sc_knn.items():
        n_cp[idx] = len(cp)
        for idx2 in cp:
            lil_cp_aff_df[int(idx), int(idx2)] = norm_aff.loc[idx, idx2]
            lil_cp_dist_df[int(idx), int(idx2)] = sc_dist.loc[idx, idx2]

    cp_aff_df = pd.DataFrame.sparse.from_spmatrix(
        lil_cp_aff_df, columns = norm_aff.columns, index = norm_aff.index)
    cp_dist_df = pd.DataFrame.sparse.from_spmatrix(
        lil_cp_dist_df, columns = sc_dist.columns, index = sc_dist.index)
    '''

    mask = cp_dist_df != 0
    cp_dist_df[mask] = 1 / cp_dist_df[mask]
    cp_aff_adj = utils.scale_global_MIN_MAX(cp_aff_df,MIN,MAX)
    cp_dist_adj = utils.scale_global_MIN_MAX(cp_dist_df,MIN,MAX)
    tmp1 = cp_aff_adj - cp_dist_adj
    tmp2 = tmp1.fillna(0)
    term3_LR = 2*rl_agg.dot(tmp2.T)/n_cp
    # fillna(0) because if a cell has no neighbor, /n_cp cause divide by zero error; generates NA.
    term3_LR = term3_LR.fillna(0)
    # print('\t filled na')
    # Calculating the loss; Normlize by total neighbor count
    loss = np.sum(tmp2**2).sum()
    loss /= n_cp.sum()
    # v4 simplify
    term3_df = complete_other_genes(alter_sc_exp, term3_LR)
    return term3_df,loss


def cal_aff_profile(exp, nn_list, lr_df):
    # TODO: 重复计算的LRDF？
    lr_df_align = lr_df[lr_df[0].isin(exp.columns) & lr_df[1].isin(exp.columns)].copy()
    st_L = exp[lr_df_align[0]]
    st_R = exp[lr_df_align[1]]
    st_L_list = []
    st_R_list = []
    for i in range(st_L.shape[0]):
        l = st_L.loc[nn_list[st_R.iloc[i].name]]
        l = l.append(st_L.loc[st_R.iloc[i].name])
        r = st_R.loc[nn_list[st_L.iloc[i].name]]
        r = r.append(st_R.loc[st_L.iloc[i].name])
        st_L_list.append(l * st_R.values[i])
        st_R_list.append(r * st_L.values[i])
    st_LR_df = pd.concat(st_L_list, keys=st_R.index.tolist())
    st_RL_df = pd.concat(st_R_list, keys=st_L.index.tolist())
    st_aff_profile_df = st_LR_df + st_RL_df.values
    return st_aff_profile_df


def cal_sc_aff_profile(cell, cell_n, exp, lr_df):
    # TODO: 重复计算的LRDF？
    lr_df_align = lr_df[lr_df[0].isin(exp.columns) & lr_df[1].isin(exp.columns)].copy()
    st_L1 = exp.loc[cell,lr_df_align[0]]
    st_R1 = exp.loc[cell_n,lr_df_align[1]]
    st_L2 = exp.loc[cell_n,lr_df_align[0]]
    st_R2 = exp.loc[cell,lr_df_align[1]]
    #print(st_R2)
    #st_LR_df1 = pd.concat([st_L1 * st_R1.values[i] for i in range(st_R1.shape[0])], keys=st_R1.index.tolist())
    st_LR_df1 = st_R1 * st_L1.values
    #print(st_LR_df1)
    #st_LR_df2 = pd.concat([st_L2 * st_R2.values[i] for i in range(st_R2.shape[0])], keys=st_R2.index.tolist())
    st_LR_df2 = st_L2 * st_R2.values
    #print(st_LR_df2)
    st_aff_profile_df = st_LR_df1.values + st_LR_df2
    return st_aff_profile_df


def multiply_spots(df,res_tmp):
    spot_lst = df.index.get_level_values('spot').tolist()
    return df.multiply(res_tmp.loc[spot_lst].values,axis = 1)


def calSumNNRL(exp, spot_knn_df, cell_neigbors, gene_lst):
    '''
    Calculating the multiplier in term 4
    sum_{c \in s, c' \in s', c' \in N(c),g' = R(g)} e_{c'g'}
    '''
    tmp_sum_r = exp.loc[cell_neigbors,gene_lst]
    tmp_sum_r['spot']=spot_knn_df['spot'].values
    tmp_sum_r['cell_idx']=spot_knn_df['cell_idx'].values    
    # v3 sum => mean
    sum_ncg = tmp_sum_r.groupby(['spot','cell_idx']).mean()
    return sum_ncg


def cal_term4_fn(spot,sc_knn,st_aff_profile_df,alter_sc_exp,sc_meta,spot_cell_dict,lr_df):
    spot_cells = spot_cell_dict[spot]
    # 1. find knn id and its affiliated spot
    # spot_knn_df = knn_df[knn_df['cell_idx'].isin(spot_cells)]
    spot_knn_df = pd.DataFrame(
        [(x, sc_knn[x]) for x in spot_cells if x in sc_knn],
        columns = ['cell_idx', 'nn_cell_idx'])
    spot_knn_df = spot_knn_df.explode('nn_cell_idx')
    spot_knn_df['spot'] = sc_meta.loc[spot_knn_df['nn_cell_idx'].tolist()]['spot'].values
    cell_idx = spot_knn_df['cell_idx']
    cell_nn_idx = spot_knn_df['nn_cell_idx']
    # n_knn += len(cell_nn_idx)
    # some spot has no nn for any cell in it.
    res = None
    loss_tmp = 0
    if cell_nn_idx.tolist():
        # 2. calculate acc
        tmp_acc = cal_sc_aff_profile(cell_idx, cell_nn_idx, alter_sc_exp, lr_df)
        tmp_acc = tmp_acc.reset_index()
        del tmp_acc[tmp_acc.columns[0]]
        tmp_acc['spot'] = spot_knn_df['spot'].values
        a_cc = tmp_acc.groupby('spot').mean()
        # 3. calculate ass
        a_ss = st_aff_profile_df.loc[(spot,a_cc.index.tolist()),:]
        a_cc_modi = np.sqrt(a_cc/2)
        a_ss_modi = np.sqrt(a_ss/2)
        res_tmp = a_cc_modi - a_ss_modi.values
        loss_tmp = np.sum((res_tmp**2).values)
        # loss_4 += loss_tmp
        if np.isnan(loss_tmp):
            print(f'{spot}')
            print(f'acc{a_cc}')
            print(f'ass{a_ss}')
            print(f'a_cc_modi{a_cc_modi}')
            print(f'a_ss_modi{a_ss_modi}')
            print(f'res_tmp{res_tmp}')
        # 4. calculate multiplier
        sum_r = calSumNNRL(alter_sc_exp, spot_knn_df, cell_nn_idx, lr_df[1])
        sum_l = calSumNNRL(alter_sc_exp, spot_knn_df, cell_nn_idx, lr_df[0])
        res_L = sum_r.groupby('cell_idx').apply(multiply_spots,res_tmp = res_tmp)
        res_R = sum_l.groupby('cell_idx').apply(multiply_spots,res_tmp = res_tmp)
        # ? res_R.columns = sum_r.columns
        res_LR = pd.concat([res_L,res_R],axis =1)
        # sum for each adj spot
        ave_res_LR = res_LR.groupby('cell_idx').sum()
        # ave for same gene in LRdb
        res = ave_res_LR.T.groupby('symbol').mean().T
        #term4_LR = pd.concat([term4_LR,res],axis =0)
    return (res, len(cell_nn_idx), loss_tmp)

def cal_term4(st_exp,sc_knn,st_aff_profile_df,sc_exp,sc_meta,spot_cell_dict,lr_df):
    ''' 
    st_exp = obj_spex.st_exp
    sc_knn = obj_spex.sc_knn 
    st_aff_profile_df = obj_spex.st_aff_profile_df
    alter_sc_exp = obj_spex.alter_sc_exp
    sc_meta = obj_spex.sc_agg_meta
    spot_cell_dict = obj_spex.spot_cell_dict
    lr_df = obj_spex.lr_df
    '''
    alter_sc_exp = sc_exp[st_exp.columns].copy()
    # generate knn_df: cell_idx	-> nn_cell_idx
    # calculate when need
    # knn_df = pd.DataFrame(sc_knn.items(), columns=['cell_idx', 'nn_cell_idx'])
    # knn_df = knn_df.explode('nn_cell_idx')
    # nn_cell_idx = knn_df['nn_cell_idx'].tolist()
    # df = sc_meta.loc[nn_cell_idx].copy()
    # knn_df['spot'] = df['spot'].values
    term4_LR = pd.DataFrame()
    loss_4 = 0
    n_knn = 0
    pool = multiprocessing.Pool(8)
    arglist = [(x,sc_knn,st_aff_profile_df,sc_exp,sc_meta,spot_cell_dict,lr_df)
               for x in st_exp.index]
    ret_list = pool.starmap(cal_term4_fn, arglist)
    for ret in ret_list:
        term4_LR = pd.concat([term4_LR,ret[0]],axis =0)
        n_knn += ret[1]
        loss_4 += ret[2]

    loss_4 /= n_knn
    # if np.isnan(loss_4):
    #     print('nananananana')
    term4_df = complete_other_genes(sc_exp, term4_LR)
    return term4_df, loss_4


def cal_term5(alter_sc_exp):
    term5_df = alter_sc_exp*2
    loss5 = np.mean((alter_sc_exp**2).values)
    return term5_df, loss5

#### first edition ####
def generate_LR_agg(alter_sc_exp,lr_df):
    ''' L(g1): g1 as Receptor
        R(g1): g1 as Ligand
        L(g1) \ne R(g1)
        Since they were calculated by diff gene sum
    '''
    # V3: summed expression of pair RL => mean
    # V4: sum L(g1), R(g1) together
    # keep lr gene pairs exist in sc_exp
    genes = alter_sc_exp.columns.tolist()
    lr_df = lr_df[lr_df[0].isin(genes) & lr_df[1].isin(genes)]
    # summation of paired Receptor genes for each Ligand (row) in every cell (col).
    r_agg = alter_sc_exp[lr_df[1]].T
    r_agg['L'] = lr_df[0].values
    # v3
    # r_agg = r_agg.groupby('L').sum()
    r_agg = r_agg.groupby('L').mean()
    # summation of paired Ligand genes for each Receptor (row) in every cell (col).
    # v4 bug: lr_df[1] to lr_df[0],
    # should select L gene exp, agg by R
    l_agg = alter_sc_exp[lr_df[0]].T
    l_agg['R'] = lr_df[1].values
    # l_agg = l_agg.groupby('R').sum()
    l_agg = l_agg.groupby('R').mean()
    
    rl_agg = pd.concat([r_agg,l_agg])
    # v4 added
    rl_agg = rl_agg.groupby(level=0).sum()
    rl_agg.columns = alter_sc_exp.index
    return rl_agg


def chunk_cal_aff(adata, sc_dis_mat, lr_df):
    genes = list(adata.columns)
    lr_df = lr_df[lr_df[0].isin(genes) & lr_df[1].isin(genes)]
    gene_index =dict(zip(genes, range(len(genes))))
    index = lr_df.replace({0: gene_index, 1:gene_index}).astype(int)
    ligandindex = index[0].reset_index()[0]
    receptorindex = index[1].reset_index()[1]
    scores = index[2].reset_index()[2]
    Atotake = ligandindex
    Btotake = receptorindex
    allscores = scores
    idx_data = csr_matrix(adata).T #为什么不一开始就把sc_exp转为稀疏矩阵保存呢
    for i in range(len(ligandindex)):
        if ligandindex[i] != receptorindex[i]:
            Atotake = Atotake.append(pd.Series(receptorindex[i]),ignore_index=True)
            Btotake = Btotake.append(pd.Series(ligandindex[i]),ignore_index=True)
            allscores = allscores.append(pd.Series(scores[i]),ignore_index=True)
    A = idx_data[Atotake.tolist()]
    B = idx_data[Btotake.tolist()]
    full_A = np.dot(csr_matrix(np.diag(allscores)), A).T  
    chunk_size = 20
    cells = list(range(adata.shape[0]))
    affinitymat = np.array([[]]).reshape(0,adata.shape[0])
    affinitymat = csr_matrix(affinitymat)
    #s = time.time()

    for process_i in range(chunk_size):
        #a = time.time() 
        cell_chunk = list(np.array_split(cells, chunk_size)[process_i])
        chunk_A = full_A[cell_chunk]
        chunk_aff = np.dot(chunk_A, B)
        chunk_dis_mat = sc_dis_mat[cell_chunk]
        sparse_A = chunk_dis_mat.multiply(chunk_aff)
        #print(chunk_aff.sum())
        affinitymat = sparse.vstack([affinitymat, sparse_A])
        #b = time.time()
        #print(f'{process_i} done, cost {(b - a):.2f}s.')
    return affinitymat


def sc_prep(st_coord, sc_meta):
    # picked_sc_meta = sc_meta.copy()
    # broadcast_st_adj_sc
    '''没必要创建这么多字典，直接造个dataframe不就好了
    idx_lst = st_coord.index.tolist()
    idx_dict = {k: v for v, k in enumerate(idx_lst)}
    picked_sc_meta['indice'] = picked_sc_meta['spot'].map(idx_dict)
    coord_dict_x = {v: k for v, k in enumerate(list(st_coord['x']))}
    coord_dict_y = {v: k for v, k in enumerate(list(st_coord['y']))}
    picked_sc_meta['st_x'] = picked_sc_meta['indice'].map(coord_dict_x)
    picked_sc_meta['st_y'] = picked_sc_meta['indice'].map(coord_dict_y)
    picked_sc_meta = picked_sc_meta.sort_values(by = 'indice')
    # dist calculation
    sc_coord = picked_sc_meta[['st_x','st_y']]
    '''
    # 删掉空spot并返回待修改的sc坐标集
    st_coord = st_coord.loc[sc_meta['spot'].unique()]
    # 虽然这里写了个O(n)了，但是**或许**直接比前面copy一次开销还要小
    # 因为要用到st_coord还写不了apply + lambda了
    sc_range = range(len(sc_meta))
    sc_coord = pd.DataFrame(columns=['st_x', 'st_y'],index=list(sc_range))
    for i in sc_range:
        sc = sc_meta.iloc[i]
        coord = st_coord.loc[sc['spot']]
        sc_coord.loc[i] = [ coord.x, coord.y ]
    return st_coord, sc_coord


def sc_adj_cal(st_coord, picked_sc_meta,chunk_size = 12):
    alpha = 0
    # TODO: 好像多算了一次st_coord，先不管
    st_coord, sc_coord = sc_prep(st_coord, picked_sc_meta)
    # alpha = 0 for visum data
    all_x = np.sort(list(set(st_coord.iloc[:,0])))
    unit_len = all_x[1] - all_x[0]
    r = 2 * unit_len + alpha

    logging.debug("Calc ans...")
    indicator = lil_matrix((len(sc_coord),len(sc_coord)))
    ans = {}
    n_last_row = 0
    for process_i in range(chunk_size):
        X = np.array_split(np.array(sc_coord), chunk_size)[process_i]
        Y = sc_coord
        chunk = distance_matrix(X,Y)
        ans[process_i] = chunk
        neigh = [np.flatnonzero(d < r) for d in chunk]
        #print(process_i, chunk.shape)
        for i in range(len(neigh)):
            #print(i + n_last_row)
            indicator[i + n_last_row, neigh[i]] = 1
            #break
        n_last_row += chunk.shape[0]
        #break
    return st_coord, indicator, ans


def calcAffinityAns(st_coord, sc_meta, chunk_size = 12):
    alpha = 0
    # TODO: 好像多算了一次st_coord，先不管
    st_coord, sc_coord = sc_prep(st_coord, sc_meta)
    # alpha = 0 for visum data
    all_x = np.sort(list(set(st_coord.iloc[:,0])))
    unit_len = all_x[1] - all_x[0]
    r = 2 * unit_len + alpha

    logging.debug("Calc ans...")
    #indicator = lil_matrix((len(sc_coord),len(sc_coord)))
    #ans = {}
    #n_last_row = 0
    mat = distance_matrix(sc_coord, sc_coord)
    return np.array_split(mat, chunk_size)


# 计算distance矩阵，不过只计算各个细胞近邻Spot的，其他留0
def calcNeighborDistanceMat(spots_nn_lst, spot_cell_dict, sc_coord):
    sc_count = len(sc_coord)
    dis_mat = lil_matrix( (sc_count, sc_count) )

    for spot in spots_nn_lst:
        st_neighbors = [] + spots_nn_lst[spot]
        st_neighbors.append(spot)
        sc_neighbors = []
        for x in st_neighbors: sc_neighbors += spot_cell_dict[x]
        sc_myself = [] + spot_cell_dict[spot]
        # FIXME: 只对demo下的格式有效
        sc_neighbors = [int(k) for k in sc_neighbors]
        sc_myself = [int(k) for k in sc_myself]

        mat = distance_matrix( sc_coord[sc_myself], sc_coord[sc_neighbors] )
        for i in range(len(sc_myself)):
            for j in range(len(sc_neighbors)):
                v = mat[i, j]
                if v == 0:
                    v = 1e-50 # 为了corrcoef中形状相同
                dis_mat[sc_myself[i], sc_neighbors[j]] = v

    return dis_mat.tocsr()


def pear(D,D_re):
    tmp = np.corrcoef(D.flatten(order='C').astype('float64'), D_re.flatten(order='C').astype('float64'))
    return tmp[0,1] 

# https://stackoverflow.com/questions/19231268/correlation-coefficients-for-sparse-matrix-in-python
def sparse_corrcoef(A, B=None):

    if B is not None:
        A = sparse.vstack((A, B), format='csr')

    A = A.astype(np.float64)
    n = A.shape[1]

    # Compute the covariance matrix
    rowsum = A.sum(1)
    centering = rowsum.dot(rowsum.T.conjugate()) / n
    C = (A.dot(A.T.conjugate()) - centering) / (n - 1)

    # The correlation coefficients are given by
    # C_{i,j} / sqrt(C_{i} * C_{j})
    d = np.diag(C)
    coeffs = C / np.sqrt(np.outer(d, d))

    return coeffs


def coord_eva(sc_coord, spots_nn_lst, spot_cell_dict, sc_spot_center):
    logging.debug("Evaluating coord generated by UMAP...")
    sc_count = len(sc_coord)
    d1 = calcNeighborDistanceMat(spots_nn_lst, spot_cell_dict, sc_spot_center.to_numpy())
    d2 = calcNeighborDistanceMat(spots_nn_lst, spot_cell_dict, sc_coord)
    d1.reshape( (sc_count * sc_count, 1) )
    d2.reshape( (sc_count * sc_count, 1) )
    cor = sparse_corrcoef(d1[0], d2[0])[0, 1]
    logging.debug(f'shape correlation is: {cor}')
    return cor
    '''
    coord = np.array(coord)
    cor_all = 0
    for process_i in range(chunk_size):
        X = np.array_split(coord, chunk_size)[process_i]
        Y = coord
        chunk = distance_matrix(X,Y)
        cor = pear(ans[process_i], chunk)
        # print(cor)
        cor_all += cor
    logging.debug(f'Average shape correlation is: {cor_all/chunk_size}')
    return cor_all/chunk_size
    '''


def embedding(sparse_A, spots_nn_lst, spot_cell_dict, sc_spot_center, path, left_range = 0, right_range = 30, steps = 30, dim = 2, verbose = False):
    logging.debug("Preprocessing affinity matrix")
    aff = np.array(sparse_A, dtype = 'f')
    mask1 = (aff < 9e-300) & (aff >= 0)
    aff[mask1]=0.1
    np.fill_diagonal(aff,0)
    mask = aff != 0
    aff[mask] = 1 / aff[mask]
    #D = csr_matrix(aff) too less neighbor will occur
    del mask

    max_shape = 0
    for i in range(int(left_range),int(right_range)):
        for j in range(steps):
            coord = umap.UMAP(
                n_components = dim, metric = "precomputed", n_neighbors=(i+1)*15,
                random_state = 100 * j + 3).fit_transform(aff)
            cor = coord_eva(coord, spots_nn_lst, spot_cell_dict, sc_spot_center)
            if verbose:
                # save all reconstructed result
                pd.DataFrame(coord).to_csv(path + str(i) + '_' + str(j) + '.csv',index = False, header= False, sep = ',')
            # print(f'neighbor_{(i+1)*15}, random_{j} cor: {cor}')
            if cor > max_shape:
                max_shape = cor
                best_in_shape = coord
    if verbose:
        pd.DataFrame(best_in_shape).to_csv(path + 'coord_best.csv',index = False, header= False, sep = ',')
        print(f'max shape cor is {max_shape}')
    #print('Reached a correlation in shape at:', max_shape)
    return best_in_shape


def calculate_affinity_mat(lr_df, data):
    '''
    This function calculate the affinity matrix from TPM and LR pairs.
    '''
    # fetch the ligands' and receptors' indexes in the TPM matrix 
    # data.shape = gene * cell
    genes = data.index.tolist()
    lr_df = lr_df[lr_df[0].isin(genes) & lr_df[1].isin(genes)]
    # replace Gene ID to the index of each gene in data matrix #
    gene_index =dict(zip(genes, range(len(genes))))
    index = lr_df.replace({0: gene_index, 1:gene_index}).astype(int)

    ligandindex = index[0].reset_index()[0]
    receptorindex = index[1].reset_index()[1]
    scores = index[2].reset_index()[2]

    Atotake = ligandindex
    Btotake = receptorindex
    allscores = scores
    idx_data = data.reset_index()
    del idx_data[idx_data.columns[0]]
    
    for i in range(len(ligandindex)):
        if ligandindex[i] != receptorindex[i]:
            Atotake = Atotake.append(pd.Series(receptorindex[i]),ignore_index=True)
            Btotake = Btotake.append(pd.Series(ligandindex[i]),ignore_index=True)
            allscores = allscores.append(pd.Series(scores[i]),ignore_index=True)

    A = idx_data.loc[Atotake.tolist()]
    B = idx_data.loc[Btotake.tolist()]

    affinitymat = np.dot(np.dot(np.diag(allscores), A).T , B)
    
    return affinitymat


# 计算Affinity矩阵，不过只计算各个细胞近邻Spot的，其他留0
def calcNeighborAffinityMat(spots_nn_lst, spot_cell_dict, lr_df, sc_exp):
    sc_count = len(sc_exp.index.tolist())
    aff_mat = lil_matrix( (sc_count, sc_count) )

    # fetch the ligands' and receptors' indexes in the TPM matrix 
    # data.shape = gene * cell
    data = sc_exp.T
    genes = data.index.tolist()
    lr_df = lr_df[lr_df[0].isin(genes) & lr_df[1].isin(genes)]
    # replace Gene ID to the index of each gene in data matrix #
    gene_index = dict(zip(genes, range(len(genes))))
    index = lr_df.replace({0: gene_index, 1:gene_index}).astype(int)

    ligandindex = index[0].reset_index()[0]
    receptorindex = index[1].reset_index()[1]
    scores = index[2].reset_index()[2]

    Atotake = ligandindex
    Btotake = receptorindex
    allscores = scores
    idx_data = data.reset_index()
    del idx_data[idx_data.columns[0]]
    
    for i in range(len(ligandindex)):
        if ligandindex[i] != receptorindex[i]:
            Atotake = Atotake.append(pd.Series(receptorindex[i]),ignore_index=True)
            Btotake = Btotake.append(pd.Series(ligandindex[i]),ignore_index=True)
            allscores = allscores.append(pd.Series(scores[i]),ignore_index=True)

    # A = idx_data.loc[Atotake.tolist()]
    # B = idx_data.loc[Btotake.tolist()]

    # affinitymat = np.dot(np.dot(np.diag(allscores), A).T , B)

    score_diag = np.diag(allscores)
    for spot in spots_nn_lst:
        st_neighbors = [k for k in spots_nn_lst[spot]]
        st_neighbors.append(spot)
        sc_neighbors = []
        for x in st_neighbors: sc_neighbors += spot_cell_dict[x]
        sc_myself = spot_cell_dict[spot]
        # 计算sc_myself与sc_neighbor中所有的aff并填表
        A = idx_data[sc_myself].loc[Atotake]
        B = idx_data[sc_neighbors].loc[Btotake]

        # FIXME: 虽然节约内存了但是时间方面远不如之前的只进行一次dot
        # FIXME: 必须找一个好办法来循环进行这么多次dot操作
        # FIXME: multiprocessing在这里非常不好使 overhead太严重了
        affinitymat = np.dot(np.dot(score_diag, A).T , B)
        for i in range(len(sc_myself)):
            for j in range(len(sc_neighbors)):
                aff_mat[int(sc_myself[i]), int(sc_neighbors[j])] = affinitymat[i, j]

    return aff_mat.tocsr()


def aff_embedding(spots_nn_lst, spot_cell_dict, sc_exp,st_coord,sc_meta,lr_df,save_path, left_range = 1, right_range = 2, steps = 1, dim = 2,verbose = False):
    # 3.1 prep initial embedding that term3 requires
    logging.debug("Start embedding...")
    # ordered_st_coord, sc_dis_mat, ans = sc_adj_cal(st_coord, sc_meta, chunk_size = 12)
    #ans = calcAffinityAns(st_coord, sc_meta, chunk_size = 12)
    ########################print(f'Start affinity calculation...')
    logging.debug("Calc aff mat...")
    #sparse_A = chunk_cal_aff(sc_exp, sc_dis_mat, lr_df)
    sparse_A = calcNeighborAffinityMat(spots_nn_lst, spot_cell_dict, lr_df, sc_exp)
    sparse_A[sparse_A!=0] = sparse_A[sparse_A!=0] - 0.1
    sparse_A = sparse_A + np.ones(sparse_A.shape) * 0.1
    np.fill_diagonal(sparse_A,1)

    _, sc_spot_center = sc_prep(st_coord, sc_meta)
    #########################print(f'End affinity calculation.')
    coord = embedding(sparse_A, spots_nn_lst, spot_cell_dict, sc_spot_center, save_path, left_range, right_range, steps, dim, verbose = verbose)
    logging.debug("End embedding.")
    #return coord,ordered_st_coord,sparse_A,ans
    return coord, None, None, None


'''
def aff_embedding(sc_exp, spots_nn_lst, spot_cell_dict, lr_df, save_path, left_range = 1, right_range = 2, steps = 1, dim = 2, verbose = False):
    logging.debug("Start embedding...")
    aff = calcNeighborAffinityMat(spots_nn_lst, spot_cell_dict, lr_df, sc_exp)
    #aff[aff!=0] = aff[aff!=0] - 0.1
    #aff = aff + np.ones(aff.shape) * 0.1
    aff[aff == 0] = 0.1
    np.fill_diagonal(aff, 1)
    coord = embedding(aff, ans, save_path, left_range, right_range, steps, dim, verbose = verbose)
    logging.debug("End embedding.")
    return coord
'''


def get_hvg(adata, data_type = 'sc', n_top_genes = 3000):
    p_adata = sc.pp.normalize_total(adata, target_sum=1e4,copy = True)
    sc.pp.log1p(p_adata)
    if data_type == 'sc':
        sc.pp.highly_variable_genes(p_adata, flavor="seurat_v3",n_top_genes = n_top_genes)
    else:
        sc.pp.highly_variable_genes(p_adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    # sc.pl.highly_variable_genes(p_adata)
    p_adata = p_adata[:, p_adata.var.highly_variable]
    # adata.layers["log"] = p_adataUMAP
    return set(p_adata.var_names)


def center_shift_embedding(sc_coord, sc_meta_orig, max_dist):
    # added in v6
    '''
    shift cells belongs to each spot by their centroid and spot coordinates 
    sc_meta must have st_x and st_y
    sc_meta_orig = obj_spex.sc_meta
    sc_coord = obj_spex.sc_coord
    max_dist = 1
    '''
    ##### tailored for each spot #####
    sc_meta = sc_meta_orig.copy()
    sc_meta[['spex_UMAP1','spex_UMAP2']] = sc_coord
    umap_core = sc_meta.groupby('spot').mean()[['spex_UMAP1','spex_UMAP2']]
    idx_lst = umap_core.index.tolist()
    idx_dict = {k: v for v, k in enumerate(idx_lst)}
    coord_dict_x = {v: k for v, k in enumerate(list(umap_core['spex_UMAP1']))}
    coord_dict_y = {v: k for v, k in enumerate(list(umap_core['spex_UMAP2']))}
    sc_meta['indice'] = sc_meta['spot'].map(idx_dict)
    sc_meta['core1'] = sc_meta['indice'].map(coord_dict_x)
    sc_meta['core2'] = sc_meta['indice'].map(coord_dict_y)
    # calculating the unit length for the gap between two spot
    x_coors = np.sort(list(set(sc_meta['st_x'])))
    unit_len = x_coors[1] - x_coors[0]
    spot_space = unit_len/2
    # calculating the scale factor
    core_dist = pd.DataFrame(distance_matrix(sc_coord,umap_core))
    core_dist['spot'] = sc_meta['spot'].values
    max_center_dist = pd.DataFrame(np.diag(core_dist.groupby('spot').max()))
    scale_factor = list(spot_space/(max_center_dist[max_center_dist!=0])[0])
    scale_factor_dict = {v: k for v, k in enumerate(scale_factor)}
    sc_meta['centering_scale_factor'] = sc_meta['indice'].map(scale_factor_dict)
    #print(sc_meta['centering_scale_factor'].head(5))
    # center shift and scale
    tmp = sc_meta[['spex_UMAP1','spex_UMAP2']] - sc_meta[['core1','core2']].values
    tmp1 = tmp*(max_dist * sc_meta[['centering_scale_factor','centering_scale_factor']].values)
    sc_meta[['adj_spex_UMAP1','adj_spex_UMAP2']] = tmp1 + sc_meta[['st_x','st_y']].values
    for idx,row in sc_meta.iterrows():
        # add v9
        # if spot only have one cell, the scale factor would be nan
        if (row['core1'] - row['spex_UMAP1'] < 0.000001) and (row['core2'] - row['spex_UMAP2'] < 0.000001):
            sc_meta.loc[idx,'adj_spex_UMAP1'] = row['st_x']
            sc_meta.loc[idx,'adj_spex_UMAP2'] = row['st_y']
            sc_meta.loc[idx,'centering_scale_factor'] = 0
    return sc_meta
