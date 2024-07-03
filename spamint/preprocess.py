import pickle
import numpy as np
import pandas as pd
import anndata
# import time
# import logging
import os


def read_csv_tsv(filename):
    if ('csv' in filename) or ('.log' in filename):
        tmp = pd.read_csv(filename, sep = ',',header = 0,index_col=0)
    else:
        tmp = pd.read_csv(filename, sep = '\t',header = 0,index_col=0)
    return tmp


def scale_sum(x,SUM):
    res = x.divide(x.sum(axis = 1),axis=0)
    return res*SUM


def load_lr_df(species = 'Human',lr_dir = None):
    if lr_dir:
        lr_df = read_csv_tsv(lr_dir)
    else:
        lr_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/LR/'
        # print(lr_path)
        if species in ['Human','Mouse']:
            # to lowercase
            species = species.lower()
            lr_df = pd.read_csv(f'{lr_path}/{species}_LR_pairs.txt',sep='\t',header=None)
        else:
            raise ValueError(f'Currently only support Human and Mouse, get {species}')
    return lr_df


def make_adata(mat,meta,species,save_path = None, save_adata = False):
    # mat: exp matrix, should be cells x genes
    # index should be strictly set as strings
    meta.index = meta.index.map(str)
    mat.index = mat.index.map(str)
    mat = mat.loc[meta.index]
    adata = anndata.AnnData(mat,dtype=np.float32)
    adata.obs = meta
    adata.var = pd.DataFrame(mat.columns.tolist(), columns=['symbol'])
    adata.var_names = adata.var['symbol'].copy()
    #sc.pp.filter_cells(adata, min_genes=200)
    #sc.pp.filter_genes(adata, min_cells=3)
    # remove MT genes for spatial mapping (keeping their counts in the object)
    if species == 'Mouse':
        adata.var['MT_gene'] = [gene.startswith('mt-') for gene in adata.var['symbol']]
    if species == 'Human':
        adata.var['MT_gene'] = [gene.startswith('MT-') for gene in adata.var['symbol']]
    adata.obsm['MT'] = adata[:, adata.var['MT_gene'].values].X.toarray()
    adata = adata[:, ~adata.var['MT_gene'].values]
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = os.getcwd()
    
    figpath = save_path + '/figures/'
    
    if not os.path.exists(figpath):
        os.makedirs(figpath)

    if save_adata:
        adata.write(f'{save_path}/adata.h5ad')
    adata.uns['save_path'] = save_path
    adata.uns['species'] = species
    adata.uns['figpath'] = figpath
    return adata



def data_clean(sc_exp, st_exp):
    # cell x genes
    # 1. remove unexpressed genes
    filtered_sc = sc_exp.loc[:,(sc_exp != 0).any(axis=0)]
    filtered_st = st_exp.loc[:,(st_exp != 0).any(axis=0)]
    st_gene = set(filtered_st.columns)
    sc_gene = set(filtered_sc.columns)
    shared_genes = list(st_gene.intersection(sc_gene))
    filtered_sc1 = filtered_sc.loc[:,shared_genes]
    filtered_st1 = filtered_st.loc[:,shared_genes]
    return filtered_sc1, filtered_st1 


def denoise_genes(sc_exp, st_exp, sc_distribution,species):
    sc_genes = sc_exp.columns.tolist()
    st_genes = st_exp.columns.tolist()
    genes = list(set(sc_genes).intersection(set(st_genes)))
    genes = list(set(genes).intersection(set(sc_distribution.columns)))

    if species == 'Mouse':
        mt = [gene for gene in genes if gene.startswith('mt-')]
    if species == 'Human':
        mt = [gene for gene in genes if gene.startswith('MT-')]
    genes = list(set(genes).difference(set(mt)))
    genes.sort()
    return genes


def subset_inter(st_exp, sc_exp):
    '''
    subset df by the intersection genes between st and sc
    '''
    genes = list(set(st_exp.columns).intersection(set(sc_exp.columns)))
    st_exp = st_exp[genes]
    sc_exp = sc_exp[genes]
    return st_exp, sc_exp


def prep_all_adata(sc_exp = None, st_exp = None, sc_distribution = None, 
                   sc_meta = None, st_coord = None, lr_df = None, SP = 'Human'):
    '''
    1. remove unexpressed genes
    2. select shared genes
    3. transform to adata format
    '''
    # scale all genes to [0,10]
    # v5 
    # SUM = st_exp.sum(axis = 1).mean()
    # v6 from st sum to 1e4
    if (SP != 'Human') and (SP != 'Mouse'):
        raise ValueError(
            f'Species should be choose among either Human or Mouse.')
    SUM = 1e4
    # Data Clean
    sc_exp, st_exp = data_clean(sc_exp, st_exp)
    genes = denoise_genes(sc_exp, st_exp, sc_distribution, SP)
    sc_exp = sc_exp[genes]
    st_exp = st_exp[genes]
    sc_distribution = sc_distribution[genes]
    # TODO：这个LR_DF后面又多算了好多次
    lr_df = lr_df[lr_df[0].isin(genes) & lr_df[1].isin(genes)]
    # Adata Preparation
    # 1. SC to adata
    scale_sc_exp = scale_sum(sc_exp,SUM)
    sc_adata = make_adata(scale_sc_exp,sc_meta,SP)
    # 2. ST to adata
    scale_st_exp = scale_sum(st_exp,SUM)
    st_adata = make_adata(scale_st_exp,st_coord,SP)
    # 3. distribution to adata
    sc_ref = scale_sum(sc_distribution,SUM)
    # v6 canceled ref adata
    # sc_ref = prep_adata(scale_poisson_spot,sc_ref_meta,SP)
    if sc_adata.shape[1] == st_adata.shape[1] and st_adata.shape[1] == sc_ref.shape[1]:
        print(f'Data clean is done! Using {st_adata.shape[1]} shared genes .')
    return sc_adata, st_adata, sc_ref, lr_df



def prep_all_adata_merfish(sc_exp = None, st_exp = None, sc_distribution = None, 
                   sc_meta = None, st_coord = None, lr_df = None, SP = 'human'):
    '''
    1. remove unexpressed genes
    2. align genes with sc
    3. transform to adata format
    '''
    # scale all genes to [0,10]
    # v5 
    # SUM = st_exp.sum(axis = 1).mean()
    # v6 from st sum to 1e4
    if (SP != 'Human') and (SP != 'Mouse'):
        raise ValueError(
            f'Species should be choose among either human or mouse.')
    SUM = 1e4
    # Data Clean
    filtered_sc = sc_exp.loc[:,(sc_exp != 0).any(axis=0)]
    genes = list(set(filtered_sc.columns).intersection(set(sc_distribution.columns)))
    # Align genes to SC
    sc_exp = sc_exp[genes]
    # st_exp = st_exp.reindex(genes, axis=1)
    lr_df = lr_df[lr_df[0].isin(genes) & lr_df[1].isin(genes)]
    # Adata Preparation
    # 1. SC to adata
    scale_sc_exp = scale_sum(sc_exp,SUM)
    sc_adata = make_adata(scale_sc_exp,sc_meta,SP)
    # 2. ST to adata
    # 1e5 is too large for merfish data, merfish only has 200~500 genes, gene sum is around 1e3.
    st_gene_sum = int(st_exp.sum(axis = 1).mean())
    st_genes = list(set(st_exp.columns).intersection(set(genes)))
    st_exp = st_exp[st_genes]
    scale_st_exp = scale_sum(st_exp,st_gene_sum)
    st_adata = make_adata(scale_st_exp,st_coord,SP)
    # 3. distribution to adata
    # sc_adata filtered mt genes in prep_adata
    sc_distribution = sc_distribution[sc_adata.var_names.tolist()]
    sc_ref = scale_sum(sc_distribution,SUM)
    # v6 canceled ref adata
    # sc_ref = prep_adata(scale_poisson_spot,sc_ref_meta,SP)
    if sc_adata.shape[1] == sc_ref.shape[1]:
        print(f'Data clean and scale are done! Single-cell data has {sc_adata.shape[1]} genes, spatial data has {st_adata.shape[1]} genes.')
    return sc_adata, st_adata, sc_ref, lr_df




def lr2kegg(lri_df, use_lig_gene = True, use_rec_gene = True):
    '''
    Use both ligand and receptor
    '''
    if use_lig_gene:
        a = lri_df[['ligand','lr_co_exp_num','lr_co_ratio_pvalue']]
        a.columns = ['gene','lr_co_exp_num','lr_co_ratio_pvalue']
    else:
        a = pd.DataFrame(columns = ['gene','lr_co_exp_num','lr_co_ratio_pvalue'])

    if use_rec_gene:
        b = lri_df[['receptor','lr_co_exp_num','lr_co_ratio_pvalue']]
        b.columns = ['gene','lr_co_exp_num','lr_co_ratio_pvalue']
    else:
        b = pd.DataFrame(columns = ['gene','lr_co_exp_num','lr_co_ratio_pvalue'])
    c = pd.concat((a,b))
    c = c.groupby('gene').mean().reset_index()
    return c


def filter_kegg(df, pval_thred = 0.05):
    tmp = df.copy()
    tmp = tmp[tmp['pvalue'] < pval_thred].copy()
    tmp['-log10 pvalue'] = np.log10(tmp['pvalue']) * (-1)
    tmp[['tmp1','tmp2']] = tmp['GeneRatio'].str.split('/',expand=True)
    tmp['GeneRatio'] = tmp['tmp1'].astype(int) / tmp['tmp2'].astype(int)
    tmp['Count'] = tmp['Count'].astype(int)
    tmp['-log10 pvalue'] = tmp['-log10 pvalue'].astype(float)
    tmp = tmp.sort_values('GeneRatio',ascending=False)
    # remove suffix
    tmp['Description'] = tmp['Description'].str.split(' - ', expand=True)[0]
    return tmp



