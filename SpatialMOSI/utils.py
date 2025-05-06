import numpy as np
import pandas as pd

import csv
import sklearn.neighbors
import networkx as nx
import random
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import squidpy as sq
import scanpy as sc
import scipy
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import anndata as ad

from annoy import AnnoyIndex
import itertools
import networkx as nx
import hnswlib

def create_csp_dict(adata, use_rep, batch_name, k = 50, save_on_disk = True, rough = True, show = 0, csp_groups = None):
    """
    Create a dictionary of cross-slice pairs (CSP) for batches in the dataset.

    Parameters:
    - adata: AnnData object containing the data.
    - use_rep: Key in adata.obsm to use as the representation (e.g., 'omic').
    - batch_name: Column name in adata.obs that contains batch information.
    - k: Number of nearest neighbors to consider.
    - save_on_disk: Whether to save the index on disk.
    - rough: Whether to use csps roughSly.
    - show: Whether to show the csps.
    - csp_groups: List of CSP groups.

    Returns:
    - Dictionary of cross-slice pairs.
    """
    cell_names = adata.obs_names
    batch_list = adata.obs[batch_name]
    datasets = []
    datasets_pcs = []
    cells = []
    for i in batch_list.unique():
        datasets.append(adata[batch_list == i])
        datasets_pcs.append(adata[batch_list == i].obsm[use_rep])
        cells.append(cell_names[batch_list == i])

    batch_name_df = pd.DataFrame(np.array(batch_list.unique()))
    csps = dict()

    if csp_groups is None:
        csp_groups = list(itertools.combinations(range(len(cells)), 2))
    for csp in csp_groups:
        i = csp[0]
        j = csp[1]
        key_name1 = batch_name_df.loc[csp[0]].values[0] + "_" + batch_name_df.loc[csp[1]].values[0]
        csps[key_name1] = {}
        if(show > 0):
            print('Processing datasets {}'.format((i, j)))

        que_data = list(cells[j])
        ref_data = list(cells[i])
        batch1_data = adata[que_data].obsm[use_rep]
        batch2_data = adata[ref_data].obsm[use_rep]


        match = csp(ds1, ds2, que_data, ref_data, knn=k, save_on_disk = save_on_disk, approx = approx)

        G = nx.Graph()
        G.add_edges_from(match)
        node_names = np.array(G.nodes)
        anchors = list(node_names)
        adjacency_matrix = nx.adjacency_matrix(G)
        tmp = np.split(adjacency_matrix.indices, adjacency_matrix.indptr[1:-1])

        for anchor_index in range(0, len(anchors)):
            key = anchors[anchor_index]
            anchor_index = tmp[anchor_index]
            names = list(node_names[anchor_index])
            csps[key_name1][key]= names
    return(csps)

def csp_rough(batch1, batch2, cell1, cell2, kcsp=50):
    """
    Exact nearest neighbors using scikit-learn.

    Parameters:
    - batch1: batch 1.
    - batch2: batch 2.
    - cell1: cells in batch 1.
    - cell2: cells in batch 2.
    - kcsp: Number of csp to consider.

    Returns:
    - Set of matched pairs.
    """
    dim = batch2.shape[1]
    num_elements = batch2.shape[0]
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=100, M=16)
    p.set_ef(10)
    p.add_items(batch2)
    ind, distances = p.knn_query(batch1, k=kcsp)
    match = set()
    for i, j in zip(range(batch1.shape[0]), ind):
        for jj in j:
            match.add((cell1[i], cell2[jj]))
    return match


def csp_nn(batch1, batch2, cell1, cell2, kcsp=50, metric_p=2):
    """
    Exact nearest neighbors using scikit-learn.

    Parameters:
    - batch1: batch 1.
    - batch2: batch 2.
    - cell1: cells in batch 1.
    - cell2: cells in batch 2.
    - kcsp: Number of nearest neighbors to consider.
    - metric_p: Power parameter for the Minkowski metric.

    Returns:
    - Set of matched pairs.
    """
    nn_obj = NearestNeighbors(kcsp, p=metric_p)
    nn_obj.fit(batch2)
    ind = nn_obj.kneighbors(batch1, return_distance=False)
    match = set()
    for i, j in zip(range(batch1.shape[0]), ind):
        for jj in j:
            match.add((cell1[i], cell2[jj]))
    return match


def csp(batch1, batch2, cell1, cell2, kcsps=20, save_on_disk=True, rough=True):
    if rough: 
        # Find nearest neighbors in first direction.
        # output KNN point for each point in ds1.  match1 is a set(): (points in names1, points in names2), the size of the set is ds1.shape[0]*knn
        match1 = csp_rough(batch1, batch2, cell1, cell2, kcsps=kcsp)#, save_on_disk = save_on_disk)
        # Find nearest neighbors in second direction.nn_approx
        match2 = csp_rough(ds2, ds1, cell1, cell2, kcsps=kcsp)#, save_on_disk = save_on_disk)
    else:
        match1 = csp_nn(batch1, batch2, cell1, cell2, kcsps=kcsp)
        match2 = csp_nn(batch2, batch1, cell2, cell1, kcsps=kcsp)
    # Compute mutual nearest neighbors.
    return match1 & set([(j, i) for i, j in match2])

def csp_nn(batch1, batch2, cell1, cell2, knn=50, metric_p=2):
    """
    Exact nearest neighbors using scikit-learn.

    Parameters:
    - batch1: batch 1.
    - batch2: batch 2.
    - cell1: cells in batch 1.
    - cell2: cells in batch 2.
    - knn: Number of nearest neighbors to consider.
    - metric_p: Power parameter for the Minkowski metric.

    Returns:
    - Set of matched pairs.
    """
    nn_obj = NearestNeighbors(knn, p=metric_p)
    nn_obj.fit(batch2)
    ind = nn_obj.kneighbors(batch1, return_distance=False)
    match = set()
    for i, j in zip(range(batch1.shape[0]), ind):
        for jj in j:
            match.add((cell1[i], cell2[jj]))
    return match



def lsi(
        adata, n_components: int = 20,
        use_highly_variable=False, **kwargs
       ) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    #X = adata_use.X
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    #adata.obsm["X_lsi"] = X_lsi
    adata.obsm["X_lsi"] = X_lsi[:,1:]

def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf   

def clr_normalization(adata):
    if sp.issparse(adata.X):
        GM = np.exp(np.sum(np.log(adata.X.toarray() + 1)/adata.X.shape[1], axis = 1))
        adata.X = np.log(adata.X.toarray()/GM[:, None] + 1)
    else:
        GM = np.exp(np.sum(np.log(adata.X + 1)/adata.X.shape[1], axis = 1))
        adata.X = np.log(adata.X/GM[:, None] + 1)
    return adata

def generate_spatial_graph(adata, radius=None, knears=None, self_loops:bool=False):
    '''
    radius: the radius when selecting neighors
    knears: the number of neighbors
    method: the method to generate graph: radius or knn
    '''
    #knears=0
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['row', 'col']
    if (radius):
        nbrs = NearestNeighbors(radius = radius).fit(coor)
        distance, indices = nbrs.radius_neighbors(coor, return_distance=True)
    elif (knears):
        nbrs = NearestNeighbors(n_neighbors=knears).fit(coor)
        distance, indices = nbrs.kneighbors(coor)
    else:
        raise ValueError("method Error:radius or knn!")
    edge_list = []
    if(self_loops):
        for i in range(len(indices)):
            for j in range(len(indices[i])):
                edge_list.append([i, indices[i][j]])  
    else:
        for i in range(len(indices)):
            for j in range(len(indices[i])):
                if (i != indices[i][j]):
                    edge_list.append([i, indices[i][j]])  
    edge_list = np.array(edge_list)
    print('graph includs edges:',len(edge_list),'average edges per node:', "{:.3f}".format(len(edge_list) / adata.n_obs))

    edge_list = edge_list.transpose()
    #adata.uns['graph']=edge_list
    return edge_list
    
def generate_csl_graph(num_nodes, kner):
    '''
    Generate a graph randomly.
    num_nodes:  the number of nodes in the graph
    kner:       the number of edges per node
    '''
    num_edges = num_nodes * kner
    edge_index = np.zeros((2, num_edges), dtype=int)
    edge_set = set()

    for i in range(num_nodes):
        indices = random.sample(range(num_nodes), kner)
        for j in indices:
            if j != i and (i, j) not in edge_set:
                edge_set.add((i, j))
                edge_index[:, len(edge_set)-1] = [i, j]

    sorted_index = np.argsort(edge_index[0])
    edge_index = edge_index[:, sorted_index]
    mask = (edge_index[0] == 0) & (edge_index[1] == 0)
    mask = ~mask
    edge_index = edge_index[:,mask]
    print("Negative spots selection completed!")
    return edge_index

def load_data(path, slice_ids, omics=['RNA','ADT'], radius=0.2, self_loops=True,
        n_csl=5, omics_HVGs=[3000,3000], n_norm=[1e4, 10],chr_hvgs = False, n_lsi=51
        ):
    Batch_list_omic1 = []
    Batch_list_omic2 = []
    adj_list = np.empty((2,0))
    adj_list = adj_list.astype(int)
    csl_list = np.empty((2,0))
    csl_list = csl_list.astype(int)
    n_obs_all = 0
    for slice_id in slice_ids:
        adata_omic1 = sc.read_h5ad(f'{path}/{slice_id}/adata_RNA.h5ad')
        adata_omic2 = sc.read_h5ad(f'{path}/{slice_id}/adata_ADT.h5ad')
        
        adata_omic1.var_names_make_unique(join="++")
        adata_omic2.var_names_make_unique(join="++")
        adata_omic1.obs_names=[x +'_'+ omics[0] +'_'+ slice_id for x in adata_omic1.obs_names]
        adata_omic2.obs_names=[x +'_'+ omics[1] +'_'+ slice_id for x in adata_omic2.obs_names]
        adata_omic1.uns['adj']=generate_spatial_graph(adata_omic1, radius=radius,self_loops=True) +n_obs_all
        adata_omic1.uns['edgecsl'] = generate_csl_graph(adata_omic1.n_obs, n_csl) + n_obs_all
        n_obs_all = n_obs_all + adata_omic1.n_obs
        csl_list = np.concatenate((csl_list, adata_omic1.uns['edgecsl']),axis = 1)
        adj_list = np.concatenate((adj_list, adata_omic1.uns['adj']),axis = 1)
        #adata_omic2.uns['adj'] = adata_omic1.uns['adj']
        #adata_omic2.uns['edgecsl'] = adata_omic1.uns['edgecsl']
        sc.pp.highly_variable_genes(adata_omic1, flavor="seurat_v3", n_top_genes=omics_HVGs[0])
        sc.pp.highly_variable_genes(adata_omic2, flavor="seurat_v3", n_top_genes=omics_HVGs[1])
        
        
        sc.pp.normalize_total(adata_omic1, target_sum=n_norm[0])
        sc.pp.normalize_total(adata_omic2, target_sum=n_norm[1])
        sc.pp.log1p(adata_omic1)
        sc.pp.log1p(adata_omic2)
        adata_omic1 = adata_omic1[:, adata_omic1.var['highly_variable']]

        if omics[0]=='CHR':
            lsi(adata_omic1, use_highly_variable=chr_hvgs, n_components=n_lsi)
        elif omics[1]=='CHR':
            lsi(adata_omic2, use_highly_variable=chr_hvgs, n_components=n_lsi)

        adata_omic2 = adata_omic2[:, adata_omic2.var['highly_variable']]

        Batch_list_omic1.append(adata_omic1)
        Batch_list_omic2.append(adata_omic2)
        '''
        if omics_counts == 3:
            adata_omic3 = sc.read_h5ad(f'{path}/{slice_id}/adata_CHR.h5ad')
            adata_omic3.var_names_make_unique(join="++")
            adata_omic3.obs_names=[x +'_'+ omics[2] +'_'+ slice_id for x in adata_omic3.obs_names]
            sc.pp.highly_variable_genes(adata_omic3, flavor="seurat_v3", n_top_genes=omics_HVGs[2])
            adata_omic3 = adata_omic3[:, adata_omic3.var['highly_variable']]
            if omics[2]=='CHR':
                lsi(adata_omic3, use_highly_variable=chr_hvgs, n_components=n_lsi)
        '''     
    return Batch_list_omic1, Batch_list_omic2, adj_list, csl_list

def concat(Batch_list_omic1,adj_list, csl_list,slice_ids=['S1_RNA','S2_RNA']):
    adata_concat = ad.concat(Batch_list_omic1, label="batch_name", keys=slice_ids)
    adata_concat.uns['edgecsl']=csl_list
    adata_concat.uns['adj']=adj_list
    print('concat matrix shape:',adata_concat.shape)
    return adata_concat

import numpy as np
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity

def evaluate_metrics(matrix1, matrix2):
    """
    计算两个表达矩阵的 JS、PCC、SPCC、SSIM 和 RMSE 指标
    
    参数:
    matrix1 (numpy.ndarray): 第一个表达矩阵
    matrix2 (numpy.ndarray): 第二个表达矩阵
    
    返回:
    dict: 包含计算得到的五个指标的字典
    """
    def normalize_matrix(matrix):
        """
        对矩阵进行归一化处理
        """
        row_sums = matrix.sum(axis=1)
        return matrix / row_sums[:, np.newaxis]
    
    def js_divergence(matrix1, matrix2):
        """
        计算两个表达矩阵的 JS 散度
        """
        p = normalize_matrix(matrix1)
        q = normalize_matrix(matrix2)
        m = 0.5 * (p + q)
        js = 0.5 * (entropy(p, m, axis=1) + entropy(q, m, axis=1))
        return np.mean(js)
    
    def spcc(matrix1, matrix2):
        """
        计算两个表达矩阵的样本皮尔逊相关系数（SPCC）
        """
        num_genes1, num_cells1 = matrix1.shape
        num_genes2, num_cells2 = matrix2.shape
        if num_cells1!= num_cells2:
            raise ValueError("The number of cells in both matrices must be the same.")
        spcc_matrix = np.zeros((num_genes1, num_genes2))
        for i in range(num_genes1):
            for j in range(num_genes2):
                spcc_matrix[i, j] = np.corrcoef(matrix1[i, :], matrix2[j, :])[0, 1]
        return spcc_matrix
    
    def rmse(matrix1, matrix2):
        """
        计算两个表达矩阵的均方根误差（RMSE）
        """
        return np.sqrt(np.mean((matrix1 - matrix2) ** 2))
    
    num_genes1, num_cells1 = matrix1.shape
    num_genes2, num_cells2 = matrix2.shape
    if num_genes1!= num_genes2 or num_cells1!= num_cells2:
        raise ValueError("The matrices must have the same dimensions.")
    
    # 计算 JS 散度
    js_result = js_divergence(matrix1, matrix2)
    # 计算 PCC
    pcc_result, _ = pearsonr(matrix1.flatten(), matrix2.flatten())
    # 计算 SPCC
    #spcc_result = spcc(matrix1, matrix2)
    # 计算 SSIM
    matrix1_image = np.reshape(matrix1, (num_genes1, num_cells1))
    matrix2_image = np.reshape(matrix2, (num_genes2, num_cells2))
    ssim_result = structural_similarity(matrix1_image, matrix2_image)
    # 计算 RMSE
    rmse_result = rmse(matrix1, matrix2)
    
    metrics = {
        "JS": js_result,
        "PCC": pcc_result,
        "SSIM": ssim_result,
        "RMSE": rmse_result
    }
    return metrics

