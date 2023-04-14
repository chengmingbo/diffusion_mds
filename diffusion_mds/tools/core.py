import numpy as np
import networkx as nx
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from typing import List
from scipy.spatial import distance_matrix, Delaunay, distance
from sklearn.manifold import MDS
from anndata import AnnData

from .diffusionmap import diffusionMaps
from .lmds import LMDS

def stretching(dm, n_neighbors=8, random_state=2023, prog='sfdp', verbose=False):

    """
    use knn to add edges
    next apply graphy layout to display all nodes
    """
    def ti(a,b):
        if a < b:
            return (a,b)
        return(b,a)

    if verbose:
        print(datetime.now(),'Triangulating graph...')
    tri = Delaunay(dm)
    tri_edges =[[ti(a,b),ti(a,c),ti(b,c)] for a,b,c in tri.simplices]
    tri_edges = list(set([item for sublist in tri_edges for item in sublist])) # flatten
    edges_distance = [distance.euclidean(tuple(dm[a]),tuple(dm[b])) for (a,b) in tri_edges]
    trunc_quantile=0.6
    trunc_times=3
    while True: ## only connected graph approved
        threshold = np.quantile(edges_distance, trunc_quantile) * trunc_times
        keep_edges = [tri_edges[i] for i in range(len(tri_edges)) if edges_distance[i] < threshold]
        tmpG = nx.Graph()
        tmpG.add_nodes_from(range(dm.shape[0]))
        tmpG.add_edges_from(keep_edges)
        if nx.is_connected(tmpG):
            break
        else:
            trunc_quantile += 0.05
            if trunc_quantile>= 1:
                print(datetime.now(),"warning: failed to add connected Delaunay, use the original embedding")
                return dm

            threshold = np.quantile(edges_distance, trunc_quantile) * trunc_times
        if trunc_quantile >= 1:
            print(datetime.now(),"warning: failed to add connected Delaunay, use the original embedding")
            return dm

    G = nx.Graph()
    G.add_nodes_from(range(dm.shape[0]))
    G.add_edges_from(keep_edges)
    if verbose:
        print(datetime.now(),f'Running graph layout {prog}...')
    layouts = nx.nx_pydot.graphviz_layout(G, prog=prog)
    dm = np.array([layouts[i] for i in range(dm.shape[0])])

    return dm


def diffusion_mds_embedding(dm:np.ndarray,
                  dims:List =None,
                  diffusion_components=4,
                  affinity_k=7,
                  affinity_sigma=None,
                  random_state=2023,
                  noise_sigma_ratio=0.01,
                  landmark_mds= 0.1,
                  stretch = False,
                  prog='sfdp',
                  verbose=False,
                  ):

    """
    Diffusion MDS embedding
    add gaussian noise to alleviate overlapping
    stretch: if True, use stretch to furture alleviate overlapping
    """
    np.random.seed(random_state)
    if dims is None:
        dims = range(dm.shape[1])

    if verbose:
        print(datetime.now(),'Calculating distances...')
    R = distance_matrix(dm[:, dims], dm[:, dims])

    ##1.  run diffusion maps
    if verbose:
        print(datetime.now(),'Running diffusion maps...')
    d = diffusionMaps(R,k=affinity_k,sigma=affinity_sigma, eig_k=diffusion_components+1, verbose=verbose)

    if verbose:
        print(datetime.now(),'Running MDS...')
    ##2. run mds on diffusion components to retain the distances
    if landmark_mds >= 1:
        mds  = MDS(n_components=2, random_state=random_state)
        mds_dm = mds.fit_transform(d['psi'][:, 1:])
    else:
        mds_dm = LMDS(d['psi'][:, 1:], landmark=landmark_mds, random_state=random_state)


    ##3. add guassian noise to the mds embedding
    if verbose:
        print(datetime.now(),'Adding noise...')
    rg1 = np.max(mds_dm[:, 0]) - np.min(mds_dm[:, 0])
    rg2 = np.max(mds_dm[:, 1]) - np.min(mds_dm[:, 1])
    m1 = np.mean(mds_dm[:, 0])
    m2 = np.mean(mds_dm[:, 1])

    X_noise=np.random.normal(m1, noise_sigma_ratio*rg1, size=mds_dm.shape[0])
    Y_noise=np.random.normal(m2, noise_sigma_ratio*rg2, size=mds_dm.shape[0])

    mds_dm[:, 0] += X_noise
    mds_dm[:, 1] += Y_noise

    if stretch:
        if verbose:
            print(datetime.now(),'Adding stretch...')
        mds_dm = stretching(mds_dm, prog=prog,random_state=random_state, verbose=verbose)

    return mds_dm

def diffusion_mds(adata:AnnData,
                  basis:str='pca',
                  output_basis:str='dm',
                  dims:List =None,
                  diffusion_components=4,
                  affinity_k=7,
                  affinity_sigma=None,
                  random_state=2023,
                  noise_sigma_ratio=0.01,
                  landmark_mds = 0.1,
                  stretch = False,
                  prog = 'sfdp',
                  verbose=False,
                  copy = False):

    adata = adata.copy() if copy else adata

    dm = adata.obsm['X_'+basis]
    if dims is None:
        dims = range(dm.shape[1])
    mds_dm = diffusion_mds_embedding(dm=dm,
                                     dims=dims,
                                     diffusion_components=diffusion_components,
                                     affinity_k=affinity_k,
                                     affinity_sigma=affinity_sigma,
                                     random_state=random_state,
                                     noise_sigma_ratio=noise_sigma_ratio,
                                     landmark_mds=landmark_mds,
                                     stretch=stretch,
                                     prog=prog,
                                     verbose=verbose,
                                     )

    adata.obsm['X_'+output_basis] = mds_dm

    return adata if copy else None
