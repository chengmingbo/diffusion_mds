import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from typing import List
from scipy.spatial import distance_matrix
from sklearn.manifold import MDS
from anndata import AnnData
from .diffusionmap import diffusionMaps
from .lmds import LMDS

def add_broadview(dm, n_neighbors=8, random_state=2023):

    """
    use knn to add edges
    next apply graphy layout to display all nodes
    """
    neigh = NearestNeighbors(n_neighbors=max(n_neighbors, 30))
    neigh.fit(dm)
    neighbors = neigh.kneighbors(dm)[1]
    for n_neighbors_choose in range(n_neighbors, 30):
        edges = [(neighbors[i,0], j) for i in range(n_neighbors_choose) for j in neighbors[i, 1:]]
        G = nx.Graph()
        G.add_edges_from(edges)
        if nx.is_connected(G):
            break

    if not nx.is_connected(G):
        print("warning: failed to add broadview, use the original embedding")
        return dm
    layouts = nx.nx_pydot.graphviz_layout(G, prog='sfdp')
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
                  broadview = False,
                  broadview_k = 8,
                  verbose=False,
                  ):

    """
    Diffusion MDS embedding
    add gaussian noise to alleviate overlapping
    broadview: if True, use broadview to furture alleviate overlapping
    """
    if dims is None:
        dims = range(dm.shape[1])

    if verbose:
        print('Calculating distances...')
    R = distance_matrix(dm[:, dims], dm[:, dims])

    ##1.  run diffusion maps
    if verbose:
        print('Running diffusion maps...')
    d = diffusionMaps(R,k=affinity_k,sigma=affinity_sigma, eig_k=diffusion_components+1, verbose=verbose)

    if verbose:
        print('Running MDS...')
    ##2. run mds on diffusion components to retain the distances
    #slow version
    #mds  = MDS(n_components=2, random_state=random_state)
    #mds_dm = mds.fit_transform(d['psi'][:, 1:])
    if landmark_mds >= 1:
        mds  = MDS(n_components=2, random_state=random_state)
        mds_dm = mds.fit_transform(d['psi'][:, 1:])
    else:
        mds_dm = LMDS(d['psi'][:, 1:], landmark=landmark_mds, random_state=random_state)


    ##3. add guassian noise to the mds embedding
    if verbose:
        print('Adding noise...')
    rg1 = np.max(mds_dm[:, 0]) - np.min(mds_dm[:, 0])
    rg2 = np.max(mds_dm[:, 1]) - np.min(mds_dm[:, 1])
    m1 = np.mean(mds_dm[:, 0])
    m2 = np.mean(mds_dm[:, 1])

    X_noise=np.random.normal(m1, noise_sigma_ratio*rg1, size=mds_dm.shape[0])
    Y_noise=np.random.normal(m2, noise_sigma_ratio*rg2, size=mds_dm.shape[0])

    mds_dm[:, 0] += X_noise
    mds_dm[:, 1] += Y_noise

    if broadview:
        if verbose:
            print('Adding broadview...')
        mds_dm = add_broadview(mds_dm, n_neighbors=broadview_k, random_state=random_state)

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
                  broadview = False,
                  broadview_k = 8,
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
                                     broadview=broadview,
                                     broadview_k=broadview_k,
                                     verbose=verbose,
                                     )

    adata.obsm['X_'+output_basis] = mds_dm

    return adata if copy else None