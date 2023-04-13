import scanpy as sc

def plot_mds(adata, basis='dm',color=None, **kwargs):
    sc.pl.embedding(adata, basis,color=color, **kwargs)
