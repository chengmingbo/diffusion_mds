import scanpy as sc

def diffusion_mds(adata, basis='dm',color=None, **kwargs):
    sc.pl.embedding(adata, basis,color=color, **kwargs)
