import numpy as np
from datetime import datetime
from scipy.sparse import linalg


def logsumexp(x):
    maxx=max(x)
    return maxx + np.log(np.sum(np.exp(x - maxx)))

def affinity(R, k=7, sigma=None, log=False, normalize=False):
    """
    Gaussian affinity matrix constructor
    W = exp(-r_{ij}^2/sigma)

    Parameter
    -----------
    R: symmetric matrix(positive semi-definite); Distance matrix
    k: number of neighbors in adaptive-scaling, ignore if sigma is not None
    log: transform the affinity by logrithm or not
    normalize: return transition matrix
    """
    def top_k(lst, k=1):
        assert(len(lst) >k)
        return np.partition(lst, k)[k]
    R = np.array(R)
    if not sigma:
        s = [top_k(R[:, i], k=k)  for i in range(R.shape[1])]
        S = np.sqrt(np.outer(s, s))
    else:
        S = sigma
    logW = -np.power(np.divide(R, S), 2)

    if normalize:
        denominator = [logsumexp(logW[i,:]) for i in range(logW.shape[0])]
        logW = np.divide(logW.T, denominator).T
    if log:
        return logW
    return np.exp(logW)


def  diffusionMaps(R,k=7,sigma=None, eig_k=10, verbose=False):
    """
    Diffusion map(Coifman, 2005)
    https://en.wikipedia.org/wiki/Diffusion_map
    Parameter
    ----------
    R: distance matrix
    k: number of neighbors in adaptive-scaling
    sigma: for isotropic diffussion
    Return
    ----------
    dic:
        psi: right eigvector of P = D^{-1/2} * evec
        phi: left eigvector of P = D^{1/2} * evec
        eig: eigenvalues
    """
    k=k-1 ## k is R version minus 1 for the index
    if verbose:
        print(datetime.now(), "Affinity matrix construction...")

    logW = affinity(R,k,sigma,log=True,normalize=False)
    rs = np.exp([logsumexp(logW[i,:]) for i in range(logW.shape[0])]) ## dii=\sum_j w_{i,j}
    D = np.diag(np.sqrt(rs))        ## D^{1/2}
    Dinv = np.diag(1/np.sqrt(rs))   ##D^{-1/2}
    Ms = Dinv @ np.exp(logW) @ Dinv ##

    if verbose:
        print(datetime.now(), "Eigen decomposition...")

    e = linalg.eigsh(Ms, k=eig_k) ## eigen decomposition of P'
    evalue= e[0][::-1]
    evec = np.flip(e[1], axis=1)
    s = np.sum(np.sqrt(rs) * evec[:,0]) # scaling
    #0:Psi, 1:Phi, 2:eig
    dic = {'psi':s * Dinv@evec, 'phi': (1/s)*D@evec, "eig": evalue}
    return dic
