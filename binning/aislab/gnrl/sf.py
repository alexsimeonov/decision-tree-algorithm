"""
author: OPEN-MAT
date: 	15.06.2019
Matlab version: 
    author: Alexander Efremov
    date:   26 Apr 2009
    course: Multivariable Control Systems
"""
import numpy as np

from gnrl.bf import *
# ##############################################################################
# def evdinv():
# ##############################################################################
# def luinv():
# ##############################################################################
# def qrinv():
# ##############################################################################
# def roots():

###################################################################################
def dist1(x, w, met, xtp):
    n = x.shape[0]
    if xtp == 'num' or xtp == 'ord':
        ind = np.column_stack((np.arange(2,n+1), np.arange(1,n))) - 1
    else:
        ind = np.column_stack((np.arange(2,n+1), np.arange(1,n)))
        for i in range(1,n):
            ind = np.row_stack([ind, np.column_stack((np.arange(i+2,n+1), np.arange(1,n-i)))])
        ind = ind - 1

    if met == 'manh' or met == 'minkl':
        D = abs((x[ind[:,0], :] - x[ind[:, 1], :]))@w.T
    elif met == 'eucl' or met == 'mink2':
        D = (x[ind[:,0], :] - x[ind[:, 1], :])**2@w.T
    # elif met == 'minkinf':
    elif met[0:4] == 'mink':
        p = float(met[4:len(met)])
        D = (x[ind[:,0], :] - x[ind[:, 1], :])**p@w.T
    return D, ind
###################################################################################
def inv2(X, X_1, smin):
    n = X.shape[0]
    n1 = X_1.shape[0]
    #A = X[:n1, :n1]
    B = X[:n1, n1:n+1]
    C = X[n1:n+1, :n1]
    D = X[n1:n+1, n1:n+1]
    T1 = X_1@B
    Ai = X_1 + T1@nsinv(D - C@T1, smin)@T1.T
    rcD = 1/np.linalg.cond(D)
    if rcD < 1e-6: D_1 = nsinv(D)
    else:          D_1 = np.linalg.inv(D)
    D_1 = nsinv(D, smin)
    T2 = B@D_1
    Bi = -Ai@T2
    Di = D_1 - Bi.T@T2
    Xi = np.vstack((np.hstack((Ai, Bi)), np.hstack((Bi.T, Di)))) # Xi = [Ai Bi; Bi' Di];
    return Xi
###################################################################################
def lindp(X, tol=None):
    # determines linear dependent rows/columns of a symetric matrix X
    #--------------------------------------------
    # Author: Alexander Efremov                  
    # Date:   19 Jan 2010                        
    # Course: Multivariable System Identification
    #--------------------------------------------
    S = X.T@X
    n = S.shape[0]
    if tol is None: tol = n*eps()*max(np.sum(np.abs(S), axis=0))
    i = 0
    ind = np.arange(n)
    for j in range(n):
        p, ip = max1(abs(S[i:n, j]))
        ip += i# - 1
        if p <= tol: 
            S[i:n, j] = 0
        else:
            S[[i, ip], j:n] = S[[ip, i], j:n]
            ind[[i, ip]] = ind[[ip, i]]
            S[i, j:n] = S[i, j:n]/S[i, j]
            ii = np.arange(i + 1, n)
            S[ii, j:n] = S[ii, j:n] - c_(S[ii, j])@r_(S[i, j:n])
            i += 1
    inld, = np.where(np.sum(abs(S), axis=1) > tol)
    nld = ind[range(inld[-1]+1)]
    ld = ind[range(inld[-1]+1, n)]
    Xnld = X[:, nld]
    return Xnld, ld, nld
##############################################################################
def nsinv(X, s_min=1e-6, met='EVD', mtype='PDMR'):
    if met=='SVD':
        U, S, V = np.linalg.svd(X, full_matrices=True)
        # if np.max(S) == 0:
        #     print('Matrix is not invertible.') # todo: make it error
        #     return None # return Xinv = None
        if mtype in ['SPDM', 'SNDM', 'PDM', 'NDM', 'ANY']: 	ind = np.asarray(np.nonzero(S/(np.max(S) + 1e-6) > s_min)).flatten()
        else:            									ind = np.asarray( np.nonzero(S > s_min) ).flatten()
        invS1 = np.diag(S[ind]**-1)
        U1 = U[:, ind]
        V1 = V[:, ind]
        Xinv = V1@invS1@U1.T
    elif met=='EVD':
        d, V = np.linalg.eig(X)
        # if np.max(d) == 0:
        #     print('Matrix is not invertible.') # todo: make it error
        #     return None # return Xinv = None
        if mtype=='PDMR':   d1 = np.real(d) 	# eigenvalues should be real & positive, like us all :)
        elif mtype=='NDMR': d1 = -np.real(d)	# eigenvalues should be real & negative
        elif mtype=='NDM': 	d1 = -d			# eigenvalues should be negative
        ind = np.asarray( np.nonzero(d1/(np.max(d1) + 1e-6) > s_min)).flatten() # ind = find(abs(d)/(max(abs(d)) + 1e-6) > s_min);
        D1inv = np.diag(d[ind]**-1)
        V1 = V[:, ind]
        Xinv = V1@D1inv@V1.T
        if mtype[-1] == 'R': Xinv = np.real(Xinv)
        return Xinv
    elif met=='QR1' or met=='QR2':
        print('QR1 and QR2 invertions are not implemented yet.')
        return None # return Xinv = None
    else: 
        print('Unkmown method for matrix inversion...')
        return None # return Xinv = None
##############################################################################
def rand(size=1, l=0, h=1, tp='float', seed=None):
    if isinstance(size, (int, float)): size = [size, size]
    np.random.seed(seed=seed)
    if tp=='float': r = np.random.uniform(low=l, high=h, size=size)
    if tp=='int':   r = np.random.randint(low=l, high=h+1, size=size)
    return r
##############################################################################
def randn(size=1, m=0, s=1, seed=None):
    if isinstance(size, (int, float)): size = [size, size]
    np.random.seed(seed=seed)
    r = np.random.normal(loc=m, scale=s, size=size)
    return r
##############################################################################    
def rng(x): return max(x) - min(x)
##############################################################################    
def sort(x, order='asscend'):
    sz = x.shape
    x = x.flatten()
    if order == 'descend': s = -1
    else:                  s = 1
    ix = np.argsort(s*x)
    xs = x[ix]
    if len(sz) > 1:
        xs = np.reshape(xs, sz)
        ix = np.reshape(ix, sz)        
    return xs, ix
##############################################################################

