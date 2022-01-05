"""
author: OPEN-MAT
date: 	15.06.2019
Matlab version: 
    author: Alexander Efremov
    date:   26 Apr 2009
    course: Multivariable Control Systems
"""
import numpy as np
from md_reg import * # needed for vif()
from dp_feng.normlz import *
from gnrl.bf import *


# def assoc(model = None,st = None,y = None,ym = None):
#     #--------------------------
# # Author: Alexander Efremov
# # Date:   20.06.2011
# #--------------------------
# # Measures of association /non-parametric measures/
#     
#     # matrixProc.processRecord(mrec.inputs, normInputs
#     
#     model = model(end())
#     rankBins = 500
#     N = st.ovr.N
#     dym = (np.amax(ym) - np.amin(ym)) / rankBins
#     
#     ind = int(np.floor((ym - np.amin(ym)) / dym)) + 1
#     ind[find[ind < 1]] = 1
#     ind[find[ind > rankBins]] = rankBins
#     for i in np.arange(1,rankBins+1).reshape(-1):
#         ind2 = find(ind == i)
#         cells[i,np.arange[1,2+1]] = sum(np.array([(1 - y(ind2)),y(ind2)]))
#     
#     jbins = 2
#     nn = rankBins * jbins
#     ties = 0
#     en1 = 0
#     en2 = 0
#     Nc = 0
#     
#     Nd = 0
#     
#     for i in np.arange(1,nn - 1+1).reshape(-1):
#         ki = int(np.floor((i + 1) / jbins))
#         kj = i - jbins * ki + 2
#         for j in np.arange(i + 1,nn+1).reshape(-1):
#             li = int(np.floor((j + 1) / jbins))
#             lj = j - jbins * li + 2
#             m1 = li - ki
#             m2 = lj - kj
#             mm = m1 * m2
#             pairs = cells(ki,kj) * cells(li,lj)
#             if mm != 0:
#                 if mm > 0:
#                     Nc = Nc + pairs
#                 else:
#                     Nd = Nd + pairs
#             else:
#                 if m1 != 0:
#                     en1 = en1 + pairs
#                 if m2 != 0:
#                     en2 = en2 + pairs
#                     ties = ties + pairs
#     
#     Nt = Nc + Nd + ties
#     
#     c = (Nc + 0.5 * ties) / Nt
#     
#     SomersD = (Nc - Nd) / Nt
#     gamma = (Nc - Nd) / (Nc + Nd)
#     tau_a = (Nc - Nd) / (0.5 * N * (N - 1))
#     
#     Pc = Nc / Nt * 100
#     Pd = Nd / Nt * 100
#     Pt = ties / Nt * 100
#     st.ovr.assoc.Pc = Pc
#     st.ovr.assoc.Pd = Pd
#     st.ovr.assoc.Pt = Pt
#     st.ovr.assoc.Nt = Nt
#     st.ovr.assoc.tau_a = tau_a
#     st.ovr.assoc.SomersD = SomersD
#     st.ovr.assoc.gamma = gamma
#     st.ovr.assoc.tau_a = tau_a
#     st.ovr.assoc.c = c
#     return st


###################################################################################
def auc(g, b):
    if g.ndim == 1: g = g.reshape(len(g), 1); b = b.reshape(len(b), 1)
    [m, n] = g.shape
    G = g.sum(axis=0)
    B = b.sum(axis=0)
    pg = g/G*100
    pb = b/B*100
    cpg = np.cumsum(pg, axis=0)
    tmp = pb*(cpg + np.vstack((np.zeros((1, n)), cpg[:-1, :])))
    gini = abs(100 - 0.005*tmp.sum(axis=0))
    return gini
###################################################################################
def betacf(a, b, x, Eps=3e-7, minFP=1e-30, maxIter=int(1e2)):
    qap, qab, qam, c = a + 1, a + b, a - 1, 1
    d = 1 - qab*x/qap
    d[abs(d) < minFP] = minFP
    d = 1/d
    bcf = d
    res = nans(bcf.shape)
    r = np.arange(bcf.shape[0])
    r1 = np.arange(bcf.shape[0])
    itr = 0
    iterate = 1
    while iterate:
        itr += 1
        if itr > maxIter: break
        m2 = 2*itr
        aa = itr*(b - itr)*x/((qam + m2)*(a + m2))
        d = 1 + aa*d
        d[abs(d) < minFP] = minFP
        c = 1 + aa/c
        c[abs(c) < minFP] = minFP
        d = 1/d
        bcf = bcf*d*c
        aa = -(a + itr)*(qab + itr)*x/((a + m2)*(qap + m2))
        d = 1 + aa*d
        d[abs(d) < minFP] = minFP
        c = 1 + aa/c
        c[abs(c) < minFP] = minFP
        d = 1/d
        bcf = bcf*d*c
        ii = (abs(d*c - 1) <= Eps).flatten()
        if ii.any():
            res[r[r1[ii]]] = bcf[r1[ii]]
            r1 = r1[np.invert(ii)]
            r = r[r1]
            a = a[r1]
            b = b[r1]
            qab = qab[r1]
            qam = qam[r1]
            qap = qap[r1]
            aa = aa[r1]
            d = d[r1]
            c = c[r1]
            bcf = bcf[r1]
            x = x[r1]
            r1 = np.arange(bcf.shape[0])
            if not r1.shape[0]: iterate = 0
    return res
###################################################################################
def betai(a, b, x):
    a = c_(a)
    b = c_(b)
    x = c_(x)
    bi = nans(x.shape)
    if ((x < 0) | (x > (1 + 1e-14))).any():
        print('ERROR: SpecialFunctions, betai():', 'x not in [0, 1].')
        return bi
    ii = x == 0
    if ii.any(): bi[ii] = 0
    ii = x >= 1
    if ii.any(): bi[ii] = 1
    ii = (x > 0) & (x < 1)
    if ii.any(): bi[ii] = np.exp(gamln(a[ii] + b[ii]) - gamln(a[ii]) - gamln(b[ii]) + a[ii]*np.log(x[ii]) + b[ii]*np.log(1 - x[ii])).flatten()
    ii = (x > 0) & (x < (a + 1)/(a + b + 2))
    if ii.any():
        bi[ii] = bi[ii]*betacf(a[ii], b[ii], x[ii])/a[ii]
    ii = (x >= (a + 1)/(a + b + 2)) & (x < 1)
    if ii.any():
        bi[ii] = 1 - bi[ii]*betacf(b[ii], a[ii], 1 - x[ii])/b[ii]
    return bi
##############################################################################
def chi2(g, b):
    if g.ndim == 1: g = g.reshape(len(g), 1); b = b.reshape(len(b), 1)
    G = g.sum(axis=0)
    B = b.sum(axis=0)
    egi = (g + b)*G/(G + B)
    ebi = (g + b)*B/(G + B)
    egi[egi < 1e-12] = 1e-12
    ebi[ebi < 1e-12] = 1e-12
    chi = ((g - egi)**2/egi + (b - ebi)**2/ebi).sum(axis=0).T
    return chi
###################################################################################
def confm(x1, x2, w=None):
    # confusion matrix
    ux1 = np.unique(x1)
    ux2 = np.unique(x2)
    cm = nans((len(ux1), len(ux2)))
    for i in range(len(ux1)):
        for j in range(len(ux2)):
            cm[i, j] = np.sum(w[(x1 == ux1[i]) & (x2 == ux2[j])])
    return cm
###################################################################################
def corf(x, y=None):
    if y is None: y = x
    N = x.shape[0]
    X = hnkl(np.vstack((x, x)), N)
    R = X @ y / (N - 1)
    return R
###################################################################################
def covf(x, y=None):
    x, __ = stdn(x)
    if y is None: y = x
    C = corf(x, y)
    return C
##############################################################################
def dind(g, b):
    G = sum(g)
    B = sum(b)
    gb = g + b
    gb[gb < 1e-12] = 1e-12
    dind = 100 - sum((g * b / gb * (G + B) / (G * B))) * 100
    return dind
##############################################################################
def frat(y, y2, nw):
    if y.ndim == 1 and len(y) < 1: f = np.nan; return f
    if y.ndim == 1:
        m, n = len(y), 1
    else:
        m, n = y.shape
    eps = 1e-6  # np.finfo(float).eps
    nw[nw < eps] = eps
    ssy = sum(y2)
    rc = sum(y.astype('float')**2/nw)
    ys = sum(y)
    Nw = sum(nw)
    s1 = rc - float(ys)**2/Nw
    s2 = ssy - rc
    f = s1/s2*(Nw - m)/(m - 1)
    if type(f) != np.array: f = np.array([f])
    return f
###################################################################################
def gamln(a):
    cof = c_(np.array(
        [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2,
         0.5395239384953e-5]))
    y = a + c_(np.arange(1, 7))
    tmp = a + 5.5
    tmp = tmp - (a + 0.5) * np.log(tmp)
    ser = 1.000000000190015 + sum(cof / y)
    return -tmp + np.log(2.5066282746310005 * ser / a)
###################################################################################
def gamq(a, x):
    # a must be real and non-negative
    if type(a) != np.array: a = np.array([[a]])
    if type(x) != np.array: x = np.array([[x]])

    # if x.min() < 0 or a.min() <= 0:
    #     print('ERROR: SpecialFunctions, gamq():', 'Invalid arguments in routine gamq')
    #     return -1
    if a.shape == (1, 1): a = np.tile(a, x.shape)
    a = a.flatten()
    x = x.flatten()
    gq = nans((x.shape[0], 1))
    gq[a == 0] = 0
    gq[(a > 0) & (x == 0)] = 1
    for i in range(0, len(gq.flatten())):
        if x[i] <= 0 or a[i] <= 1e-300: continue
        if x[i] < a[i] + 1:
            gq[i] = 1 - gser(a[i], x[i])
        else:
            gq[i] = gcf(a[i], x[i])
    return gq
###################################################################################
def gbi(g, b):
    if g.ndim == 1: g = c_(g); b = c_(b)
    [m, n] = g.shape
    G = np.tile(g.sum(axis=0), (m, 1))
    B = np.tile(b.sum(axis=0), (m, 1))
    b[b < 1e-12] = 1e-12
    gbo = g / b;
    GBO = G / B
    gbo = gbo.flatten()
    GBO = GBO.flatten()
    gbi = np.zeros(len(gbo))
    ind = gbo >= GBO;
    gbi[ind] = gbo[ind] / GBO[ind] * 100
    gbo[gbo < 1e-12] = 1e-12
    ind = gbo < GBO;
    gbi[ind] = -GBO[ind] / gbo[ind] * 100
    gbi = gbi.reshape(m, n)
    gbi1 = gbi + 200 * (gbi < 0).astype(int)
    gbi1 = gbi1.reshape(m, n)
    return gbi
###################################################################################
def gcf(a, x, delta=1e-7, maxIter=int(1e3)):
    gln, fac, b1, b0, a0 = gamln(a), 1, 1, 0, 1
    gcf = 0
    gold = [0];
    if x <= 0:
        print('ERROR: SpecialFunctions, gcf():', 'Wrong gamma-function parameter')
        return -1
    a1 = x
    N = range(1, maxIter + 1)
    for n in N:
        an = n;
        ana = an - a
        a0 = (a1 + a0 * ana) * fac;
        b0 = (b1 + b0 * ana) * fac
        anf = an * fac
        a1 = x * a0 + anf * a1;
        b1 = x * b0 + anf * b1
        if a1 != 0:
            fac = 1 / a1
            g = b1 * fac
            if abs((g - gold) / g) < delta:
                temp = -x + a * np.log(x) - gln
                if abs(temp) > 700:
                    if temp < -700:
                        temp = -700
                    else:
                        temp = 700
                gcf = np.exp(temp) * g
                break
            gold = g
    return gcf
###################################################################################
def gini(g, b):
    if g.ndim == 1: g = g.reshape(len(g), 1); b = b.reshape(len(b), 1)
    [m, n] = g.shape
    G = g.sum(axis=0)
    B = b.sum(axis=0)
    pg = g / G * 100
    pb = b / B * 100
    cpg = np.cumsum(pg, axis=0)
    tmp = pb * (cpg + np.vstack((np.zeros((1, n)), cpg[:-1, :])))
    gini = abs(100 - 0.01 * tmp.sum(axis=0))
    return gini
###################################################################################
def gser(a, x):
    maxIter = 1000
    Eps = 1e-12
    if x < 1e-300: return 0
    if abs(a) < 1e-300:
        print('ERROR: SpecialFunctions, gser():', 'Second parameter in the gamma function gser too close to 0')
        return -1
    gln = gamln(a)
    gamser = 0
    ap = a;
    inva = 1 / a;
    s = inva
    for n in range(0, maxIter):
        ap = ap + 1;
        inva = inva * x / ap;
        s = s + inva
        if abs(inva) < abs(s) * Eps:
            gamser = s * np.exp(-x + a * np.log(x) - gln)
            break
    return gamser
###################################################################################
def ks(g, b):
    if g.ndim == 1: g = g.reshape(len(g), 1); b = b.reshape(len(b), 1)
    [m, n] = g.shape
    G = g.sum(axis=0)
    B = b.sum(axis=0)
    pg = g / G * 100
    pb = b / B * 100
    cpg = np.cumsum(pg, axis=0)
    cpb = np.cumsum(pb, axis=0)
    ks = np.max(abs(cpg - cpb), axis=0)
    return ks


###################################################################################
def pvalC(v, C2):
    return gamq(0.5 * v, 0.5 * C2)
###################################################################################
def pvalF(F, v1, v2, tp='ot'):
    # Computes the p-value for Fisher
    # Inputs:
    #   F - F value
    #   v1 - regression degeres of freedom
    #   v2 - error degeres of freedom
    #   tp - type of test /default is 'ot'/
    #      'tt' - two tail
    #      'ot' - one tail
    # Output
    #   p - p-value of F value
    F = c_(F)
    if v2 <= 0 or v2 <= 0: print('ERROR: SpecialFunctions, pvalF():',
                                 'Degrees of freedom must be positive...'); return []
    if (F < 0).any():      print('ERROR: SpecialFunctions, pvalF():', 'x must be non negative...'); return []
    if tp == 'ot':
        p = betai(v2 / 2, v1 / 2, v2 / (v2 + F * v1))
    else:
        p = 2 * betai(v2 / 2, v1 / 2, v2 / (v2 + F * v1))
    p[p < 0] = 0  # p may be out of range due to round-off errors
    p[p > 1] = 1  #
    return p
###################################################################################
def vaf(Y, Ym, w=np.empty([0, 0]), p=0):
    Nw, r = Y.shape
    if w.size > 0:
        if w.size < Y.size: w = np.tile(w, (1, r))
        Nw = np.sum(w, axis=0).T
        mY = sum(Y*w)/Nw
        SSE = sum((Y - Ym)**2*w)
        SST = sum((Y - mY)**2*w)
    else:
        mY = np.mean(Y, axis=0)
        SSE = sum((Y - Ym)**2)
        SST = sum((Y - mY)**2)
    VAF = 1 - (SSE/ SST*(Nw - 1)/(Nw - p - 1))
    VAF[VAF < 0] = 0
    return VAF * 100
###################################################################################
def vif(x):
    m = x.shape[1]
    vifi = nans((m, 1))
    for i in range(m):
        yi = c_(x[:, i])
        xi = np.delete(x, i, axis=1)
        mdl = lspm(xi, yi)
        yim = lspm_apl(xi, yi, mdl=mdl)
        R2 = vaf(yi, yim) / 100
        if R2 == 1:
            vifi[i] = np.inf
        else:
            vifi[i] = 1 / (1 - R2)
    return vifi
###################################################################################
def vinf(g, b, tau=1e-12):
    G = sum(g)
    B = sum(b)
    gG = g / G
    bB = b / B
    gG[gG < tau] = tau
    bB[bB < tau] = tau
    vinf = sum((gG - bB) * np.log(gG / bB))
    return vinf
###################################################################################
def woe(g, b, tau=1e-12):
    G = sum(g)
    B = sum(b)
    gG = g / G
    bB = b / B
    gG[gG < tau] = tau
    bB[bB < tau] = tau
    woe = np.log(gG / bB)
    return woe
##############################################################################
