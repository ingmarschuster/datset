# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:13:23 2015

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import scipy as sp
import scipy.stats as stats

from numpy import exp, log, sqrt
from scipy.misc import logsumexp
from numpy.linalg import inv

import distributions as dist

__all__ = ["credit_plain", "banana", "mvgauss",
           "gauss_mixture_bimod",
           "gauss_mixture_trim",
           "t_mixture_trim",
           "gauss_grid"]

def log_sign(a):
    a = np.array(a)
    sign_indicator = ((a < 0 ) * -2 + 1)
    return (log(np.abs(a)), sign_indicator)

def exp_sign(a, sign_indicator):
    return exp(a) * sign_indicator    

def logsumexp_sign(a, axis = None, b = None, sign = None):
    if sign is None:
        abs_res = logsumexp(a, axis = axis, b = b)
        sign_res = np.ones_like(abs_res)
        return (abs_res, sign_res)
    else:
        if not (np.abs(sign) == 1).all():
            raise ValueError("sign arguments expected to contain only +1 or -1 elements")

        m = np.copy(a)
        m[sign == -1] = -np.infty # don't look at negative numbers
        m = logsumexp(m, axis = axis, b = b)
        m[np.isnan(m)] = -np.infty # replace NaNs resulting from summing infty

        s = np.copy(a)
        s[sign == +1] = -np.infty # don't look at positive numbers
        s = logsumexp(s, axis = axis, b = b)
        s[np.isnan(s)] = -np.infty # replace NaNs resulting from summing infty


        sign_res =  np.ones(np.broadcast(m, s).shape) #default to positive sign
        abs_res  = -np.infty * sign_res #default to log(0)

        idx = np.where(m > s)
        abs_res[idx] = log(1 - exp(s[idx] - m[idx])) + m[idx]

        idx = np.where(m < s)
        sign_res[idx] = -1
        abs_res[idx] = log(1 - exp(m[idx] - s[idx])) + s[idx]
        return (abs_res, sign_res)


def convert_to_lpost_and_grad_form(lp, lg):
    def rval(param, grad = True):
        if grad:
            return (lp(param), lg(param))
        else:
            return lp(param)
    return rval


def credit_plain(sigma):
    import datset.data.german_credit as d

    data = np.hstack((stats.zscore(d.data[:, :24]), np.atleast_2d(-2 * (d.data[:, -1] - 1.5)).T))
    
    (llab, llab_s) = log_sign(data[:,-1:])
    (lpred, lpred_s) = log_sign(data[:,:-1])
    
    def log_exp_term(alpha, beta):
        if len(beta.shape) <= 1:
            beta = beta[:,None]
        tmp = -(alpha+np.tensordot(data[:,:-1], beta, 1))

        if len(tmp.shape) == 3:
            rval = np.atleast_3d(data[:,-1:]) * tmp
        elif len(tmp.shape) == 2:
            rval = data[:,-1:] * tmp
        else:
            raise RuntimeError()
        #rval = -data[:,-1:]*(alpha+data[:,:-1].dot(beta))
        
        return rval
        
    def rval_lp_lgrad(param, grad = False, lpost = True):
        assert(len(param) == 25)
        alpha = param[:1]
        beta = param[1:]
        le = log_exp_term(alpha, beta)
        le_p1 = logsumexp((le, np.zeros_like(le)),0)

        if lpost:
            #assert(np.all(logsumexp((np.zeros_like(le), le),0) == log(1 + exp(le))))
            lp = (- le_p1.sum(0)
                    - 1/(2*sigma**2) * (alpha**2 + (beta*2).sum(0)))
            if len(lp.shape) == 3:
                lp.shape = lp.shape[1:]
            if not grad:
                return lp
        if grad:
            #assert(not np.isnan(exp_le).any())
            #assert(not np.isinf(le).any())
        
            da = exp_sign(*logsumexp_sign(le - le_p1 + llab, axis = 0, sign = llab_s)) - 1/(sigma**2)*alpha
            db = exp_sign(*logsumexp_sign(le - le_p1 + llab + lpred, axis = 0, sign = llab_s * lpred_s)) - 1/(sigma**2)*beta
            #da = np.sum(exp(le-le_p1) * data[:,-1:], 0) - 1/(sigma**2)*alpha
            #db = np.sum(exp(le-le_p1) * data[:,-1:]*data[:,:-1], 0) - 1/(sigma**2)*beta
    
            lgrad = np.hstack((da, db))
            assert(lgrad.size==25)
            if not lpost:
                return lgrad
        return (lp, lgrad)
        
    def rval_lpost(param):
        return rval_lp_lgrad(param, grad = False, lpost = True)
        
        assert(len(param)  == 25)
        alpha = param[:1]
        beta = param[1:]
        #assert()
        rval = (- np.sum(log(1 + exp(log_exp_term(alpha, beta))), 0)
                - 1/(2*sigma**2) * (alpha**2 + (beta*2).sum(0)))
        if len(rval.shape) == 3:
            rval.shape = rval.shape[1:]
        return rval
    
    def rval_lgrad(param):
        return rval_lp_lgrad(param, grad = True, lpost = False)
        
        assert(len(param) == 25)
        alpha = param[:1]
        beta = param[1:]
        le = log_exp_term(alpha, beta)
        exp_tmp = exp(le)
        le_p1 = logsumexp((le, np.zeros_like(le)),0)
        assert(not np.isnan(exp_tmp).any())
        da = exp_sign(*logsumexp_sign(le - le_p1 + llab, axis = 0, sign = llab_s)) - 1/(sigma**2)*alpha
        db = exp_sign(*logsumexp_sign(le - le_p1 + llab + lpred, axis = 0, sign = llab_s * lpred_s)) - 1/(sigma**2)*beta
        #da = np.sum(exp(le-le_p1) * data[:,-1:], 0) - 1/(sigma**2)*alpha
        #db = np.sum(exp(le-le_p1) * data[:,-1:]*data[:,:-1], 0) - 1/(sigma**2)*beta

        rval = np.hstack((da, db))
        assert(rval.size==25)
        return rval
    rval_lpost.opt5 = np.array([  7.62825007e+00,   2.01712061e-01,  -2.10225678e-01,
         1.53269288e-01,   2.69474439e-02,   1.41285640e-01,
         2.58587817e-02,   4.62547318e-02,  -2.84436347e-02,
        -8.94241445e-02,   7.40826247e-03,   4.03908500e-02,
        -6.33738178e-02,  -1.05608238e-02,   1.59252838e-02,
         4.54528407e-02,  -9.03591353e-02,   5.47324301e-02,
        -1.06833474e-01,  -1.10238240e-01,  -1.21175245e-01,
        -4.75305361e-02,  -2.35780676e-03,  -6.68107257e-02,
        -6.85008768e-02])
    rval_lpost.opt10 = np.array([  8.88668598e+00,   2.04117603e-01,  -2.21549592e-01,
         1.58496419e-01,   3.52187959e-02,   1.32091725e-01,
         3.94060496e-02,   4.87432410e-02,  -2.49557521e-02,
        -9.69452968e-02,   5.06734149e-03,   4.30752351e-02,
        -5.83791208e-02,  -1.03549489e-02,   1.76974267e-02,
         4.32276774e-02,  -9.17635023e-02,   5.06685218e-02,
        -1.12617320e-01,  -1.22767638e-01,  -1.41373948e-01,
        -5.32635384e-02,  -1.38282395e-04,  -7.54911103e-02,
        -7.66090620e-02]) #lpost: -0.48210984
    rval_lpost.opt10_int = np.array([ 22.36481313,   0.16458135,  -1.00014313,   0.35911893,
         2.57071533,  -0.12540802,  -0.03206204,   0.40642445,
        -0.87807069,  -0.58052996,   0.57273506,  -1.05636089,
        -0.06166775,   1.57400721,  -0.16801705,   1.64253854,
         0.76699827,   0.39459233,  -1.23495934,   0.86640948,
        -0.45675256,  -1.77389188,   2.01329111,  -0.31598541,   0.57181302])
    return (rval_lpost, rval_lgrad, rval_lp_lgrad)

def banana(ssban= 100, bban = 0.03, lev = 0):
    def b(x):
        x = np.ravel(x)
        return -(((0.5/ssban)*x[0]**2+0.5*(x[1]-bban*(x[0]**2-ssban))**2)) + lev

    def gradb(x):
        x = np.ravel(x)
        return -np.array([ (1./ssban)*x[0]-2.*bban*x[0]*(x[1]-bban*(x[0]**2-ssban)),
                          x[1]-bban*(x[0]**2-ssban) ]) + lev
    b.mean = np.array([0, 0])
    b.lev = lev
    return (b, gradb, convert_to_lpost_and_grad_form(b, gradb))

def mvgauss(seed = 2, dim = 250, df_cov = 250):
    st = np.random.RandomState
    np.random.seed(seed)
    prec = np.eye(dim) #dist.wishart_rv(np.eye(dim), df_cov)
    np.random.seed(None)
    
    def rval_lpost(param):
        return -0.5*param.T.dot(prec).dot(param)
    
    def rval_lgrad(param):
        return -np.tensordot(prec, param, 1)
    rval_lpost.mean = np.zeros(dim)
    return (rval_lpost, rval_lgrad, convert_to_lpost_and_grad_form(rval_lpost, rval_lgrad))

def gmix(weights, dists, lev):
    weights = log(weights)
    weights = weights - logsumexp(weights)
    
    lpost = lambda param: logsumexp(weights + [d.logpdf(param) for d in dists]) + lev
    lgrad = lambda param: logsumexp([exp(weights[i]) * dists[i].logpdf_grad(param) for i in range(len(dists))], 0)
    
    lpost.lev = lev
    lpost.mean = np.sum([exp(weights[i]) * dists[i].mu for i in range(len(dists))],0)
    
    return (lpost, lgrad)

def gauss_mixture_bimod(distance = 2, dim = 2, lev = 0):
    d1 = dist.mvnorm(np.zeros(dim), np.eye(dim))#dist.wishart_rv(np.eye(dim)*3, 250))
    d2 = dist.mvnorm(np.zeros(dim)+distance, np.eye(dim))#dist.wishart_rv(np.eye(dim)*3, 250))
    
    (lp, lg) = gmix((3, 7), (d1, d2), lev)
    
    return (lp, lg, convert_to_lpost_and_grad_form(lp, lg))

def gauss_mixture_trim(distance = 1, dim = 2, lev = 0):
    (lp, lg) = gmix((3, 5, 2), (dist.mvnorm(np.zeros(dim), np.eye(dim)),
                             dist.mvnorm(np.zeros(dim)+distance, np.eye(dim)),
                             dist.mvnorm(np.zeros(dim)+distance*2, np.eye(dim))), lev)
    
    return (lp, lg, convert_to_lpost_and_grad_form(lp, lg))

def t_mixture_trim(distance = 1, dim = 2, df = 2, lev = 0):
    (lp, lg) = gmix((3, 5, 2), (dist.mvt(np.zeros(dim), np.eye(dim), df),
                             dist.mvt(np.zeros(dim)+distance, np.eye(dim),df),
                             dist.mvt(np.zeros(dim)+distance*2, np.eye(dim),df)), lev)
    
    return (lp, lg, convert_to_lpost_and_grad_form(lp, lg))
def gauss_grid(reweight = False, lev = 0):
    r = range(-3,10,3)    
    means = np.array([(x,y) for x in r for y in r])
    
    if not reweight:
        weights = np.ones(len(means))
    else:
        weights = 1./(np.abs(means*0.125).sum(1)+1)
        #print(weights)

    (lp, lg) = gmix(weights, [dist.mvnorm(m, np.eye(2)) for m in means], lev)
    return (lp, lg, convert_to_lpost_and_grad_form(lp, lg))
    

def mvt(seed = 2, dim = 250, df_cov = 250, df_t = 200):
    import distributions as dist
    np.random.seed(seed)
    prec = dist.wishart_rv(np.eye(dim), df_cov)
    t = dist.mvt(np.zeros(dim), np.linalg.inv(prec), df_t, Ki = prec)
    t.Ki = prec
    np.random.seed(None)
    
    
    def rval_lpost(param):
        return t.logpdf(param)
    
    def rval_lgrad(param):
        return t.logpdf_grad(param)
    
    return (rval_lpost, rval_lgrad)
 