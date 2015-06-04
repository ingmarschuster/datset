# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 08:58:38 2015

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import scipy as sp
import scipy.stats as stats

from numpy import exp, log, sqrt
from scipy.misc import logsumexp
from numpy.linalg import inv

import theano.tensor as T
import theano

import os

__all__=["stoch_vol"]

def tnorm(mu, stddev):
    return stats.truncnorm(-mu/stddev, np.inf, loc=mu, scale=stddev)

def t_logpdf(x, nu):
    const = (T.gammaln((nu + 1) / 2) -(T.gammaln(nu/2) + (T.log(nu)+T.log(np.pi)) * 0.5))
    return const - (nu+1)/2*T.log(1+x**2/nu)

def stoch_vol(transform = None, theano_symbol = False, neg=False, opt_grad = False):
    from datset.data.stoch_vol_opt import optimum
    import pickle
    
    
    def LogSumExp(x, axis=None):
        x_max = T.max(x, axis=axis, keepdims=True)
        return T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max
        
    y = np.genfromtxt(os.path.split(__file__)[0]+"/data/SP500_Returns.csv",
                       skip_header=1, delimiter=",", usecols=[4])[:3001][::-1]
                       
    param = T.vector("param")
    nu = param[0]
    s = param[1:]
    
    if transform is None:
        pass
    elif transform == "sp":
        nu = T.nnet.softplus(nu)
        s = T.nnet.softplus(s)
        
    lpost = -0.01*(nu+s[0])
    lpost = lpost + T.sum(t_logpdf((y[1:]-y[:-1])/s, nu))
    lpost.eval({param:np.zeros(3001)})
    lpost = lpost - 1500.5* T.log(0.01+0.5*T.sum((T.log(s[1:]) - T.log(s[:-1]))**2))
    
    if neg:
        lpost = -lpost
    if theano_symbol:
        return lpost
    
    rval_lpost = theano.function([param], lpost)        
    rval_lgrad = theano.function([param],  T.grad(lpost, param))
    rval_lpost_lgrad_definitive = theano.function([param], [lpost, T.grad(lpost, param)])
    if not opt_grad:
        lp_lg = rval_lpost_lgrad_definitive
    else:
        def lp_lg(x, grad=False):
            if grad:
                return rval_lpost_lgrad_definitive(x)
            else:
                return rval_lpost(x)
    #rval_lhess = theano.function([param],  T.hessian(lpost, param))
    rval_lpost.opt = optimum
    rval_lpost.lev = -798.93510930385889
    with open(os.path.split(__file__)[0]+"/data/stoch_vol_ground_truth.pickle", "rb") as fil:
        (rval_lpost.mean, rval_lpost.var, rval_lpost.var_var) = pickle.load(fil)
    return (rval_lpost, rval_lgrad, lp_lg)

def factor_analysis():
    prior_lambd_diag = tnorm(1,2)
    prior_lambd_tri = stats.norm(0,2)
    prior_xi = stats.norm()
    prior_sigm_sq = stats.invgamma(1,scale=1)
    def logpost(xi, lambd, sigm_sq, idx_dat = None):
        pr = (prior_lambd_diag.logpdf(lambd.flat[diag_idx]).sum() +
              prior_lambd_tri.logpdf(lambd.flat[tri_idx]).sum() +
              prior_sigm_sq.logpdf(sigm_sq).sum())
        llh_dist = stats.multivariate_normal(cov = np.diag(sigm_sq))
        
        if idx_dat is None:
            return (pr + prior_xi.logpdf(xi).sum(1)
                    + llh_dist.logpdf(data - xi.dot(lambd)))
        else:
            return (pr + prior_xi.logpdf(xi[idx_dat]).sum()
                    + llh_dist.logpdf(data[idx_dat] - xi[idx_dat].dot(lambd))).sum()