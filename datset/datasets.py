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
from datset.tools import *

import distributions as dist

__all__ = ["banana", "mvgauss",
           "gauss_mixture_bimod",
           "gauss_mixture_trim",
           "t_mixture_trim",
           "gauss_grid",
           "negate", "t_mixture_trim_bad"]




def banana(ssban= 100, bban = 0.03, lev = 0, ):
    def b(x):
        x = np.ravel(x)
        return -(((0.5/ssban)*x[0]**2+0.5*(x[1]-bban*(x[0]**2-ssban))**2)) + lev

    def gradb(x):
        x = np.ravel(x)
        return -np.array([ (1./ssban)*x[0]-2.*bban*x[0]*(x[1]-bban*(x[0]**2-ssban)),
                          x[1]-bban*(x[0]**2-ssban) ])
    b.mean = np.array([0, 0])
    b.lev = lev
    return (b, gradb, convert_to_lpost_and_grad_form(b, gradb))

def mvgauss(dim = 250, df_cov = 250, mean = None, lev = 0):
    if mean is None:
        mean = np.zeros(250)

    np.random.seed(2)
    cov = dist.invwishart_rv(np.eye(dim), df_cov)
    np.random.seed(None)
    mvn = dist.mvnorm(mean, cov)
    def lpost(x):
        return mvn.logpdf(x) + lev
    def lgrad(x):
        return mvn.logpdf_grad(x) + lev
    def lp_and_lgr(x,grad=False):
         if not grad:
             return lpost(x)
         else:
             res = mvn.log_pdf_and_grad(x)
             return [r+lev for r in res]
    lpost.mean=mean
    lpost.var = np.diag(cov)
    lpost.lev = lev
    
    return (lpost, lgrad, lp_and_lgr)

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
    

def t_mixture_trim_bad(distance=2.5, lev = 0, cut_dim = 10):
    dim = 10
    assert(dim >= cut_dim)
    assert(dim > 0 and cut_dim>0)
    np.random.seed(2)
    iwd = dist.invwishart(np.eye(dim),dim)
    mean3 = np.zeros(cut_dim)+distance*2
    mean3[-1] = -1
    weights = log((0.23, 0.7, 0.37))
    weights = exp(weights - logsumexp(weights))
    dists =  (dist.mvt(np.zeros(cut_dim), iwd.rv()[-cut_dim:, -cut_dim:], dim),
              dist.mvt(np.zeros(cut_dim)+distance, iwd.rv()[-cut_dim:, -cut_dim:],dim),
              dist.mvt(mean3, iwd.rv()[-cut_dim:, -cut_dim:],dim))
    (lp, lg) = gmix(weights, dists, lev)
    #rvs = np.vstack([dists[i].rvs(np.int(weights[i]*5000)) for i in range(3)])
    lp.var = np.array([  3.2464031 ,   6.77864311,   9.38763421,   7.46896529,
        43.68620454,  11.944712  ,   3.23960617,  56.10139093,
         7.95252566,  46.91379548]) #rvs.var(0)
    np.random.seed(None)    
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

def negate(f_gradf):
    def negated(x):
        rval = f_gradf(x, grad=True)
        return (-rval[0], -rval[1])
    return negated