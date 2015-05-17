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
           "negate"]




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

def negate(f_gradf):
    def negated(x):
        rval = f_gradf(x, grad=True)
        return (-rval[0], -rval[1])
    return negated