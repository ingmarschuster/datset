# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:09:01 2015

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

__all__ = ["credit_plain", "credit_hlr"]

def credit_plain(sigma):
    from  datset.data.german_credit import data_z as data
    
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
                    - 1/(2*sigma**2) * (alpha**2 + beta.dot(beta)))
                    
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
        
    rval_lpost = lambda param: rval_lp_lgrad(param, grad = False, lpost = True)
    
    rval_lgrad = lambda param: rval_lp_lgrad(param, grad = True, lpost = False)
    
    rval_lpost.opt5 = rval_lpost.opt10 = np.array([ 1.17903129,  0.72327228, -0.41081111,  0.40429438, -0.12126461,
        0.35671227,  0.17615053,  0.14985379, -0.01319223, -0.17613749,
        0.10723348,  0.22140805, -0.12024603, -0.03042954,  0.13299037,
        0.27224582, -0.27502516,  0.2889329 , -0.29613563, -0.26507539,
       -0.12209968,  0.06004149,  0.08724909,  0.02553138,  0.02414447]) #lpost: ~= -467.7

    return (rval_lpost, rval_lgrad, rval_lp_lgrad)

def credit_hlr(lambd, softplus_transform = False):
    from  datset.data.german_credit import data_z_2way as data
    
    (llab, llab_s) = log_sign(data[:,-1:])
    (lpred, lpred_s) = log_sign(data[:,:-1])
    
    if softplus_transform:
        transf = lambda par0: log(1+exp(par0))
    else:
        transf = lambda par0: par0
    
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
        assert(len(param) == 302)      
        sigmasq = transf(param[0])
        assert(sigmasq > 0)
        
        alpha = param[1:2]
        beta = param[2:]
        alsq_p_betsq = alpha**2 + beta.dot(beta)
        le = log_exp_term(alpha, beta)
        le_p1 = logsumexp((le, np.zeros_like(le)),0)

        if lpost:
            #assert(np.all(logsumexp((np.zeros_like(le), le),0) == log(1 + exp(le))))
            lp = (- le_p1.sum(0)
                    - 1/(2*sigmasq) * alsq_p_betsq
                    - data.shape[0]/2*log(sigmasq)-lambd*sigmasq)
            if len(lp.shape) == 3:
                lp.shape = lp.shape[1:]
            #assert()
            if not grad:
                return lp
        if grad:
            #assert(not np.isnan(exp_le).any())
            #assert(not np.isinf(le).any())
        
            da = exp_sign(*logsumexp_sign(le - le_p1 + llab, axis = 0, sign = llab_s)) - 1/(sigmasq)*alpha
            dssq = alsq_p_betsq/(2*sigmasq**2) - lambd - data.shape[0]/(2*sigmasq)
            db = exp_sign(*logsumexp_sign(le - le_p1 + llab + lpred, axis = 0, sign = llab_s * lpred_s)) - 1/(sigmasq)*beta

            #da = np.sum(exp(le-le_p1) * data[:,-1:], 0) - 1/(sigma**2)*alpha
            #db = np.sum(exp(le-le_p1) * data[:,-1:]*data[:,:-1], 0) - 1/(sigma**2)*beta
    
            lgrad = np.hstack((dssq, da, db))
            assert(lgrad.size==302)
            if not lpost:
                return lgrad
        return (lp, lgrad)
        
    rval_lpost = lambda param: rval_lp_lgrad(param, grad = False, lpost = True)
    
    rval_lgrad = lambda param: rval_lp_lgrad(param, grad = True, lpost = False)
    
    rval_lpost.opt_0_01 = np.array([  1.10603710e-07,  -1.00498968e-03,  -2.08838669e-03,
        -4.36793740e-05,  -1.32672098e-03,   8.44761326e-04,
        -1.69426660e-03,  -7.40781813e-04,  -7.86501422e-04,
         2.41702840e-04,   6.29503833e-04,  -1.03329870e-03,
        -3.32851354e-04,  -1.10064901e-03,   4.89766183e-04,
        -8.95457462e-04,   2.75004222e-04,   8.12176681e-04,
        -3.13342095e-04,  -1.61415782e-03,   1.51315668e-03,
         5.41609789e-04,  -8.38312963e-04,   2.35193638e-04,
         1.45629626e-03,  -1.49412626e-03,   3.07492281e-04,
        -5.92730223e-04,  -2.87807465e-04,  -3.49179813e-04,
         1.87811350e-04,   1.86751471e-05,   2.24568838e-04,
        -1.47984762e-04,   2.24384503e-04,  -6.66247060e-04,
        -5.19378772e-04,   4.23460142e-04,  -2.52747683e-04,
        -2.35266867e-04,  -4.87311684e-04,   1.95654196e-04,
        -5.95809481e-04,   5.61573768e-04,   7.70634006e-05,
        -8.28568513e-05,   1.30868293e-04,   4.85607697e-04,
        -2.80195007e-04,  -1.44238227e-04,   6.50051699e-04,
        -8.98059964e-04,  -3.14369645e-04,  -4.49364391e-04,
         1.08769941e-04,  -3.32504581e-05,   4.06052611e-04,
         1.37036314e-03,   6.98708906e-05,  -1.13796402e-03,
         8.86440744e-04,   3.41442490e-04,  -7.78803531e-04,
        -5.77707800e-04,  -1.73904049e-04,   2.57591376e-04,
         3.50954216e-04,  -7.25658517e-04,   3.32545732e-04,
        -1.16277753e-03,   4.43670887e-04,   6.30398502e-04,
        -3.59821315e-05,  -8.54271351e-04,   5.11772204e-04,
         9.60347854e-05,   1.57995730e-04,   4.32009249e-04,
        -6.23974730e-04,  -1.27644955e-03,   4.99053295e-06,
         2.63282558e-04,   5.12797508e-04,   2.95963062e-04,
         2.20677572e-05,  -2.93732883e-04,   7.70287040e-04,
         3.38798199e-04,  -3.26213813e-04,   9.42745655e-04,
        -3.54294812e-04,  -6.33464841e-05,  -1.10268515e-03,
         8.89705652e-05,  -1.97948591e-04,   2.35279688e-04,
        -1.39598510e-04,  -3.02472558e-05,   1.00916480e-03,
         6.21443410e-04,  -9.96722037e-04,   1.85572841e-05,
         1.69922508e-04,  -1.23960147e-03,  -1.64589378e-03,
        -3.04421373e-04,   1.49934721e-04,   2.06979799e-04,
         1.57490516e-04,  -1.31658846e-03,  -1.59920373e-04,
         9.56638511e-04,  -5.70123190e-04,   3.57037797e-05,
        -4.63363609e-04,  -8.33105489e-04,  -5.79060834e-04,
        -3.03369111e-04,   9.90447358e-05,  -1.48196897e-06,
        -3.77306254e-04,   1.59475688e-04,  -7.70595871e-04,
        -6.49282468e-04,  -2.96568905e-04,   2.64402339e-04,
         8.13838221e-05,   8.83733698e-04,  -9.40413478e-04,
         5.94693462e-04,   4.22055914e-05,   1.32463181e-04,
        -7.94356965e-04,  -6.16375167e-04,  -4.70002894e-04,
        -7.03099991e-04,  -6.13119945e-04,   2.60717695e-05,
        -4.14311987e-04,   1.58170365e-04,   4.91614243e-04,
         2.24267232e-04,   1.57725540e-04,   1.21462611e-04,
         6.65510806e-04,  -1.45689303e-04,   1.02850012e-04,
         3.58216308e-04,  -1.52825745e-04,   1.03698337e-04,
        -4.74762460e-04,   1.91036440e-04,  -3.83125603e-04,
         5.20996508e-05,   2.46819564e-04,   3.90400818e-05,
        -1.25827112e-04,   4.40346930e-04,  -5.81302017e-04,
        -2.53596357e-04,   5.99939379e-04,  -6.64204923e-04,
         2.91713778e-04,   4.78678459e-04,   2.41885569e-05,
        -1.08367781e-04,   2.30365714e-04,  -1.62678329e-03,
        -2.65515106e-04,   1.83398045e-04,   2.88217002e-04,
        -5.60774600e-04,   2.75219618e-04,  -5.41768050e-04,
        -6.93218401e-06,   2.35481748e-04,  -8.48319598e-06,
         3.46224805e-04,  -2.17045885e-04,  -2.20191983e-04,
        -2.48674190e-04,   8.82383778e-04,   9.53385203e-05,
        -1.68044211e-06,  -6.91149595e-05,  -8.11208749e-04,
        -7.27219403e-04,  -1.01804085e-04,  -2.41651242e-04,
        -6.42025997e-04,  -1.23842416e-04,   5.78997190e-04,
         1.41548342e-04,  -2.92178130e-04,  -6.04660109e-05,
         6.85053480e-05,  -5.20064501e-04,   3.28793748e-04,
         1.63748410e-04,   2.64921828e-04,  -7.29132958e-04,
         7.72168245e-04,   1.93438569e-04,  -1.26465500e-04,
         3.16573325e-04,  -4.45536033e-04,   4.28051932e-04,
        -4.52456547e-04,   1.15817109e-03,  -6.53562404e-04,
         8.43574979e-04,  -5.20991476e-04,   3.71582519e-04,
         3.89014609e-04,  -3.06726788e-04,  -1.50848582e-04,
        -2.33440949e-04,  -2.18167836e-04,   2.36871027e-04,
         5.21267692e-04,  -3.37174127e-04,   4.67583803e-04,
        -2.92314540e-04,   2.68149557e-04,  -2.90837518e-04,
         2.36880069e-04,   6.68799904e-04,   1.60313533e-04,
         4.96127705e-04,  -2.16548230e-05,   3.95299507e-05,
         6.40970596e-04,  -6.31790651e-04,   1.36285009e-03,
        -3.77993860e-04,   1.29781018e-04,  -3.75533684e-04,
         6.17104766e-04,   3.49204302e-04,  -3.92913944e-04,
        -1.36362093e-04,  -9.03910609e-04,   1.63871436e-04,
         4.43133752e-05,   1.80947751e-04,   4.42083071e-04,
        -4.33113283e-04,   4.27834484e-04,  -6.07234780e-04,
        -7.86328721e-04,  -3.69450564e-04,   2.71098711e-04,
         1.43308343e-04,   1.71114198e-04,  -4.29702032e-04,
         4.81791202e-06,   8.13784736e-04,   4.14267608e-04,
         4.57478587e-04,   1.33293842e-04,   2.20344350e-04,
         2.65000493e-04,  -1.31595061e-04,  -2.28873569e-04,
         8.35396481e-04,  -4.08057982e-04,   8.61953924e-05,
         1.24155162e-04,  -1.69573358e-04,  -3.15460448e-04,
         3.84096292e-04,  -4.81712983e-04,   3.40433364e-04,
         1.38979202e-04,  -3.96488978e-04,  -6.35116217e-05,
        -4.35171681e-04,   6.23621294e-04,  -5.69421955e-05,
        -1.78828783e-04,   3.69869803e-04,  -3.74278311e-03,
        -5.75784442e-04,   4.70900358e-04,  -4.89798621e-04,
         8.55338918e-04,  -1.82233179e-04,   4.84297135e-04,
        -1.75193336e-04,   8.67196298e-04,  -3.77545188e-04,
        -6.72743877e-04,   2.77405670e-04,   1.87435731e-04,
        -7.84535684e-04,   8.18930919e-04,   3.83759863e-04,
         8.18409713e-04,  -7.47196315e-05,  -2.60650777e-04,
         1.13880587e-04,  -4.97529934e-04]) # ~ 6813.2176032 (yes, this is positive. remember this is only proportional to the true log posterior)
    if softplus_transform:
        rval_lpost.opt_0_01[0] = log(exp(rval_lpost.opt_0_01[0]) - 1)
    return (rval_lpost, rval_lgrad, rval_lp_lgrad)