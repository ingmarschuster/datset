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

import distributions as dist

from datset.tools import *

__all__ = ["credit_plain", "credit_plain_theano", "credit_hlr", "credit_hlr_theano", "credit_simple_is"]

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


def credit_plain_theano(sigma):
    import theano.tensor as T
    import theano
    
    def LogSumExp(x, axis=None):
        x_max = T.max(x, axis=axis, keepdims=True)
        return T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    
    from  datset.data.german_credit import data_z as data

    
    param = T.vector("param")
    alpha = param[0]
    beta = param[1:]
    ssq = sigma**2 # T.scalar("ssq")
    
    le = (-data[:,-1]*(alpha+T.tensordot(data[:,:-1], beta,1)))
    lpost = (- LogSumExp((le, T.zeros_like(le)),0).sum()
                    - 1/(2*ssq) * (alpha**2 + beta.dot(beta)))
    
    rval_lpost = theano.function([param], lpost[np.newaxis])
    rval_lgrad = theano.function([param],  T.grad(lpost, param))
    rval_lpost_lgrad_definitive = theano.function([param], [lpost, T.grad(lpost, param)])
    def rval_lpost_lgrad(x, grad=False):
        if grad:
            return rval_lpost_lgrad_definitive(x)
        else:
            return rval_lpost(x)
    rval_lhess = theano.function([param],  T.hessian(lpost, param))
    #rval_lpost_lgrad_lhess = theano.function([ab], [lpost, T.grad(lpost, ab), T.hessian(lpost, ab)])
    rval_lpost.start = np.array([ -7.06854908e-03,  -1.05940740e-03,  -2.22595260e-03,
        -1.76019755e-04,  -1.50490322e-02,   8.78030345e-03,
        -7.37716090e-03,  -7.12096046e-04,   2.63549291e-03,
        -2.72145365e-03,  -1.24480951e-03,   1.53656546e-03,
        -2.10982582e-03,   9.54294662e-03,   3.28415735e-03,
        -4.27019717e-03,  -5.40388605e-03,   1.59955152e-05,
        -8.39165505e-04,  -8.70450458e-03,   4.59230665e-03,
        -3.88058536e-03,   7.14414576e-04,   1.57565107e-03,
        -3.55516986e-04]) # -693.12774797
    rval_lpost.opt5 = rval_lpost.opt10 = np.array([ 1.17903129,  0.72327228, -0.41081111,  0.40429438, -0.12126461,
        0.35671227,  0.17615053,  0.14985379, -0.01319223, -0.17613749,
        0.10723348,  0.22140805, -0.12024603, -0.03042954,  0.13299037,
        0.27224582, -0.27502516,  0.2889329 , -0.29613563, -0.26507539,
       -0.12209968,  0.06004149,  0.08724909,  0.02553138,  0.02414447])
    rval_lpost.lev_10 = -504.5011569532183
    rval_lpost.mean10 = np.array([ 1.21912752,  0.74494375, -0.42524601,  0.41907027, -0.12642431,
        0.36976911,  0.18167385,  0.15450298, -0.0126782 , -0.18248051,
        0.11177484,  0.22735348, -0.12517907, -0.02876698,  0.13781747,
        0.29856619, -0.28219943,  0.30364463, -0.31440319, -0.28047437,
       -0.12575181,  0.06176199,  0.09505578,  0.02372538,  0.02272984]) # ess:38k
    rval_lpost.var10 = np.array([ 0.0086826 ,  0.00822621,  0.01115241,  0.0092435 ,  0.01212074,
        0.00906877,  0.00869139,  0.00691414,  0.00834668,  0.01123536,
        0.00966755,  0.00626672,  0.00908454,  0.00744986,  0.00917602,
        0.01472509,  0.00699316,  0.01105414,  0.01507119,  0.01262483,
        0.0195209 ,  0.0212143 ,  0.00831238,  0.0168367 ,  0.01607689]) # ess:38k
    return (rval_lpost, rval_lgrad, rval_lpost_lgrad,rval_lhess)#, rval_lpost_lgrad_lhess)


def credit_hlr_theano(lambd, transform = None, theano_symbol = False, neg=False):
    import theano.tensor as T
    import theano
    
    def LogSumExp(x, axis=None):
        x_max = T.max(x, axis=axis, keepdims=True)
        return T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    
    from  datset.data.german_credit import data_z_2way as data

    
    param = T.vector("param")
    ssq = param[0]
    if transform is None:
        pass
    elif transform == "sp":
        ssq = T.nnet.softplus(ssq)
        
    alpha = param[1]
    beta = param[2:]
    
    le = (-data[:,-1]*(alpha+T.tensordot(data[:,:-1], beta,1)))
    lpost = (- LogSumExp((le, T.zeros_like(le)),0).sum()
                    - 1/(2*ssq) * (alpha**2 + beta.dot(beta)) - data.shape[0]/2*T.log(ssq)-lambd*ssq)
    if neg:
        lpost = -lpost
    if theano_symbol:
        return lpost
    else:             
        rval_lpost = theano.function([param], lpost[np.newaxis])
        rval_lgrad = theano.function([param],  T.grad(lpost, param))
        rval_lpost_lgrad = theano.function([param], [lpost, T.grad(lpost, param)])
        rval_lhess = theano.function([param],  T.hessian(lpost, param))
        
        
        rval_lpost.opt_0_01 = np.array([  1.48330894e-03,   2.32123189e-01,   2.10581621e-01, #without transforms
        -1.37343452e-01,   1.10535811e-01,  -5.34547911e-02,
         9.76128206e-02,   7.31552147e-02,   3.39595702e-02,
         1.09578012e-02,  -5.64125001e-02,   2.32323899e-02,
         6.61636971e-02,  -2.27567000e-02,   7.64824332e-03,
         1.64770910e-02,   2.56178157e-02,  -7.48164442e-02,
         7.82501141e-02,  -3.28552764e-02,  -1.74748743e-02,
        -5.37079683e-02,   6.60081787e-02,  -1.15141308e-02,
        -8.03231759e-03,   9.45767737e-06,   5.70771504e-02,
        -1.67812530e-02,  -4.63400058e-02,  -5.93271569e-03,
         4.84038000e-02,  -1.04979823e-03,   6.78325490e-03,
         3.85596778e-02,   1.31623568e-02,   5.17508343e-02,
        -2.03388354e-02,  -1.97031214e-02,   6.88271876e-04,
        -1.88241566e-02,   3.14845811e-03,   1.09737749e-02,
         6.33233867e-02,   2.44021354e-03,  -2.38332689e-02,
         1.04791926e-02,   1.52008087e-02,  -1.97676826e-02,
         5.60896981e-02,  -4.47555813e-02,   8.93739911e-02,
         6.88181332e-02,   2.50701685e-02,  -1.49262635e-05,
        -2.30763314e-02,   4.15521456e-02,   9.47263938e-03,
         1.83105780e-02,  -1.81995132e-02,  -6.73780916e-04,
         6.14959633e-02,  -3.54029263e-02,  -8.99008201e-03,
         5.93523924e-02,   3.07826246e-02,   1.47499306e-02,
        -5.04738819e-02,  -9.85327035e-03,   1.22732344e-02,
        -7.20580520e-02,  -5.02610361e-03,   1.80339148e-02,
        -2.43079224e-02,   1.42379856e-02,   2.00284754e-03,
         2.50452469e-02,   2.48632110e-02,   5.59120116e-02,
         5.69747396e-02,   1.29797762e-01,   1.60026879e-02,
         1.33449534e-02,  -6.88846417e-03,   1.85348841e-02,
         5.97326029e-03,   4.36514427e-02,   2.63778710e-02,
         1.88591045e-03,  -2.00610537e-02,   3.05143298e-02,
        -9.41296531e-03,   7.73652683e-05,   3.90191492e-02,
        -2.82429583e-02,   5.16560747e-03,   2.38133607e-02,
         3.26736525e-02,  -1.94812850e-02,  -4.97476918e-03,
        -3.28829506e-03,   2.13963887e-02,  -6.14751449e-02,
        -5.48245043e-02,   2.41075493e-03,  -1.37561933e-02,
         1.02916355e-02,  -1.74604813e-03,  -3.56202427e-03,
        -9.15419132e-03,  -1.95489034e-02,   3.83475646e-02,
        -4.16735482e-03,   1.00511269e-02,  -3.02719311e-02,
         4.19490416e-02,   1.14798812e-02,  -3.85734442e-03,
        -1.44240622e-02,  -1.90316999e-02,   1.48070301e-02,
         2.57163554e-02,   2.80255108e-02,  -2.80932636e-02,
         1.11571743e-02,   4.54924821e-02,   2.58996220e-02,
         2.61509917e-02,   1.33785099e-02,   3.78600636e-02,
        -1.05038886e-03,   9.08467607e-03,   1.60464047e-02,
         1.49192491e-02,   1.24314205e-03,   9.77079537e-03,
         1.70685547e-02,   2.60251044e-02,   8.03313091e-03,
         6.61072719e-03,  -3.97656720e-02,  -3.12861622e-02,
        -1.57039823e-02,   3.50513532e-02,   1.69355309e-02,
         3.07620083e-02,   2.47554908e-02,  -4.09233509e-02,
         1.63744636e-02,   1.91192252e-02,   3.65160711e-03,
        -4.13080242e-02,  -3.98164336e-02,  -2.18264779e-02,
         1.28016878e-02,   1.07371236e-02,  -8.79462872e-04,
         2.22773041e-02,   1.51235942e-03,   8.03865994e-03,
         2.87451660e-02,   1.97441906e-02,  -1.21854171e-02,
        -1.70303673e-02,   1.94899251e-02,   1.77271892e-02,
        -1.73663615e-03,  -7.90606040e-03,   4.42908550e-02,
         3.19946887e-02,  -9.03399993e-03,  -7.27845673e-03,
         3.91284433e-02,  -1.26320496e-02,   2.10613945e-02,
         1.14229846e-02,   4.27058281e-03,   1.69271094e-02,
         3.73949411e-02,  -2.98521676e-02,   1.64118414e-02,
        -2.65196500e-03,   1.76838922e-02,  -3.20983118e-02,
        -3.88779462e-02,  -1.97898139e-02,  -5.87389934e-02,
         3.27699047e-02,  -2.38977478e-02,  -9.73785074e-03,
         5.66540379e-02,   4.45371164e-02,   3.05180597e-02,
        -3.23245188e-02,  -5.37416689e-02,  -1.50999264e-02,
        -1.07079811e-01,   2.27276551e-02,   4.66713189e-03,
         4.31241791e-03,  -6.16783079e-03,   2.62164314e-02,
        -9.96499874e-03,   6.25358185e-02,  -1.59782510e-02,
        -2.03834467e-03,  -2.11435432e-02,  -2.56492270e-02,
         1.54577296e-02,   5.17533530e-02,   9.52093930e-03,
        -1.42424343e-02,  -2.35835914e-02,   1.52108804e-02,
        -1.95170204e-02,  -6.06709147e-03,   6.14514010e-02,
        -4.78882510e-03,   1.55438378e-03,  -1.34556524e-02,
         4.80654392e-02,   7.06766890e-04,  -2.85965676e-02,
         1.07643140e-02,   5.20248852e-02,   3.43185108e-02,
        -5.21632297e-03,   7.86715148e-04,  -1.94046947e-02,
        -9.96072704e-03,   1.71061532e-02,  -1.36344908e-02,
        -4.19952211e-02,   4.57279153e-02,   3.54418407e-03,
         3.49428199e-02,  -1.47009512e-03,   4.89579855e-02,
         5.02867685e-02,   5.94545896e-02,   1.38900884e-02,
        -5.12376900e-02,  -4.64849999e-02,   1.50191026e-02,
         2.45199374e-02,  -2.70030891e-02,   1.72061375e-02,
        -3.32215924e-02,  -3.69152504e-02,   1.33135692e-02,
         3.07855550e-02,   9.58427742e-03,  -1.25760648e-02,
         1.23400158e-02,  -2.02094775e-02,   2.11168297e-03,
        -4.94346307e-02,   2.17070643e-02,   6.12825425e-02,
         5.06400667e-03,  -1.37552458e-02,   2.83468352e-02,
         6.15868624e-03,   7.74285635e-03,   1.37415361e-02,
         6.34516122e-02,  -9.63845455e-03,  -6.13713091e-02,
         4.20455719e-02,   1.19717616e-03,  -1.31413102e-02,
         3.06006270e-03,   4.27981091e-02,  -9.58403999e-04,
        -2.26399082e-02,  -9.83366353e-03,  -7.62603959e-03,
         6.89370095e-03,  -1.22740820e-02,   7.00931281e-03,
        -6.31101464e-02,  -1.69374814e-02,  -8.85205697e-02,
         3.80205252e-02,  -4.15448766e-03,   9.88235052e-03,
        -1.08853074e-04,   7.02349341e-03,  -3.89461386e-02,
         2.90273673e-03,   1.47432234e-03,  -8.24054746e-04,
        -3.96488889e-02,  -1.17003556e-01,   3.89862183e-02,
        -7.29617434e-03,  -6.32151059e-04,   8.32364318e-03,
         3.88253331e-02,   5.55654248e-02,  -1.04454851e-02,
        -3.04055521e-02,  -1.40969595e-01])
        return (rval_lpost, rval_lgrad, rval_lpost_lgrad, rval_lhess)

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







class PlainPropDistr(object):
    def __init__(self, def_ratio, mu, K):
        self.def_ratio = def_ratio
        self.def_prop_dist = dist.mvt(mu, K*1.5, 25)
        self.prop_dist = dist.mvnorm(mu, K)
    
    def logpdf(self, s):
        return logsumexp([self.def_prop_dist.logpdf(s) + log(self.def_ratio),
                           self.prop_dist.logpdf(s) + log(1-self.def_ratio)],0)
    def rvs(self, num_samps):
        def_num_samps = int(num_samps * self.def_ratio)
        nondef_num_samps = num_samps - def_num_samps
        samps = self.prop_dist.rvs(nondef_num_samps)
        if self.def_ratio > 0:
            samps = np.vstack([samps, self.def_prop_dist.rvs(def_num_samps)])
        
        return samps

class HlrPropDistr(object):
    def __init__(self, def_ratio, mu):
        self.def_ratio = 0.1
        self.def_sigsq_dist = stats.norm(invsoftplus(mu[0]), 20)
        self.def_rest_dist = dist.mvt(mu[1:], np.eye(mu.size - 1)*20, 10)
        self.sigsq_dist = stats.norm(invsoftplus(mu[0]), 3)
        self.rest_dist = dist.mvnorm(mu[1:], np.eye(mu.size - 1)*4)
    
    def logpdf(self, s):
        
        first = invsoftplus(s[:,0])
        rval = logsumexp([self.def_sigsq_dist.logpdf(first) + self.def_rest_dist.logpdf(s[:, 1:]) + log(self.def_ratio),
                           self.sigsq_dist.logpdf(first) + self.rest_dist.logpdf(s[:, 1:]) + log(1.-self.def_ratio)],0)
        return rval
        
    def rvs(self, num_samps):
        def_num_samps = int(num_samps * self.def_ratio)
        nondef_num_samps = num_samps - def_num_samps
        samps = np.hstack([self.sigsq_dist.rvs(nondef_num_samps)[:, np.newaxis], self.rest_dist.rvs(nondef_num_samps)])
        if self.def_ratio > 0:
            samps = np.vstack([samps,
                               np.hstack([self.def_sigsq_dist.rvs(def_num_samps)[:, np.newaxis], self.def_rest_dist.rvs(def_num_samps)])])
        
        samps[:,0] = softplus(samps[:,0])
        assert(np.all(samps[:, 0]>0))
        return samps

def ess(lw):
    return exp(-(logsumexp((lw - logsumexp(lw))*2)))

def credit_simple_is(name_dataset, num_samps, prop_dist = None):
    def_ratio = 0.1    
    if name_dataset == "plain":
        (f, _, _, hess) = credit_plain_theano(10)
        if prop_dist is None:
            prop_dist =  PlainPropDistr(0.1, f.opt10, np.linalg.inv(-hess(f.opt10))) #PlainPropDistr(0.1, m3)
    elif name_dataset == "hlr":
        (f, _, _, hess) = credit_hlr_theano(0.01)
        K = np.linalg.inv(-hess(f.opt_0_01))
        K = K + np.diag([-np.min(np.diag(K))*1.5]*302)
        K = np.eye(302)*0.00001
        prop_dist = dist.transform.softplus(dist.mvnorm(f.opt_0_01, K), [0])
        
    samps = prop_dist.rvs(num_samps)
    lprop = prop_dist.logpdf(samps)[:, None]
    lpost = np.apply_along_axis(f, 1, samps)
    lw = lpost - lprop
    assert(np.all(1- np.isnan(lw)))
    lw_norm = lw - logsumexp(lw)
    
    w_norm = exp(lw_norm)
    mu = np.sum(samps*w_norm, 0)
    var = np.sum((samps-mu)**2*w_norm, 0)
    print("expectation", mu, var)
    print("ess:", ess(lw))
    return {"mu": mu, "var": var, "samps":samps, "lw":lw, "w_norm":w_norm}