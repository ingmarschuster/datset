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

import pymc3 as pm
import theano.tensor as T
import theano
from datset.data.stoch_vol_opt import optimum
import pickle  



import os

__all__=["stoch_vol", "StochVolNUTS"]


class GradLogpOp(theano.Op):
    __props__ = ("input_shape", "grad_logp")
    def __init__(self, input_shape, grad_logp):
        self.input_shape = input_shape
        self.grad_logp = grad_logp

    def infer_shape(self, node, shapes):
        return (self.input_shape,)

    def make_node(self, x, logp_x, outputs):
        x = theano.tensor.as_tensor_variable(x)
        #assert x.shape == self.shape_input
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        #outputs[0][0] = self.f(inputs[0])
        outputs[0][0] = np.array(self.grad_logp(inputs[0]))
        #assert()



class LogpOp(theano.Op):
    __props__ = ("logp", "grad_logp_op")
    def __init__(self, input_shape, logp, grad_logp):
        self.logp = logp
        self.grad_logp_op = GradLogpOp(input_shape, grad_logp)

    def infer_shape(self, node, shapes):
        return (),

    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x, )
        #assert x.shape == self.shape_input
        rval = theano.Apply(self, [x], [theano.tensor.TensorType(x.dtype,())()]) #scalar output
        #rval = theano.tensor.as_tensor_variable(rval)
        #print(rval.ndim)
        return rval


    def perform(self, node, inputs, outputs):
        #outputs[0][0] = self.f(inputs[0])
        for i in range(len(inputs)):
            outputs[i][0] = np.array(self.logp(inputs[i]))
        #print(outputs[0][0].ndim, )

    def grad(self, inputs, gradients):
            return [self.grad_logp_op(inputs[0], self(inputs[0]),
                                             gradients[0])]

class StochVolNUTS(pm.Continuous):
    """
    Black Box Continuous distribution which only calls an underlying log density function
    """
    def __init__(self, *args, **kwargs):
        default_value = np.zeros(3001)
        (f, gradf, f_gradf) = stoch_vol(transform = 'sp', theano_symbol = False, neg=False, opt_grad = False)
        kwargs['shape'] = default_value.shape
        super(BlackBoxContinuous, self).__init__(*args, **kwargs)
        self.mean = self.mode = default_value
        self.logp_op = LogpOp(kwargs['shape'], f, gradf) 

    def logp(self, value):
        return self.logp_op(value)
        
        

def tnorm(mu, stddev):
    return stats.truncnorm(-mu/stddev, np.inf, loc=mu, scale=stddev)

def t_logpdf(x, nu):
    import theano.tensor as T

    const = (T.gammaln((nu + 1) / 2) -(T.gammaln(nu/2) + (T.log(nu)+T.log(np.pi)) * 0.5))
    return const - (nu+1)/2*T.log(1+x**2/nu)

def stoch_vol(transform = None, theano_symbol = False, neg=False, opt_grad = False):
    import theano.tensor as T
    import theano
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
    lpost = lpost - 1500.5* T.log(0.01+0.5*T.sum((T.log(s[1:]) - T.log(s[:-1]))**2))
    
    if neg:
        lpost = -lpost
    if theano_symbol:
        return (lpost,param)
    
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