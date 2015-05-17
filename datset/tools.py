# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:18:09 2015

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import scipy as sp
import scipy.stats as stats

from numpy import exp, log, sqrt
from scipy.misc import logsumexp
from numpy.linalg import inv

__all__ = ["log_sign", "exp_sign", "logsumexp_sign", "convert_to_lpost_and_grad_form"]


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