# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:04:27 2019

@author: s.paramonov
"""

import numpy as np

from numpy.polynomial import Polynomial, Legendre, polynomial

def approx_legendre_poly(Moments):
    
    n_moments = Moments.shape[0]-1
    
    exp_coef = (np.zeros((1)))

    # For method description see, for instance: 
    # Chapter 3 of "The Problem of Moments", James Alexander Shohat, Jacob David Tamarkin
    for i in range(n_moments+1):
        p = Legendre.basis(i).convert(window = [0.0,1.0], kind=Polynomial)
       
        q = (2*i+1)*np.sum(Moments[0:(i+1)]*p.coef)
        
        pq = (p.coef*q)
                
        exp_coef = polynomial.polyadd(exp_coef, pq)

            
    expansion = Polynomial(exp_coef)
   
        
    return expansion
