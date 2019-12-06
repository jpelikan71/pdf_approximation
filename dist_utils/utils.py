# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 12:48:35 2019

@author: s.paramonov
"""

import numpy as np
from scipy import integrate
from scipy import interpolate
import random

# random transform for pdf generation/sampling

def casual_increasing_function():
    symmetry = random.randint(0,1)
    if (symmetry==0):
        nnodes = random.randint(5,12)
        x = np.linspace(0,1,(nnodes+2))
        y = np.sort((np.random.rand(nnodes)))

        y = np.append(  np.array([0])  ,y)
        y = np.append(y,  np.array([1])  )
   
    if (symmetry==1):
        snodes = random.randint(2,4)
        nnodes = snodes*2+3

        x = np.linspace(0,1,nnodes)
        y = np.zeros((nnodes))
        y1 = np.sort((np.random.rand(snodes)))*0.5

        y1 = np.append(  np.array([0])  ,y1)

        y2 = 1. - (np.flipud(  y1  ) )
        y[0:snodes+1] = y1
        y[(snodes+2):nnodes]=y2

        y[snodes+1] = 0.5
        
        
    f = interpolate.PchipInterpolator(x,y)

    return f
            
            
def moments_from_pdf(pdf,num_moments):
    
    moments = np.ones((num_moments+1))
            
    for k in range (num_moments+1):
        v = lambda x:(pdf(x))*(x**k)
        J, err = integrate.quad(v,0,1)
       
        moments[k] = J
            
    return moments
            
def moments_from_cdf(cdf,num_moments):
    
    moments = np.ones((num_moments+1))
            
    for k in range (1,num_moments+1):
        v = lambda x:(cdf(x))*(x**(k-1))
        J, err = integrate.quad(v,0,1)
       
        moments[k] = 1-J*k
            
    return moments            






def moments_from_samples(X, num_moments):
    
    Xp = np.expand_dims(X, axis = 0)
       
    powers = np.expand_dims((np.arange(num_moments+1)), axis = 1)
    
    
    moments = np.mean( np.power(Xp, powers)   , axis=1)
    
    return moments
    
    
    
    
    