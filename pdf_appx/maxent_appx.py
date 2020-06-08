# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:27:18 2019

@author: s.paramonov
"""

import numpy as np
from scipy import integrate


def z_est(lambdas):
    
    npow = np.arange(lambdas.shape[0])+1.
    
    def I(x):
        return (np.exp( -np.sum(  (np.power(x,npow)) * lambdas  )) )
    
    
    return integrate.quad(I, a=0, b=1)[0]

# means estimations from approximated pdf:
def means_est(lambdas, k):
    
    npow = np.arange(lambdas.shape[0])+1.
    
    def numerat(x):
        return ( np.power(x,k)) *  (np.exp( -np.sum(  (np.power(x,npow)) * lambdas  )))
    def denom (x):
        return np.exp( -np.sum(  (np.power(x,npow)) * lambdas  ))
     
    
    return (integrate.quad(numerat, a=0, b=1))[0] / (integrate.quad(denom, a=0, b=1))[0]
    

def Means(lambdas):
    
    M_est = np.zeros((lambdas.shape[0]))
    for i in range(M_est.shape[0]):
        
        M_est[i] = means_est(lambdas, (i+1))
        
    return M_est
        
                    
# Hessian matrix for 2-nd order mimnimization
def Hessian(lambdas):
        
    H = np.zeros(( (lambdas.shape[0] ), (lambdas.shape[0]) ))
    
    for n in range (H.shape[0]):
        for m in range (H.shape[0]):
            
            H[n,m] = means_est(lambdas, (n+1+m+1)) - means_est(lambdas, m+1)*means_est(lambdas, n+1)  
        
    return H
        


# opimize in cycle:
def optimization_4_lambdas(moments):
    
    lambdas = np.zeros((moments.shape[0] - 1)) # ecluding moments[0]==1
    
    for q in range(10):
        lambdas_ = lambdas
    
        H = Hessian(lambdas_)
           
        B = moments[1:] - Means(lambdas_)
        
        A = np.linalg.solve(H, B)
    
        lambdas = lambdas_ - A
        
    Z = z_est(lambdas)
    lambda0 = np.log(Z)
    
    return lambda0, lambdas 
        

# building approximation function for pdf

def pdf_approx(moments):
    
    Lambda0, Lambdas =    optimization_4_lambdas(moments)
    
    def pdf_calc(x):
        
        npow = np.expand_dims((np.arange(Lambdas.shape[0])+1.),axis=1 )
    
        S = -np.sum(  (np.power( x,npow)).T * Lambdas, axis=1, keepdims=True  )
    
        
        return np.exp(S - Lambda0)
    
    return pdf_calc
    


  
