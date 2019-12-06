# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:04:27 2019

@author: s.paramonov
"""

import numpy as np


# approximation interval data points:
N = 500
t= np.linspace(0,1,N)


import dist_utils.utils as cd
import pdf_appx.legendre_poly_exp_appx as lpappx


# generating 1-d increasing function of casual shape as a prototype for distribution function
getcdf = cd.casual_increasing_function()

Mycdf = getcdf(t)


import matplotlib.pyplot as plt

plt.plot(t, Mycdf, label='some casual cdf')
plt.legend()
plt.show()

getpdf = getcdf.derivative(1)
Mypdf = getpdf(t)


plt.plot(t, Mypdf, label='some casual pdf')
plt.legend()
plt.show()



num_moments=9
# number of moments, by power (excluding zero power moment ==1)
#for Legendre polynomial expansion, using of polynomials with big powers causes instability

#1  - Building approximation of some casual pdf

moments = cd.moments_from_cdf(getcdf,num_moments)


    
expansion_approx =     lpappx.approx_legendre_poly(moments)
pdf_approx = expansion_approx(t)

plt.plot(t, Mypdf, label='some casual pdf')    
plt.plot(t,pdf_approx, label='pdf approximation')
plt.legend()    
plt.show()

# 2 -  Now create pdf approximation for the random data sample

# Create some transformation as some increasing function:

gettransform = cd.casual_increasing_function()

Mytransform = gettransform(t)

plt.plot(t, Mytransform, label='some transformation function')
plt.legend()
plt.show()

samples_uni = np.random.rand(10000)

samples = gettransform(samples_uni)

# get moment esimations for the data:

moments2 = cd.moments_from_samples(samples, num_moments)
expansion_approx2 =     lpappx.approx_legendre_poly(moments2)
pdf_approx2 = expansion_approx2(t)

plt.hist(samples, bins=50, label = 'histogram', density = True) 
plt.plot(t,pdf_approx2, label='pdf approximation')
plt.legend()    
plt.show()











  
   