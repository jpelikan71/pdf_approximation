# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:27:18 2019

@author: s.paramonov
"""

import numpy as np


# points of approximation:
N = 500 
t= np.linspace(0,1,N)

# number of moments, by power (excluding zero power moment ==1)
num_moments=9

import dist_utils.utils as cd
import pdf_appx.maxent_appx as meappx

# generating 1-d increasing function of casual shape as prototype for distribution function
getcdf = cd.casual_increasing_function()

Mycdf = getcdf(t)


import matplotlib.pyplot as plt

plt.plot(t, Mycdf, label='casual cdf')
plt.legend()
plt.show()

getpdf = getcdf.derivative(1)
Mypdf = getpdf(t)


plt.plot(t, Mypdf, label='casual pdf')
plt.legend()
plt.show()

# calculate moments from cdf function:
moments = cd.moments_from_cdf(getcdf,num_moments)


maxent_approx_1 =  meappx.pdf_approx(moments)

pdf_hat = maxent_approx_1(t)


    
plt.plot(t, Mypdf, label='some casual pdf')    
plt.plot(t,pdf_hat, label='pdf approximation')
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

moments_star = cd.moments_from_samples(samples, num_moments)

maxent_approx_2 =  meappx.pdf_approx(moments_star)

pdf_star_hat = maxent_approx_2(t)


plt.hist(samples, bins=50, label = 'histogram', density = True) 
plt.plot(t,pdf_star_hat, label='pdf approximation')
plt.legend()    
plt.show()





  
