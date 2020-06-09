#!/usr/bin/env python
# coding: utf-8



import numpy as np
import matplotlib.pyplot as plt

import pdf_appx.pda_utils as cd
import pdf_appx.lp_appx as lpappx

# Define points of approximation in (0;1) and number of moments (by power, excluding zero power moment ==1) will be used for approximation:

N = 500 
t= np.linspace(0,1,N)
num_moments=9

# We have function that creates casual (nearly random) increasing fucntion that determines some CDF and PDF:

getcdf = cd.casual_increasing_function()
print(type(getcdf))

Mycdf = getcdf(t)
plt.plot(t, Mycdf, label='casual cdf')
plt.legend()
plt.show()

getpdf = getcdf.derivative(1)
Mypdf = getpdf(t)

plt.plot(t, Mypdf, label='casual pdf')
plt.legend()
plt.show()


# Get moments of this function for Legendre polynomial expansion of PDF. Using of polynomials with big powers causes instability.

num_moments=9
moments = cd.moments_from_cdf(getcdf,num_moments)

expansion_approx =     lpappx.approx_legendre_poly(moments)
pdf_approx = expansion_approx(t)

plt.plot(t, Mypdf, label='some casual pdf')    
plt.plot(t,pdf_approx, label='pdf approximation')
plt.legend()    
plt.show()

#  Now create pdf approximation for the random data sample. We will use uniform distributed samples transformed by casual increasing function.

gettransform = cd.casual_increasing_function()
Mytransform = gettransform(t)

plt.plot(t, Mytransform, label='some transformation function')
plt.legend()
plt.show()

samples_uni = np.random.rand(10000)
samples = gettransform(samples_uni)

# Lets get expansion coefficients from estimation of moments

moments2 = cd.moments_from_samples(samples, num_moments)
expansion_approx2 =     lpappx.approx_legendre_poly(moments2)
pdf_approx2 = expansion_approx2(t)

plt.hist(samples, bins=50, label = 'histogram', density = True) 
plt.plot(t,pdf_approx2, label='pdf approximation')
plt.legend()    
plt.show()

