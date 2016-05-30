# -*- coding: utf-8 -*-
"""
Created on Sun May 22 10:19:55 2016

@author: haninm_adm
"""
'''
import pdb
pdb.set_trace()
'''

import numpy as np
from scipy import signal
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pdeFunctions as pFun

plt.close('all')
N = 50 
h = 1.0/(N-1)
#F = np.zeros((N,N))
iter = 100
alpha = 10
#f1 = 20 * signal.gaussian(N, 3); f1 = np.roll(f1, 10)
#f2 = 20 * signal.gaussian(N, 4); f2 = np.roll(f2, -10)
eps = 1e-20
f1 = 100 * pFun.gauss(-1, .6, N, -3, 3)
f2 = 100 * pFun.gauss(1, .6, N, -3, 3)
hm = pFun.linearSurface(f1, f2, N)
pFun.mesh(hm)
F = hm
for i in range(N):
    F[i, 0] = f2[i]
    F[i, N-1] = f1[i]
    #F[i, 1] = i * h    
    #F[i, N-2] = i**2 * h**2    
    F[0, i] = 0
    F[N-1, i] = 0

for itr in range(iter):
    for i in range(1, N-1):
        for j in range(1, N-1):
            
            args = (F, i, j, h, alpha)
            F[i,j] = F[i+1,j] + F[i-1,j] + F[i,j+1] + F[i,j-1] - h**2 * pFun.uFun(*args)
            F[i,j] = F[i, j]/4
'''            
ind = 1
Den = np.zeros((iter*N*N))
for itr in range(iter):
    
    for i in range(1, N-1):
        for j in range(1, N-1):
            a1, b1, c1 = F[i+1, j], F[i, j], F[i-1, j]
            a2, c2 = F[i, j+1], F[i, j-1]
            fxy = (F[i+1, j+1] + F[i-1, j-1] - F[i+1, j-1] - F[i-1, j+1])/4.0
            num = 4 * b1**2 + (a1 + a2)*(c1 + c2) #+ fxy**2 + (a1 + c1 + a2 + c2 -4 * b1)
            den = 2*(a1 + a2 + c1 + c2) + .1
            Den[ind] = den
            ind += 1
            if den != 0:
                F[i,j] = num/den
'''

hi = pFun.idealRollFunc(f1, f2, N)

pFun.mesh(F)
pFun.mesh(hi)
pFun.mesh(hm)

print(pFun.area(F))
print(pFun.area(hi))
print(pFun.area(hm))

print(pFun.gradientArea(F))
print(pFun.gradientArea(hi))
print(pFun.gradientArea(hm))