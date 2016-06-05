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
N = 100
h = 1.0/(N-1)
F = np.ones((N,N))

iter = 1000
alpha = 1
#f1 = 20 * signal.gaussian(N, 3); f1 = np.roll(f1, 10)
#f2 = 20 * signal.gaussian(N, 4); f2 = np.roll(f2, -10)
eps = 1e-20
g1 = 100 * pFun.gauss(-5, 2, N, -10, 10)
g2 = 100 * pFun.gauss(5, 2, N, -10, 10)
g = g1 + g2
f2 = 100 * g/g.max() #+ 5 * np.random.rand(1,N).reshape((N,))
f1 = 100 * pFun.gauss(0, 2, N, -10, 10)#+ 5 * np.random.rand(1,N).reshape((N,))
#F = pFun.linearSurface(f1, f2, N)

for i in range(N):
    F[i, 0] = f2[i]
    F[i, N-1] = f1[i]
    #F[i, 1] = i * h    
    #F[i, N-2] = i**2 * h**2    
    F[0, i] = 0.0
    F[N-1, i] = 0.0
'''
for itr in range(iter):
    for i in range(1, N-1):
        for j in range(1, N-1):
            F[i,j] = (F[i+1,j] + F[i-1,j] + F[i,j+1] + F[i,j-1])/4
'''




for itr in range(iter):
    for i in range(1, N-1):
        for j in range(1, N-1):
            
            args = (F, i, j, alpha)
            F[i,j] = F[i+1,j] + F[i-1,j] + F[i,j+1] + F[i,j-1] - pFun.uFun(*args)
            F[i,j] = F[i, j]/4
'''
for j in range(N):
    F[:, j] = F[:,j]/np.max(F[:,j])

       
ind = 1
Den = np.zeros((iter*N*N))
for itr in range(iter):
    
    for i in range(1, N-1):
        for j in range(1, N-1):
            a1, b1, c1 = F[i+1, j], F[i, j], F[i-1, j]
            a2, c2 = F[i, j+1], F[i, j-1]
            fxy = (F[i+1, j+1] + F[i-1, j-1] - F[i+1, j-1] - F[i-1, j+1])/4.0
            #Fx, Fy = np.gradient(F)
            #Fxx, Fxy = np.gradient(Fx)            
            #_, Fyy = np.gradient(Fy)
            num = 4 * b1**2 + (a1 + c1)*(a2 + c2) - fxy**2
            #num = 4 * b1**2 + (a1 + c1)*(a2 + c2) - Fxy[i,j]**2            
            den = 2*(a1 + a2 + c1 + c2)
            if (abs(den) < 1e-10):
                den = 1e-10
            
            #num = 4 * b1**2 + (Fxx[i,j] + 2 * b1)*(Fyy[i,j] + 2*b1) - Fxy[i,j]**2          
            #den = 2*(Fxx[i,j] + Fyy[i,j] + 4*b1) + 1
            F[i,j] = num/den
            if (F[i,j] > 100):
                F[i,j] = 100
            if (F[i,j] < 0):
                F[i,j] = 0
'''



#hi = pFun.idealRollFunc(f1, f2, N)
hm = pFun.linearSurface(f1, f2, N)
hi = pFun.makeSurf()
pFun.mesh(F)
pFun.mesh(hi)
pFun.mesh(hm)

print(pFun.area(F))
print(pFun.area(hi))
print(pFun.area(hm))

(cF, cFv) = pFun.gaussCurvature(F)
(chi, chiv) = pFun.gaussCurvature(hi)
(chm, chmv) = pFun.gaussCurvature(hm)
print(cFv)
print(chiv)
print(chmv)

pFun.mesh(cF)
pFun.mesh(chi)
pFun.mesh(chm)

#%%
'''
t = np.linspace(-10,10,N)
plt.figure() 
for j in range(N):
    y = np.transpose(F[:,j])   
    plt.plot(t, y)
    plt.pause(.2)
    plt.cla()
'''