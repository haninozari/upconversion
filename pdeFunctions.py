# -*- coding: utf-8 -*-
"""
Created on Sun May 22 10:22:02 2016

@author: haninm_adm
"""
import math
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def uFun(*args):
    F, i, j, alpha = args[0], args[1], args[2], args[3]   
    fx = (F[i+1,j]-F[i-1,j])/2
    fxx = (F[i+1,j]- 2*F[i,j] + F[i-1,j])
    fy = (F[i,j+1]-F[i,j-1])/2
    fyy = (F[i,j+1]- 2*F[i,j] + F[i,j-1])
    fxy = (F[i+1,j+1] + F[i-1,j-1] - F[i+1,j-1] - F[i-1,j+1])/4
    L = math.sqrt(1 + fx**2 + fy**2)
    den = L + 2 * alpha**2 * L**2
    U = fx * (fx * fxx + fxy * fy) + fy * (fy * fyy + fxy * fx)
    return U/den
    
def gradient(func):
    fx = np.zeros_like(func)
    fy = fx
    Nx, Ny = func.shape[0], func.shape[1]
    hx, hy = 1.0/(Nx - 1), 1.0/(Ny - 1)
    
    for i in range(1,fx.shape[0] - 1):
        for j in range(1, fx.shape[1] - 1):
            fx[i, j] = (func[i + 1, j] - func[i - 1, j])/2/hx
            fy[i, j] = (func[i, j + 1] - func[i, j - 1])/2/hy
    return (fx, fy)
def area(func):
    (fx, fy) = np.gradient(func)
    #fx = 100 * fx
    #fy = 100 * fy
    g = np.ones_like(fx) + fx**2 + fy**2
    #N = np.size(g)
    A = np.sqrt(g)
    return np.sum(A)
    
def linearSurface(func1, func2, dis):
    M = np.zeros((func1.size, dis))
    M[:, 0] = func1
    M[:, dis - 1] = func2
    for i in range(1, dis - 1):
        M[:, i] = (i * func2 + (dis - 2 - i) * func1)/(dis - 2)
    return M
def mesh(func):  
    x, y = np.meshgrid(np.linspace(0, 1, func.shape[1]), np.linspace(0,1,func.shape[0]))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, func, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

def idealRollFunc(func1, func2, dis):
    M = np.zeros((func1.size, dis))
    M[:, 0] = func1
    M[:, dis - 1] = func2
    for i in range(1, dis - 1):
        M[:, i] = 100 * gauss(-1 + 2 * i/(dis - 1), .6, dis, -3, 3)
    return M
def gauss(mean,sigma, length, a, b):
    t = np.linspace(a, b, length)
    f = np.exp(- (t - mean)**2 / 2 / sigma**2) / 2*np.pi / np.sqrt(sigma)
    return f

def gaussCurvature(F):
    '''    
    Nx, Ny = F.shape[0], F.shape[1]
    hx, hy = 1.0/(Nx - 1), 1.0/(Ny - 1)
    fxx = np.zeros_like(F)
    fyy = fxx
    fxy = fxx
    
    for i in range(1, F.shape[0] - 1):
        for j in range(1, F.shape[1] - 1):
            fxx[i, j] = (F[i+1,j]- 2*F[i,j] + F[i-1,j])/hx/hy
            fyy[i, j] = (F[i,j+1]- 2*F[i,j] + F[i,j-1])/hx/hy
            fxy[i, j] = (F[i+1,j+1] + F[i-1,j-1] - F[i+1,j-1] - F[i-1,j+1])/4/hx/hy
    c = fxx * fyy - fxy**2
    '''
    Fx, Fy = np.gradient(F)
    Fxx, Fxy = np.gradient(Fx)
    _, Fyy = np.gradient(Fy)
    c = Fxx * Fyy - Fxy**2    
    return (c**2, np.sum(c**2))
    
def gradientArea(func):
    (fx, fy) = np.gradient(func)
    G = fx**2 + fy**2
    return area(G)

def gradientMagnitude(func):
    (fx, fy) = np.gradient(func)
    G = fx**2 + fy**2
    return np.sqrt(G)

def makeSurf():
    N = 100
    F = np.zeros((N,N))
    for i in range(N):
        f1 = 100 * gauss(-5 + i * 5/N, 2, N, -10, 10)
        f2 = 100 * gauss(5 - i* 5/N, 2, N, -10, 10)
        f = f1 + f2
        f = f/f.max()
        F[:, i] = f
    return F

        