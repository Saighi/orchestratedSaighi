import numpy as np
vmax = 300
g = lambda h : np.min(np.exp(h-5),vmax)

def evolve_pop():
    return -vx + g(yexc*wxa*s(va)) + yext*wxe