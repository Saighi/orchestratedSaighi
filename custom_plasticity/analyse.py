#%%
from sys import prefix
import numpy as np
from scipy.io import mmread 
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
#%%
datadir = "/mnt/data1/data_paul/custom_plas/sim_test"
prefix="rf1"
num_mpi_ranks = 4
rateE_av  = np.mean([pd.read_csv("%s/%s.%i.e.prate"%(datadir,prefix,0),delimiter=' ').values for i in range(num_mpi_ranks)],axis=0)
time_axis = rateE_av[:,0]
rateE = rateE_av[:,1]

rateI_av  = np.mean([pd.read_csv("%s/%s.%i.i.prate"%(datadir,prefix,0),delimiter=' ').values for i in range(num_mpi_ranks)],axis=0)
rateI = rateI_av[:,1]

sse  = np.concatenate([ pd.read_csv("%s/%s.%i.sse"%(datadir,prefix,0),delimiter=' ',comment='#' ).values[:,1:-1] for i in range(num_mpi_ranks)],axis=1)
sie  = np.concatenate([pd.read_csv("%s/%s.%i.sie"%(datadir,prefix,0),delimiter=' ',comment='#' ).values[:,1:-1] for i in range(num_mpi_ranks)],axis=1)
#%%
win = signal.windows.hann(100)
#%%
plt.plot(np.mean(sse,axis=1))
# %%
plt.plot(time_axis,(np.convolve(rateE,win,'same')/ sum(win)))
# %%µ
plt.plot(np.mean(sie,axis=1))
# %%
plt.plot(time_axis,(np.convolve(rateI,win,'same')/ sum(win)))
# %%
plt.hist(sse[:,0])
# %%
plt.hist(sse[-1,:],bins=100)
# %%
np.mean(sse[0,:])
# %%
