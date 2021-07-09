#%%
from sys import prefix
import numpy as np
from scipy.io import mmread 
import pandas as pd
import matplotlib.pyplot as plt
#%%
datadir = "/mnt/data1/data_paul/custom_plas/sim_test"
prefix="rf1"

rateE  = pd.read_csv("%s/%s.%i.e.prate"%(datadir,prefix,0),delimiter=' ').values 
time_axis = rateE[:,0]

see  = pd.read_csv("%s/%s.%i.sse"%(datadir,prefix,0),delimiter=' ',comment='#' ).values[:,1:-1] 



#%%
plt.plot(see)
# %%
plt.plot(time_axis,rateE)