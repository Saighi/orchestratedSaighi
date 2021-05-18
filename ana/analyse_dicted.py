# %%
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.expanduser("~/auryn/tools/python/"))
from auryntools import *
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
import quantities as pq
from elephant.spike_train_generation import homogeneous_poisson_process
from elephant.statistics import instantaneous_rate
import plotly.graph_objects as go
from neo.core import SpikeTrain
from scipy import signal
from scipy.io import mmread
import seaborn as sns
import plotly.graph_objects as go

# %%
time_step =0.0001

# %%
num_mpi_ranks = 4 # the number of sims you used in parallel
datadir = os.path.expanduser("/mnt/data1/data_paul/sim_rate20") # Set this to your data path
prefix = "rf1"

#%%
spkfiles  = ["%s/%s.%i.e.spk"%(datadir,prefix,i) for i in range(num_mpi_ranks)]
sfo = AurynBinarySpikeView(spkfiles)

# %%
####Auryn
rateE  = np.mean([pd.read_csv("%s/%s.%i.e.prate"%(datadir,prefix,i),delimiter=' ').values for i in range(num_mpi_ranks)],axis=0)
time_axis = rateE[:,0]
rateE= rateE[:,1]
rateI  = np.mean([pd.read_csv("%s/%s.%i.i2.prate"%(datadir,prefix,i),delimiter=' ' ).values for i in range(num_mpi_ranks)],axis=0)
time_axis_I = rateI[:,0]
rateI= rateI[:,1]

wmatfiles  = ["%s/rf1.%i.ee.wmat"%(datadir,i) for i in range(num_mpi_ranks)]
w = np.sum( [ mmread(wf) for wf in wmatfiles ] )

wmatfilesext  = ["%s/rf1.%i.ext.wmat"%(datadir,i) for i in range(num_mpi_ranks)]
wext = np.sum( [ mmread(wf) for wf in wmatfilesext ] )

wmatfilesie  = ["%s/rf1.%i.ie.wmat"%(datadir,i) for i in range(num_mpi_ranks)]
wie = np.sum( [ mmread(wf) for wf in wmatfilesie ] )

see  = np.concatenate([pd.read_csv("%s/%s.%i.see"%(datadir,prefix,i),delimiter=' ' ).values[:,1:-1] for i in range(3)],axis=1)
sie  = np.concatenate([pd.read_csv("%s/%s.%i.sie"%(datadir,prefix,i),delimiter=' ' ).values[:,1:-1] for i in range(3)],axis=1)
sse  = np.concatenate([pd.read_csv("%s/%s.%i.sse"%(datadir,prefix,i),delimiter=' ' ).values[:,1:-1] for i in range(3)],axis=1)

# %%
win = signal.windows.hann(10)
plt.plot(time_axis,np.convolve(rateE,win,'same')/ sum(win),label = "Auryn",alpha = 0.75)
plt.legend()
# %%
win = signal.windows.hann(10)
plt.plot(time_axis_I,np.convolve(rateI,win,'same')/ sum(win),alpha = 0.75)
#fig = go.Figure(data=go.Scatter( y=rateI[6000000:]))
#fig.add_trace(go.Scatter( y=excitatory["gampa"][0]))

#fig.show()
# %%
plt.plot(time_axis_I[10000:],rateI[10000:],label = "Auryn")
# %%
plt.hist(w.data, bins=100, log=True,label="Auryn")
plt.title("EE weight distribution")
sns.despine()
# %%
plt.hist(wext.data, bins=100, log=True,label="Auryn")
plt.title("Ext->E weight distribution")
sns.despine()
# %%
plt.hist(wie.data, bins=100, log=True,label="Auryn")
plt.title("I->E weight distribution")
sns.despine()
# %%
plt.plot(np.mean(sse,axis = 1))
# %%
plt.plot(np.mean(sie,axis = 1))
# %%
plt.plot(np.mean(see,axis = 1))
# %%
plt.plot(np.median(sie,axis = 1))
#%%
see.shape
# %%
plt.hist(see[3000,:],bins=50)
# %%
np.mean(see[4000,:])
# %%
