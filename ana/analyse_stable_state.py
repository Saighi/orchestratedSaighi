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
from elephant.statistics import isi
import plotly.graph_objects as go
from neo.core import SpikeTrain
from scipy import signal
from scipy.io import mmread
import seaborn as sns
import plotly.graph_objects as go

# %%
num_mpi_ranks = 4 # the number of sims you used in parallel
datadir = os.path.expanduser("/mnt/data1/data_paul/sim_less_stim_neurons_more_inside_connect_nocons") # Set this to your data path
prefix = "rf1"

#%%
spkfiles  = ["%s/%s.%i.e.spk"%(datadir,prefix,i) for i in range(num_mpi_ranks)]
sfo = AurynBinarySpikeView(spkfiles)

# %%
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
# %%
see  = np.concatenate([pd.read_csv("%s/%s.%i.see"%(datadir,prefix,i),delimiter=' ' ).values[:,1:-1] for i in range(3)],axis=1)
sie  = np.concatenate([pd.read_csv("%s/%s.%i.sie"%(datadir,prefix,i),delimiter=' ' ).values[:,1:-1] for i in range(3)],axis=1)
sse  = np.concatenate([pd.read_csv("%s/%s.%i.sse"%(datadir,prefix,i),delimiter=' ' ).values[:,1:-1] for i in range(3)],axis=1)
# %%
# datadi_suite = os.path.expanduser("/mnt/data2/paul_data/Auryn_archives/sim_stady_state_wii0.08_wie0.08_wei0.72_suite")
# # %%
# see_suite  = np.concatenate([pd.read_csv("%s/%s.%i.see"%(datadi_suite,prefix,i),delimiter=' ' ).values[:,1:-1] for i in range(3)],axis=1)
# sie_suite  = np.concatenate([pd.read_csv("%s/%s.%i.sie"%(datadi_suite,prefix,i),delimiter=' ' ).values[:,1:-1] for i in range(3)],axis=1)
# sse_suite  = np.concatenate([pd.read_csv("%s/%s.%i.sse"%(datadi_suite,prefix,i),delimiter=' ' ).values[:,1:-1] for i in range(3)],axis=1)
# # %%
# spkfiles_suite  = ["%s/%s.%i.e.spk"%(datadi_suite,prefix,i) for i in range(num_mpi_ranks)]
# sfo_suite = AurynBinarySpikeView(spkfiles_suite)
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
#plt.plot(time_axis_I[10000000:],np.convolve(rateI[10000000:],win,'same')/ sum(win),label = "Auryn")
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
# %%
plt.plot(np.median(see,axis = 1))
# %%
plt.plot(np.median(sse,axis = 1))
#%%
see.shape
# %%
plt.hist(wext.data,bins=50,log=True);
# %%
plt.hist(w.data,bins=50,log=True);
# %%
plt.hist(wie.data,bins=50,log=True);
# %%
see_all = np.concatenate((see,see_suite),axis=0)
# %%
sie_all = np.concatenate((sie,sie_suite),axis=0)
# %%
sse_all = np.concatenate((sse,sse_suite),axis=0)
#%%
total_time =1
#total_time = see.shape[0]
#values = np.log(see[t,1:-1][np.where(see[t,1:-1]!=0)[0]])
#print(values)
plt.hist(see[total_time,1:-1],bins=50)

#%%
fig,ax = plt.subplots()
for t in range(int(see.shape[0]/4),see.shape[0],int(see.shape[0]/4)):
    values = np.log(see[t,1:-1][np.where(see[t,1:-1]!=0)[0]])
    print(values)
    ax.hist(values,bins=50,alpha=0.2)
    #plt.hist(weight_InputE["w"][:,t],alpha=0.7,bins=15)
ax.set_xlabel("weights E->E")
ax.set_title("Time: "+str(t)+"s")
plt.show()
#%%

for t in range(int(see.shape[0]/4),see.shape[0],int(see.shape[0]/4)):
    values = np.log(see[t,1:-1][np.where(see[t,1:-1]!=0)[0]])
    plt.hist(values,bins=50,alpha=0.2)
    #plt.hist(weight_InputE["w"][:,t],alpha=0.7,bins=15)
plt.set_xlabel("weights E->E")
plt.xscale('log') 
plt.show()
# %%
see.shape
# %%
plt.plot(np.concatenate((np.mean(sse,axis = 1),np.mean(sse_suite,axis = 1)) ) )
# %%
plt.plot(np.concatenate((np.mean(see,axis = 1),np.mean(see_suite,axis = 1)) ) )
# %%
plt.plot(np.concatenate((np.mean(sie,axis = 1),np.mean(sie_suite,axis = 1)) ) )
# %%
plt.plot(np.mean(sie_suite,axis = 1))
# %%
plt.plot(np.mean(see,axis = 1))
# %%
plt.plot(np.median(sie,axis = 1))


# %%
total_time = 20000
trange = 100
numberof = 20
for t in range(int(total_time/numberof),total_time,int(total_time/numberof)):
    times, cells = np.array(sfo.get_spikes(t_start=t,t_stop=t+(trange/numberof))).T
    freqs = []
    cvs = []
    for i in range(NE):
        isi = np.diff(times[cells==i])
        isim = isi.mean()
        freqs.append(1/isim)
        cvs.append(isi.std()/isim)
    plt.hist(freqs,bins=50)
    plt.show()
# %%
# trange = 100
# for t in range(int(total_time/4),total_time,int(total_time/4)):
#     spikes = np.array(sfo_suite.get_spikes(t_start=t,t_stop=t+trange))[:,0]
#     the_isi = isi(spikes)
#     plt.hist(the_isi,log=True,alpha=0.5);
# plt.show()
# %%
NE = 4096
time_start = 21700
trange = 10
times, cells = np.array(sfo.get_spikes(t_start=time_start,t_stop=time_start+trange)).T
freqs = []
cvs = []
for i in range(NE):
    isi = np.diff(times[cells==i])
    isim = isi.mean()
    freqs.append(1/isim)
    cvs.append(isi.std()/isim)

# %%
winner = np.where(np.array(freqs)>12)[0]
# %%
inside_weight = []
for n in winner:
    for n2 in winner :
        if n!= n2 :
            poid = w[n,n2]
            if poid> 0.0:
                inside_weight.append(w[n,n2])
# %%
# La moyenne des poids internes au group sont plus grand que la moyenne des poids externes        
# %%
np.mean(inside_weight)
# %%
np.mean(w.data)
# %%
inside_weightExt = []
for n in range(NE):
    for n2 in winner :
        poid = wext[n,n2]
        if poid> 0.0:
            inside_weightExt.append(wext[n,n2])
# %%
np.mean(inside_weightExt)
# %%
np.mean(wext[:, np.where(np.array(freqs)<12)[0]].data)

# %%
wext[:, np.where(np.array(freqs)<12)[0]]
# %%
