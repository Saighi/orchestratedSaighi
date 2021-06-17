#!/usr/bin/env python
# coding: utf-8

# In[511]:


import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import signal
from tools import *

from sklearn.decomposition import NMF, PCA

from scipy.sparse import *
from scipy.io import mmread 

# Import auryn tools
sys.path.append(os.path.expanduser("~/auryn/tools/python/"))
from auryntools import *
from matplotlib import animation, rc
from IPython.display import HTML
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import tools
import elephant
import quantities as pq
from neo.core import SpikeTrain
import viziphant
from pyvis.network import Network
import networkx as nx
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import plotly.io as pio
from hmmlearn import hmm
import numba 
from multiprocessing import Pool
import pandas as pd 


# In[512]:


def show_hist_times(event_in_time,times,bins=100,log = False):
    
    pio.renderers.default = 'notebook'

    # plotly setup
    fig=go.Figure()

    # data binning and traces
    for i, col in enumerate(times):
        hist = np.histogram(event_in_time[col], bins=bins, density=False)
        if log :
            x =np.ma.log(hist[0].tolist())
            a0=x.filled(0)
        else:
            a0=hist[0].tolist()
        
        a0=np.repeat(a0,2).tolist()
        a0.insert(0,a0[0])
        a0.pop()
        a1=hist[1].tolist()
        a1=np.repeat(a1,2)
        fig.add_traces(go.Scatter3d(x=[col]*len(a0), y=a1, z=a0,
                                    mode='lines',
                                    name=col,
                                   )
                      )
        

    fig.update_layout(
    height=700,
    )
    fig.update_layout(scene_aspectmode="manual")
    fig.update_layout(scene_aspectratio=dict(x=len(times)*0.35, y=1, z=1))

    fig.show()
    


# In[513]:


dure_simu = 1000
begining_presentation = 0
duree_pattern = 0.1
time_step = 0.1
auryn_tstep = 0.0001
spls = dure_simu/time_step


# In[514]:


deb = 100
dur = 2
beg = int(deb//time_step)
end = beg+int(dur//time_step)


# In[515]:


num_mpi_ranks = 4 # the number of sims you used in parallel
datadir = os.path.expanduser("/mnt/data1/data_paul/different_Hz/sim_less_stim_neurons_nocons_corrected_2dif_10h45") # Set this to your data path
prefix = "rf1"

nb_neurons = 4096


# In[516]:


#datadir_sigal = os.path.expanduser("~/data/sim_network/sim_10Hz_cons_4h_1pat_mrco_5_demonstration") # Set this to your data path
datadir_sigal = datadir


# In[517]:


all_times = np.genfromtxt(datadir_sigal+'/pattern_times', delimiter=' ')
nb_pattern = len(set(all_times[:,1]))
signals = np.zeros((nb_pattern,int(dure_simu//time_step)+1))
signals_times = [[] for i in range(nb_pattern)]
for time,kind in all_times:
    signals_times[int(kind)].append(time)
    signals[int(kind),int((time)//time_step):int((time*auryn_tstep+(duree_pattern))//time_step)]=1
signals_times = np.array(signals_times)


# # Find low rank structure in spiking activity

# In[518]:


spkfiles  = ["%s/%s.%i.e.spk"%(datadir,prefix,i) for i in range(0,num_mpi_ranks)]
sfo = AurynBinarySpikeView(spkfiles)


# In[519]:


rateE  = np.mean([pd.read_csv("%s/%s.%i.e.prate"%(datadir,prefix,2),delimiter=' ').values for i in range(num_mpi_ranks)],axis=0)


# In[520]:


time_axis = np.linspace(0,dure_simu,int(dure_simu//time_step))
rateE= rateE[:,1]


# In[521]:


fig = go.Figure(data=go.Scatter(x=time_axis, y=rateE))
fig.show()


# In[522]:


# plt.plot(time_axis[beg:end],rateE[beg:end])
# some_signals = signals_times[0][ (signals_times[0]<(deb+dur) ) & (signals_times[0]>deb)]
# for sig in some_signals:
#     plt.axvspan(sig, sig+duree_pattern, facecolor='red', alpha=0.25)


# In[523]:


# some_signals = signals_times[0][ (signals_times[0]<(deb+dur) ) & (signals_times[0]>deb)]


# In[491]:


# plt.plot(rateE[beg:end])
# plt.plot(signals[0][beg:end]*10)


# In[492]:


# rateI  = np.mean([pd.read_csv("%s/%s.%i.i2.prate"%(datadir,prefix,i),delimiter=' ' ) for i in range(num_mpi_ranks)],axis=0)
# time_axis_I = rateI[:,0]
# rateI= rateI[:,1]


# In[493]:


# plt.plot(time_axis_I,rateI)


# In[494]:


# plt.plot(time_axis[beg:end],rateI[beg:end])
# plt.plot(time_axis[beg:end],signals[0][beg:end]*10)


# In[495]:


# win = signal.windows.hann(1000)
# plt.plot(np.convolve(rateE,win)/ sum(win))


# In[496]:


# win = signal.windows.hann(1000)
# plt.plot(np.convolve(rateI,win)/ sum(win))


# ## Raster Plot illustrant la reaction du system

# In[497]:


def plot_profile(begin,end,title,nb_sample,nb_sample_ext,alpha = 1,signal = 0,more_than = 1000):

    tm_rast = begin
    time_range_rast = end
    beg_2 = int(tm_rast//time_step)
    end_2 = beg_2+int(time_range_rast//time_step)
    
    spikes_rast = np.array(sfo.get_spikes(t_start=tm_rast,t_stop=tm_rast+time_range_rast))

    ListTrains = [[] for _ in range(nb_neurons)] 
    for s in spikes_rast:
        ListTrains[int(s[1])].append(s[0])

    spikes_rast  = np.array([li for li in spikes_rast if len(ListTrains[int(li[1])])<more_than])
    np.random.seed(0)
    sample = np.random.choice(list(range(nb_neurons)),nb_sample)
    spikes_rast_sample = np.array([i for i in spikes_rast if i[1] in sample])

    spikes_rast_ext = np.array(sfo_ext.get_spikes(t_start=tm_rast,t_stop=tm_rast+time_range_rast))

    sample_ext = np.random.choice(list(range(nb_neurons)),nb_sample_ext)
    spikes_rast_sample_ext =np.array([i for i in spikes_rast_ext if i[1] in sample_ext])
    
    new_indexes = dict(zip(list(set(spikes_rast_sample[:,1])),list(range(nb_sample))))
    
    new_indexes_ext = dict(zip(list(set(spikes_rast_sample_ext[:,1])),list(range(nb_sample_ext))))
    
    colors = in_pattern(spikes_rast_sample_ext,duree_pattern, signals_times[0])
    colors = ["r" if i ==1 else "b" for i in colors]
    
    fig, axs = plt.subplots(3,figsize=(8,10),sharex=True,gridspec_kw={'height_ratios': [2,2, 1]})
    fig.suptitle(title,y = 0.92,fontsize = 15)
    
    axs[0].scatter(spikes_rast_sample_ext[:,0], 
                   [new_indexes_ext[i] for i in spikes_rast_sample_ext[:,1]],alpha = alpha, c =colors,s = 18)
    axs[0].set_ylabel("input neuron index",fontsize = 12)

    axs[1].scatter(spikes_rast_sample[:,0], 
                   [new_indexes[i] for i in spikes_rast_sample[:,1]],alpha = alpha, c="b",s = 18)
    axs[1].set_ylabel("output neuron index",fontsize = 12)

    some_signals = signals_times[0][ (signals_times[0]<(tm_rast+time_range_rast) ) & (signals_times[0]>tm_rast)]

    axs[2].plot(time_axis[beg_2:end_2],rateE[beg_2:end_2])

    for sig in some_signals:
        axs[2].axvspan(sig, sig+duree_pattern, facecolor='red', alpha=0.25)
    
    axs[2].set_ylabel("rate (Hz)",fontsize = 12)
    
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("Time (s)",fontsize = 12)
    #plt.ylabel("common Y",x = 0)
    


# In[498]:


# plot_profile(110.4,1.1,"Before learning",1500,150,more_than = 20)


# In[499]:


# plot_profile(14385,1.1,"After learning",1500,150,more_than = 20)


# In[502]:


number_iter = 2
time_range = 100
nb_signal = 1
size_window = 0.2
procces_number = 10

all_spikes_in_time,all_dist_in_time,all_times_in_time,all_data_in_time,times = parallelize(procces_number,number_iter,time_range,nb_signal,size_window,sfo,dure_simu,signals_times,nb_neurons,duree_pattern)


# In[503]:


bin_size = 1e-3


# In[505]:


which_signal=0
spikes_in_time = all_spikes_in_time[which_signal]
dist_in_time = all_dist_in_time[which_signal]
times_in_time = all_times_in_time[which_signal]
data_in_time = all_data_in_time[which_signal]


# In[509]:


times


# In[510]:


# plt.hist(dists1,bins=100,alpha=0.5);
H, bins = np.histogram(dist_in_time[times[0]],bins=150);
plt.bar(bins[:-1],H,width=(1/(len(bins)*1.9)))
plt.show()


# In[35]:


moyenne_plus = np.mean(H) + 1*np.std(H)
delimiters = []
moyenne = np.mean(H)

for bn in range(len(H)):
    
    value = H[bn]
    
    if value > moyenne_plus:

        left = bn
        while H[left]<= H[left+1] or H[left]>moyenne:
            left-=1
            
        right = bn
        while H[right]<= H[right-1] or H[right]>moyenne:
            right+=1
    
        delimiters.append((left,right))
            
        
    


# In[36]:


#delimiters =[(50,75),(25,30)]


# In[37]:


#delimiters


# In[38]:


plt.bar(bins[:-1],H,width=(1/(len(bins)*2)))
plt.plot([np.min(bins),np.max(bins)],[moyenne_plus,moyenne_plus])


# In[39]:


pics_bins = np.array(list(set(delimiters)))


# In[40]:


pics_bins


# In[41]:


pics_bins_changed = np.array([pics_bins[:,0],pics_bins[:,1]]).T


# In[42]:


times_pics = bins[:-1][pics_bins_changed]


# In[43]:


plt.bar(bins[:-1],H,width=(1/(len(bins)*2)))
higher=np.max(H)

for i in times_pics.ravel():
    plt.plot([i,i],[0,higher],c="red")


# In[44]:


#show_hist_times(dist_in_time,times)


# In[45]:


spikes =  spikes_in_time[times[-1]]


# In[46]:


ListTrains = [[] for _ in range(nb_neurons)] 
for s in spikes:
    ListTrains[int(s[1])].append(s[0])


# In[47]:


distancesN = data_in_time[times[-1]]


# In[48]:


scores = np.array([ np.sum(np.logical_and(np.array(x)[:,0]<duree_pattern,np.array(x)[:,0]>0))/len(x) if len(x)>0 else 0 for x in distancesN])


# In[49]:


proportion_in_time_signals = dict()
proportion_in_time = dict()
for i in range(nb_signal):
    print(i)
    which_signal=i
    spikes_in_time = all_spikes_in_time[which_signal]
    dist_in_time = all_dist_in_time[which_signal]
    times_in_time = all_times_in_time[which_signal]
    data_in_time = all_data_in_time[which_signal]

    proportion_in_time = dict()
    for t in times:
        #print(t)
        proportion_in_time[t]=tools.in_pattern_proportion_neurons(spikes_in_time[t],duree_pattern,np.array(signals_times[which_signal]),nb_neurons)
    proportion_in_time_signals[i]=proportion_in_time


# In[50]:


#show_hist_times(proportion_in_time,times,log=True)


# In[51]:


proportion_in_time_signals_arrays = dict()
for k in proportion_in_time_signals:
    proportion_in_time_signals_arrays[k] = np.array(list(proportion_in_time_signals[k].values()))


# In[52]:


plt.plot(times,np.mean(proportion_in_time_signals_arrays[0],axis=1),label="pattern 1")
plt.plot(times,np.mean(proportion_in_time_signals_arrays[1],axis=1),label="pattern 2")
# plt.plot(times,np.mean(proportion_in_time_signals_arrays[2],axis=1),label="pattern 1")
# plt.plot(times,np.mean(proportion_in_time_signals_arrays[3],axis=1),label="pattern 2")
plt.legend()


# In[56]:


plt.plot(times,np.max(proportion_in_time_signals_arrays[0],axis=1),label="pattern 1")
plt.plot(times,np.max(proportion_in_time_signals_arrays[1],axis=1),label="pattern 2")
# plt.plot(times,np.max(proportion_in_time_signals_arrays[2],axis=1),label="pattern 1")
# plt.plot(times,np.max(proportion_in_time_signals_arrays[3],axis=1),label="pattern 2")
plt.legend()


# In[54]:


# def score_in_time():
#     score_in_t = [[],[]]
#     for t in times:
#         distancesN =data_in_time[t]
#         scores = np.array([ np.sum(np.logical_and(np.array(x)[:,0]<duree_pattern,np.array(x)[:,0]>0))/len(x) if len(x)>0 else 0 for x in distancesN])
#         score_in_t[0].append(np.mean(scores))
#         score_in_t[1].append(t)
#     return score_in_t


# In[55]:


# sc_in_tm = score_in_time();


# In[ ]:


# plt.plot(sc_in_tm[1],sc_in_tm[0])


# ### Une population code bien le signal

# In[ ]:


#plt.hist([len(n) for n in distancesN],bins=100);


# In[ ]:


#plt.hist([len(n) for n in distancesN if len(n)<500],bins=150);


# In[281]:


plt.hist(scores,bins=50);


# ### Un neuron parfait ?

# In[ ]:


the_one = np.argmax(scores)


# ### Et qui fire assez souvent. Si sont firing rate était à la ramasse, on aurait pu penser 
# ### qu'il se serait fait attribué un bon score par accident.

# In[ ]:


distTrainOne = np.array(distancesN[the_one])


# In[ ]:


plt.scatter(distTrainOne[:,1],distTrainOne[:,0])


# ### Des bons neurones 

# In[312]:


ratio = 0.05


# In[330]:


#the_ones = scores.argsort()[-int((len(scores)*ratio)):][::-1]
the_ones = np.where(scores>0.90)[0]
print(len(the_ones))
#the_ones = np.random.choice(list(range(0,4096)),905)


# In[331]:


len(the_ones)


# In[332]:


# dists_ones = np.array([[dist[0],dist[1]] for  n in the_ones for dist in distancesN[n]])
# dists_ones= dists_ones[np.argsort(dists_ones[:,1])]
#times_ones = np.sort([time[1] for  n in the_ones for time in distancesN[n]])


# In[333]:


dists_ones_n =dict()
dists_ones = dict()


# In[334]:


for n in the_ones:
    dists_ones_n[n]=dict()
    
for t in times:
    dists_ones[t]= np.array([])
    for n in the_ones:
        dat = np.array(data_in_time[t][n])[:,0]
        dists_ones_n[n][t]=dat
        dists_ones[t] = np.append(dists_ones[t],dat)


# Le Taux de décharge initial des neurones codant le mieux au pattern est très bas !

# In[335]:


len(dists_ones[times[0]])


# In[320]:


#show_hist_times(dists_ones,times)


# In[275]:


# dist_in_time_without = dict()
# for k,v in data_in_time.items():
#     print(k)
#     dist_in_time_without[k] = np.array([])
#     for n in range(len(v)):
#         if n not in the_ones:
#             dist_in_time_without[k]=np.append(dist_in_time_without[k],np.array(v[n])[:,0])


# In[276]:


#show_hist_times(dist_in_time_without,times)


# ### Hist of pics

# In[ ]:


which_signal=1
spikes_in_time = all_spikes_in_time[which_signal]
dist_in_time = all_dist_in_time[which_signal]
times_in_time = all_times_in_time[which_signal]
data_in_time = all_data_in_time[which_signal]


# In[ ]:


Htot, bins = np.histogram(dist_in_time[times[-1]],bins=250);


# In[ ]:


moyenne_plus = np.mean(Htot) + 1*np.std(Htot)
delimiters = []
moyenne = np.mean(Htot)

for bn in range(len(Htot)):
    
    value = Htot[bn]

    if value > moyenne_plus:

        left = bn
        while not left<0 and (Htot[left]<= Htot[left+1] or Htot[left]>moyenne_plus) :
            left-=1
            
        right = bn
        while not right>=len(Htot) and ( Htot[right]<= Htot[right-1] or Htot[right]>moyenne_plus):
            right+=1
    
        delimiters.append((left,right))
            
        


# In[ ]:


plt.bar(bins[:-1],Htot,width=(1/(len(bins)*2)))
plt.plot([np.min(bins),np.max(bins)],[moyenne_plus,moyenne_plus])


# In[ ]:


pics_bins = np.array(list(set(delimiters)))
pics_bins_changed = np.array([pics_bins[:,0],pics_bins[:,1]]).T
times_pics = bins[:-1][pics_bins_changed]


# In[ ]:


pics_bins_changed


# In[442]:


distancesN
distancesN = data_in_time[times[-1]]
scores_pics = []
for tms in times_pics:
    scores_pics.append(np.array([ np.sum(np.logical_and(np.array(x)[:,0]<tms[1],np.array(x)[:,0]>tms[0]))/len(x) if len(x)>0 else 0 for x in distancesN]))


# In[443]:


percentage_pic = []
for tms in times_pics:
    percentage_pic.append((tms[1]-tms[0])/size_window)


# In[444]:


in_pics = []
for i in range(len(percentage_pic)):
    perc = percentage_pic[i]
    pic_ones = []
    for j in range(len(scores_pics[i])):
        score = scores_pics[i][j]
        if score > perc*2: # CHANGED
            pic_ones.append(j)
    in_pics.append(pic_ones)
    


# In[445]:


def take_spikes_pics(distancesN):
    spikes_in_pics = []
    for pic_ones in in_pics:
        spikes_in_pics.append(np.array([]))
        for one in pic_ones :
            train = np.array(distancesN[one])[:,0]
            spikes_in_pics[-1]=np.append(spikes_in_pics[-1],train)
    return spikes_in_pics
spikes_in_pics=take_spikes_pics(distancesN)


# In[446]:


in_pics_1 = in_pics


# In[447]:


len(in_pics_1[0])


# In[448]:


in_pics_1 = [ i for j in in_pics_1 for i in j ]
set_pics1 = set(in_pics_1)


# In[449]:


in_pics_0 = [ i for j in in_pics_0 for i in j ]
set_pics0 = set(in_pics_0)


# In[452]:


len(set_pics0.union(set_pics1))


# In[ ]:


for s1 in in_pics_1:
    for s0 in in_pics_0:
        print(s1,s0)


# In[263]:


which_pic=6
H, bins = np.histogram(spikes_in_pics[which_pic],bins=250);

plt.bar(bins[:-1],H,width=(1/(len(bins)*2)))


# In[264]:


#Htot, _ = np.histogram(dist_in_time[times[-1]],bins=250);
plt.bar(bins[:-1],Htot,width=(1/(len(bins)*2)))


# In[265]:


#Htot, _ = np.histogram(dist_in_time[times[-1]],bins=250);
H_removed = Htot-H
plt.bar(bins[:-1],H_removed,width=(1/(len(bins)*2)))


# In[257]:


somme=0
somme_hist = np.zeros(250)
for which_pic in range(len(percentage_pic)):
    H, bins = np.histogram(spikes_in_pics[which_pic],bins=250);
    somme += np.sum(H)
    somme_hist += H
    plt.bar(bins[:-1],H,width=(1/(len(bins)*3)),alpha=0.7)


# In[233]:


plt.bar(bins[:-1],Htot-somme_hist,width=(1/(len(bins)*3)))


# In[ ]:


plt.bar(bins[:-1],H,width=(1/(len(bins)*3)),alpha=0.7)


# ### percentage of total activity

# In[107]:


print("Spikes represent "+str((somme/np.sum(Htot))*100)+"% of the activity")


# In[85]:


for t in times[len(times)::-9]:
    ditance_neurons = data_in_time[t]
    spikes_in_pics=take_spikes_pics(ditance_neurons)
    for which_pic in range(len(percentage_pic)):
        H, bins = np.histogram(spikes_in_pics[which_pic],bins=250);
        plt.bar(bins[:-1],H,width=(1/(len(bins)*3)))
    plt.show()


# In[228]:


def mean_std_neu(which_signal):
    mean_std_neurons = []
    std_neurones =dict()
    for t in times:
        #print(t)
        std_neurones[t] = []
        for n in range(nb_neurons):
            std_neurones[t].append(np.nanstd(all_data_in_time[which_signal][t][n][:,0]))
        #print(std_neurones)
        mean_std_neurons.append(np.nanmean(std_neurones[t]))
    return mean_std_neurons,std_neurones


# In[229]:


# def overlap_in_time(which_signal):
#     mean_std_neurons = []
#     for t in times:
#         print(t)
#         std_neurones = []
#         for n in range(nb_neurons):
#             val = pd.Series(all_data_in_time[which_signal][t][n][:,0])
#             std_neurones.append(val.mad)
#         mean_std_neurons.append(np.nanmean(std_neurones))
#     return mean_std_neurons


# In[230]:


std_pat_1,std_neurones_1 = mean_std_neu(0)
std_pat_2,std_neurones_2 = mean_std_neu(1)
#std_pat_3,std_neurones_3 = mean_std_neu(2)
#std_pat_4,std_neurones_4 = mean_std_neu(3)


# In[231]:


plt.plot(times,std_pat_1,label = "pattern 1")
plt.plot(times,std_pat_2,label = "pattern 2")
# plt.plot(times,std_pat_3,label = "pattern 3")
# plt.plot(times,std_pat_4,label = "pattern 4")
plt.legend()
plt.title("mean std in time of each neurones")


# In[232]:


plt.hist(std_neurones_1[times[-1]],bins=50,alpha= 0.7);
plt.hist(std_neurones_1[times[0]],bins=50,alpha= 0.7);


# ### Le pic à 0 n'est pas issu d'une assemblée mais d'une sensibilisation globale du réseau 

# ### Le nombre de neurones du réseau est divisé par le nombre d'assemblées
# 

# In[234]:


np.mean([len(i) for i in in_pics])*(len(spikes_in_pics)-1)


# In[235]:


np.mean([len(i) for i in in_pics])


# In[315]:


last_signals = signals_times[which_signal][-1000:]
all_data = []
for event_i in range(len(last_signals)):
    pat_start = last_signals[event_i]
    all_data.append(sfo.get_spikes(pat_start-(duree_pattern/2),pat_start+duree_pattern+(duree_pattern/2)))
    
len_datas = np.array([len(i) for i in all_data])
len_in_pics = np.array([len(i) for i in in_pics])

np_in_pics = np.empty((len(in_pics),np.max(len_in_pics)))
np_all_data = np.empty((len(last_signals),np.max(len_datas),2))

for p in range(len(in_pics)):
    np_in_pics[p][:len_in_pics[p]]=in_pics[p]

for d in range(len(last_signals)):
    np_all_data[d,:len_datas[d]] = np.array(all_data[d])

def function():
    all_std = [[] for _ in range(len(np_in_pics))]
    for event_i in range(len(np_all_data)):
        #print(event_i)
        data = np_all_data[event_i][:len_datas[event_i]]
        pics_spikes =[]
        for i_pics in range(len(np_in_pics)):
            neurones = np_in_pics[i_pics][:len_in_pics[i_pics]]
            #times_pic = times_pics[i_pics]
            is_in = np.isin(data[:,1],neurones)
            #print(is_in)
            pic_spikes = []
            for d in range(len(data)) :
                if is_in[d]:
                    pic_spikes.append(data[d][0])
                    
            all_std[i_pics].append(np.std(pic_spikes))
    return all_std

    


# In[316]:


all_std = function()


# ### La STD de toutes les spikes sur un grand nombre de pattern est equivalente à la STD moyenne sur un grand nombre de patterns 

# In[317]:


np.array([np.std(i) for i in spikes_in_pics])


# In[318]:


np.mean(all_std,axis=1)


# In[337]:


which_signal=1
spikes_in_time = all_spikes_in_time[which_signal]
dist_in_time = all_dist_in_time[which_signal]
times_in_time = all_times_in_time[which_signal]
data_in_time = all_data_in_time[which_signal]def SpikesEvents(spikeTrainN, duree_pattern, signal_times):
    
    spikeTrain = spikeTrainN[:,0]
    neurones = spikeTrainN[:,1]
    offset = duree_pattern/2
    
    event_sequence = [[] for _ in range(len(signal_times))] 
    
    r = 0
    for event_i in range(len(signal_times)):
        event = signal_times[event_i]
        for spike_i in range(r, len(spikeTrain)):
            if spikeTrain[spike_i] > (event+duree_pattern/2)+offset:
                break
            if spikeTrain[spike_i] < (event-duree_pattern/2)+offset:
                r += 1
            else:
                # print([neurones[spike_i],spikeTrain[spike_i]])
                # print(event_i)
                event_sequence[event_i].append([neurones[spike_i],spikeTrain[spike_i]])
                #print(event_sequence[event_i])

    good_events=[event_sequence[i] for i in range(len(event_sequence)) if signal_times[i]>spikeTrain[0] and signal_times[i]<spikeTrain[-1] ]
    return good_events

    # def proba_sequence(big_sequence,fro=0.0008,to=0.0005):
    #     size = len(set(big_sequence))
    #     trans_mat = np.zeros(())
    #     for spk_i in range(len(big_sequence)):
    #         for spkf_i in range(len(big_sequence)):
    #             if big_sequence[spkf_i][1]>big_sequence[spk_i][1]+fro+to:
    #                 break
    #             if big_sequence[spkf_i][1]-big_sequence[spk_i][1] >fro:


# In[14]:


event_sequence = SpikesEvents(spikes_in_time[times[-1]],duree_pattern,signals_times[which_signal])


# In[19]:


event_sequence1 = event_sequence


# In[30]:


Nevent_train = []
for event_train in event_sequence1:
    ListTrains1 = [[] for _ in range(nb_neurons)] 
    
    for s in event_train:
        ListTrains1[int(s[0])].append(s[1]-event_train[0][1])
    Nevent_train.append(ListTrains1)


# In[310]:


big_sequence=np.concatenate(event_sequence)


# In[130]:


# plt.style.use('seaborn-white')
# for k in dists_ones:
#     plt.hist(dists_ones[k],bins=100, ec="k",histtype='stepfilled',alpha=0.3);


# In[529]:


#random_one = np.random.choice(the_ones)


# In[530]:


#show_hist_times(dists_ones_n[random_one],times,bins = 100)


# In[133]:


# distrib_nb_spk = []
# for i in the_ones :
#     distrib_nb_spk.append(((len(np.array(distancesN[i])[:,0])))/(200*3))


# In[134]:


# plt.hist(distrib_nb_spk,bins = 50);


# In[46]:


# ListTrainsNeoTheOnes = []
# min1 = 1000000000
# max1 = 0
# for train in the_ones:
#     min2=min(ListTrains[train])
#     max2=max(ListTrains[train])
#     if min2 < min1:
#         min1=min2 
#     if max2 > max1:
#         max1=max2 

# for train in the_ones:
#  ListTrainsNeoTheOnes.append(SpikeTrain(ListTrains[train]*pq.s,t_start=min1,t_stop=max1))


# In[49]:


# trains = []
# for i in the_ones:
#     trains.append(ListTrains[i])
# trains = np.sort(np.array([j for i in trains for j in i]))
# lelu = SpikeTrain(trains*pq.s,t_start=min(trains),t_stop=max(trains))
# signi_train =elephant.statistics.instantaneous_rate(lelu,1*pq.ms)


# In[50]:


# plt.plot(signi_train.times,signi_train/len(the_ones))


# # Load and analyze EE weights

# In[336]:


wmatfiles  = ["%s/rf1.%i.ee.wmat"%(datadir,i) for i in range(num_mpi_ranks)]
w = np.sum( [ mmread(wf) for wf in wmatfiles ] )


# In[337]:


h1 = plt.hist(w.data, bins=100, log=True)
plt.title("EE weight distribution")
sns.despine()


# In[338]:


net = Network(notebook=True)
mean_weight = np.mean(w.data)


# In[339]:


# for n in the_ones:
#     net.add_node(int(n))


# In[340]:


#@numba.jit(nopython=True)
def connectivity(neurones):
    connectivity_pat =[]
    bon_poids=[]
    for n in neurones:
        for n2 in neurones:
            if w[n,n2]>0:
#                 if w[n,n2]>mean_weight:
#                     bon_poids.append(w[n,n2])
                    #net.add_edge(int(n), int(n2), weight=float(w[n,n2]))
                connectivity_pat.append([n,n2,w[n,n2]])
    return np.array(connectivity_pat)

def connectivity_two(neurones1,neurones2):
    connectivity_pat =[]
    bon_poids=[]
    for n in neurones1:
        for n2 in neurones2:
            if w[n,n2]>0:
#                 if w[n,n2]>mean_weight:
#                     bon_poids.append(w[n,n2])
#                     #net.add_edge(int(n), int(n2), weight=float(w[n,n2]))
                connectivity_pat.append([n,n2,w[n,n2]])
    return np.array(connectivity_pat)        


# In[341]:


connectivity_pat = connectivity_two(the_ones,the_ones)


# In[342]:


print("Mean excitatory weight of "+str(np.mean(connectivity_pat[:,2]))+" for coding neurons vs "+str(mean_weight)+" for all neurons")


# In[343]:


plt.hist(connectivity_pat[:,2],bins=50,log=True);


# In[344]:


connectivity_pat[:,2]


# In[375]:


highly_connected =connectivity_pat[np.where(connectivity_pat[:,2]>mean_weight)[0]]


# In[379]:


h_ones=set(highly_connected[:,1]).union(set(highly_connected[:,0]))


# In[380]:


hdists_ones_n =dict()
hdists_ones = dict()
for n in h_ones:
    hdists_ones_n[n]=dict()
    
for t in times:
    dists_ones[t]= np.array([])
    for n in h_ones:
        dat = np.array(data_in_time[t][int(n)])[:,0]
        hdists_ones_n[int(n)][t]=dat
        hdists_ones[t] = np.append(dists_ones[t],dat)


# In[381]:


show_hist_times(hdists_ones,times)


# In[212]:


#net.show("mynet.html")


# In[213]:


# def max_path(net):
#     paths = [[] for i in range(len(nodes))]
#     in_paths = [set() for i in range(len(nodes))]
#     traited = []
#     for n in nodes:
#         still_nodes = True
#         while still_nodes:
#             possible_nodes = linked_nodes-in_paths[n]
#             if len(possible_nodes)>0:
#                 next_node=max(possible_nodes)
#                 paths[n].append(next_node)
#                 in_paths[n].add(next_node)
#             else:
#                 still_nodes = False
#         traited.append(n)


# In[134]:


net_dic = net.get_adj_list()


# In[135]:


nodes = set(net_dic.keys())


# In[170]:


paths = {i:[i] for i in nodes}
in_paths = {i:set([i]) for i in nodes}
traited = []

for n in nodes:
    still_nodes = True
    while still_nodes:
        possible_nodes = list(net_dic[n]-in_paths[n])
        if len(possible_nodes)>0:
            next_node = possible_nodes[np.argmax([w[n,pn] for pn in possible_nodes])]
            #print([w[n,pn] for pn in possible_nodes])
            if next_node in traited and  w[paths[n][-1],next_node] > mean_weight:
                paths[n] = paths[n]+paths[next_node]
                in_paths[n]=in_paths[n].union(in_paths[next_node])
            elif w[paths[n][-1],next_node] > mean_weight :
                paths[n].append(next_node)
                in_paths[n].add(next_node)
                still_nodes= False
            else:
                still_nodes= False
        else:
            still_nodes = False
    traited.append(n)


# In[171]:


#  taille = 0
#  for i in paths:
#      if len(paths[i])>taille:
#          taille = len(paths[i])
#          largest_path = paths[i]


# In[192]:


strength = 0
for i in paths :
    chemin =paths[i]
    strength2 = np.sum([w[chemin[i],chemin[i+1]] for i in range(len(chemin)-1)])
    if strength2 > strength:
        strength = strength
        strongest_path = i
strongest_path = paths[strongest_path]


# In[193]:


strongest_path


# In[195]:


weights = []
for n in range(len(strongest_path)-1):
    weights.append(w[strongest_path[n],strongest_path[n+1]])


# In[196]:


weights


# ### Des neurones très connectés avec une activité liée dans le temps 

# In[197]:


distTrainPost = np.array(distancesN[int(strongest_path[0])])
distTrainPre = np.array(distancesN[int(strongest_path[1])])
distTrainPre2 = np.array(distancesN[int(strongest_path[2])])


# In[102]:


plt.hist(distTrainPost[:,0],bins=50,alpha = 0.7);
plt.hist(distTrainPre[:,0],bins=50,alpha = 0.7);
plt.hist(distTrainPre2[:,0],bins=50,alpha = 0.7);


# In[669]:


most_connected_arg = connectivity_pat[connectivity_pat[:,2].argsort()[-50:][::-1]]
most_connected= most_connected_arg[:,[0,1]].ravel().astype(int)
scores_mst =scores[most_connected]


# In[670]:


net = Network(notebook=True)
net.add_nodes(most_connected)
net.add_edges(most_connected_arg)


# In[703]:


#net.show("my_network.html")


# In[ ]:




