import numpy as np
from itertools import accumulate
from neo.core import AnalogSignal
import quantities as pq
import matplotlib.pyplot as plt
from elephant.spike_train_generation import inhomogeneous_poisson_process,homogeneous_poisson_process
from sys import argv
from neo.core import AnalogSignal
from timeit import default_timer as timer
import numba

@numba.jit(nopython=True)
def outside_pattern(spiketrain,new_spiketrain,pattern_times):
    actual_pattern = 0
    actual_spike = 0
    for i in range(len(spiketrain)):
        if actual_pattern<len(pattern_times)-1:
            if spiketrain[i]>pattern_times[actual_pattern+1]:
                actual_pattern+=1
        if spiketrain[i]>pattern_times[actual_pattern]+pattern_size:
            new_spiketrain[actual_spike] = spiketrain[i]
            actual_spike+=1
    return actual_pattern,actual_spike

@numba.jit(nopython=True)
def pattern_placement(actual_spike,pattern_times,new_spiketrain,nb_pattern,all_motifs,motifs_sizes,n):
    total_size = actual_spike
    for start_pattern in pattern_times:
        which_pattern= np.random.randint(nb_pattern)
        new_spiketrain[total_size:total_size + motifs_sizes[n,which_pattern]] = all_motifs[n,which_pattern][:motifs_sizes[n,which_pattern]]+start_pattern
        total_size += motifs_sizes[n,which_pattern]
    return total_size


@numba.jit(nopython=True)
def copy_data(datas,fill_until,total_size,new_spiketrain,n):
        datas[fill_until:fill_until+total_size,0] = new_spiketrain[:total_size]
        datas[fill_until:fill_until+total_size,1] = np.full(total_size,n)
        return fill_until+total_size

import argparse

parser = argparse.ArgumentParser(description='Generate spike trains')
parser.add_argument('-inhomogeneous',action='store_true')
parser.add_argument('-rate',type=int,required=True)
parser.add_argument('-minrate',type=float,required=False)
parser.add_argument('-maxrate',type=float,required=False)
parser.add_argument('-speedchange',type=float,required=False)
parser.add_argument('-timesim',type=float,required=True)
parser.add_argument('-samplingvar',type=float,required=False)
parser.add_argument('-nbneurons',type=int,required=True)
parser.add_argument('-nbsegment',type=int,required=True)
parser.add_argument('-outdir',type=str,required=True)
parser.add_argument('-pattern',action='store_true')
parser.add_argument('-nbpattern',type=int,required=False)
parser.add_argument('-patternsize',type=float,required=False)
parser.add_argument('-patternfrequency',type=int,required=False)
# Si il y a un pattern

args = parser.parse_args()

inhomogeneous = args.inhomogeneous
mean_rate = args.rate
min_rate = args.minrate
max_rate = args.maxrate
speed_change = args.speedchange
time_sim = args.timesim
sampling_var = args.samplingvar
nb_neurons = args.nbneurons
nb_segment = args.nbsegment
outdir = args.outdir
pattern = args.pattern

if pattern :
    nb_pattern = args.nbpattern
    pattern_size = args.patternsize
    pattern_frequency = args.patternfrequency
    new_spiketrain = np.empty( int(((mean_rate*time_sim)/nb_segment)*2) ,dtype=np.float32) #change size for longer simulations
    all_motifs = np.empty((nb_neurons,nb_pattern,int(((mean_rate*pattern_size))*10)))
    motifs_sizes = np.empty((nb_neurons,nb_pattern),dtype=int)

# print("inhomogeneous: "+str(inhomogeneous))
# print("mean_rate: "+str(mean_rate))
# print("min_rate: "+str(min_rate))
# print("max_rate: "+str(max_rate))
# print("speed_change: "+str(speed_change))
# print("time_sim: "+str(time_sim))
# print("sampling_var: "+str(sampling_var))
# print("nb_neurons: "+str(nb_neurons))
# print("nb_segment: "+str(nb_segment))
# print("outdir: "+str(outdir))
# print("pattern: "+str(pattern))
# print("nb_pattern: "+str(nb_pattern))
# print("pattern_size: "+str(pattern_size))
# print("pattern_frequency: "+str(pattern_frequency))


datas = np.empty((int(((mean_rate*time_sim*nb_neurons)/nb_segment)*2),2),dtype=np.float32)

if inhomogeneous:
    rate_var = lambda n,spd : ((np.random.rand(n)*2)-1)*spd
    rates = lambda n,mn_rate,spd,minv,maxv : np.array(list(accumulate(rate_var(n,spd),lambda i,j : min(maxv,max(minv,i+j)),initial=mn_rate)))


    rates_over_time = []

    for i in range(nb_neurons):
        rates_over_time.append(rates(int(time_sim*sampling_var),mean_rate,speed_change,min_rate,max_rate))


time_seg =  (time_sim/nb_segment)*pq.s

mesure_time_0 = 0
mesure_time_1 = 0
mesure_time_2 = 0
mesure_time_3 = 0
mesure_time_4 = 0

for s in range(nb_segment):
    start = time_seg*s

    if pattern:
        pattern_times = homogeneous_poisson_process(pattern_frequency*pq.Hz,t_start=start,t_stop =time_seg*(s+1),refractory_period = pattern_size*pq.s, as_array=True )

    #datas = np.empty((int(((mean_rate*time_sim*nb_neurons)/nb_segment)*1.5),2),dtype=np.float32) #change size for longer simulations
    fill_until = 0

    for n in range(nb_neurons):
        print(n)
        start_timer = timer()
        if inhomogeneous:
            sign = AnalogSignal(rates_over_time[n][int(start*sampling_var):int((start+time_seg)*sampling_var)],units = 'Hz',sampling_rate = sampling_var*(pq.Hz),t_start=start,t_stop=start+time_seg)
            spiketrain = inhomogeneous_poisson_process(sign,as_array=True)
        else:
            spiketrain = homogeneous_poisson_process(mean_rate*pq.Hz,t_start=start,t_stop=start+time_seg,as_array=True)

        end = timer()
        mesure_time_0 += end-start_timer
        #print("taille spike train original "+str(spiketrain.shape))

        if pattern and len(spiketrain)>0:

            if s == 0 :
                for i in range(nb_pattern):
                    motif_start = np.random.uniform(0,time_seg)
                    motif = spiketrain[np.where( (motif_start<spiketrain)&(motif_start+pattern_size>spiketrain))[0]]
                    if len(motif)>0:
                        motif = motif-motif_start
                    motifs_sizes[n,i] = len(motif)
                    all_motifs[n,i,:len(motif)] = motif

            start_timer = timer()
            actual_pattern , actual_spike = outside_pattern(spiketrain,new_spiketrain,pattern_times)
            end = timer()
            mesure_time_1 += end-start_timer

            start_timer = timer()
            total_size = pattern_placement(actual_spike,pattern_times,new_spiketrain,nb_pattern,all_motifs,motifs_sizes,n)
            end = timer()
            mesure_time_2 += end-start_timer
            
            start_timer = timer()
            fill_until = copy_data(datas,fill_until,total_size,new_spiketrain,n)
            end = timer()
            mesure_time_3 += end-start_timer

        else:
            
            datas[fill_until:fill_until+len(spiketrain),0] = spiketrain
            datas[fill_until:fill_until+len(spiketrain),1] = np.full(len(spiketrain),n)
            fill_until = fill_until+len(spiketrain)

    start_timer = timer()
    print("sorting...")
    fill_datas = datas[:fill_until]
    fill_datas = fill_datas[fill_datas[:,0].argsort()]
    print("saving...")

    with open(outdir+"/spiketrains_"+str(s),'w') as f:
        fmt = '%.4f %d'
        fmt = '\n'.join([fmt]*fill_datas.shape[0])
        data = fmt % tuple(fill_datas.ravel())
        f.write(data)

    #np.savetxt(outdir+"/spiketrains_"+str(s),fill_datas[fill_datas[:,0].argsort()] )
    end = timer()
    mesure_time_4 += end-start_timer

    print("0_chunk "+str(mesure_time_0))
    print("first_chunk "+str(mesure_time_1))
    print("second_chunk "+str(mesure_time_2))
    print("thirst_chunk "+str(mesure_time_3))
    print("forth_chunk "+str(mesure_time_4))
