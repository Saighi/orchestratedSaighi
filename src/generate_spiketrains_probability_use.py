#%%
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
from sys import argv
from timeit import default_timer as timer
from multiprocessing import Pool, TimeoutError
import numba
from scipy.signal.windows import gaussian
from tick.hawkes import SimuInhomogeneousPoisson
from tick.base import TimeFunction
from elephant.spike_train_generation import homogeneous_poisson_process
import argparse
#%%
@numba.jit(nopython=True)
def outside_pattern(spiketrain,new_spiketrain,pattern_times):
    actual_pattern = 0
    actual_spike = 0
    for i in range(len(spiketrain)):
        if actual_pattern<len(pattern_times)-1:
            if spiketrain[i]>pattern_times[actual_pattern+1]:
                actual_pattern+=1
        if spiketrain[i]>pattern_times[actual_pattern]+time_pattern:
            new_spiketrain[actual_spike,0] = spiketrain[i]
            new_spiketrain[actual_spike,1] = 0
            actual_spike+=1

    return actual_spike


@numba.jit(nopython=True)
def pattern_placement(actual_spike,pattern_times,new_spiketrain,motifs_sizes,n,choices_patterns):
    total_size = actual_spike
    
    for i in range(len(pattern_times)):
        which_pattern_kind = choices_patterns[i]
        which_pattern_pool = choices_pool[i]
        motif = all_pattern_pools[n,which_pattern_kind,which_pattern_pool,:pattern_pools_sizes[n,which_pattern_kind,which_pattern_pool]]
        new_spiketrain[total_size:total_size + len(motif),0] =motif+pattern_times[i]
        new_spiketrain[total_size:total_size + len(motif),1] = np.ones(len(motif))
        total_size += len(motif)

    return total_size

@numba.jit(nopython=True)
def pattern_placement_oscillation(actual_spike,pattern_times,new_spiketrain,motifs_sizes,n,choices_patterns):
    total_size = actual_spike
    
    for i in range(len(pattern_times)):
        which_pattern_kind = choices_patterns[i]
        which_pattern_pool = choices_pool[i]
        phase = (pattern_times[i]%times_phase[-1])
        which_phase = np.abs(times_phase-phase).argmin()+1
        motif = all_pattern_pools[n,which_pattern_kind,which_phase,which_pattern_pool,:pattern_pools_sizes[n,which_pattern_kind,which_phase,which_pattern_pool]]
        new_spiketrain[total_size:total_size + len(motif),0] = motif+pattern_times[i]
        new_spiketrain[total_size:total_size + len(motif),1] = np.ones(len(motif))
        total_size += len(motif)

    return total_size

# @numba.jit(nopython=True)
# def copy_data(datas,fill_until,total_size,new_spiketrain,n):
#         datas[fill_until:fill_until+total_size,0] = new_spiketrain[:total_size]
#         datas[fill_until:fill_until+total_size,1] = np.full(total_size,n)
#         return fill_until+total_size

@numba.jit(nopython=True)
def copy_data(datas,fill_until,total_size,new_spiketrain,n):
    datas[fill_until:fill_until+total_size,0] = new_spiketrain[:total_size,0]
    datas[fill_until:fill_until+total_size,1] = np.full(total_size,n)
    datas[fill_until:fill_until+total_size,2] =  new_spiketrain[:total_size,1]
    return fill_until+total_size

def pattern_pool_oscillation(n,index_pat_kind):
    pattern_train = homogeneous_poisson_process((rate+var_rate)*pq.Hz,t_start=0*pq.s,t_stop=time_pattern*pq.s,as_array=True)
    pattern_signal = np.zeros(int(time_pattern/delta_pat))
    osc_pattern_trains = dict()
    all_signals=dict()
    #print(pattern_train)
    for i in pattern_train:
        pattern_signal[int(i/delta_pat)]=1/delta_pat
    #print(np.sum(pattern_signal))
    pattern_signal_convolveld = np.convolve(pattern_signal,kern)/np.sum(kern)
    #print(np.sum(pattern_signal_convolveld))
    for t in sin_signals_pattern:
        osc = sin_signals_pattern[t]
        osc_pattern_trains[t] = []
        oscillation_pattern = osc[:-1]*pattern_signal_convolveld
        #print(np.sum(oscillation_pattern))
        all_signals[t] = TimeFunction((extanded_sample_pattern[:-1], oscillation_pattern), dt=delta_pat)

    for j in range(len(times_phase)):
        t = times_phase[j]
        sim = SimuInhomogeneousPoisson([all_signals[t]], end_time=(time_pattern)*2, verbose=False)
        for p in range(pool_size):
            sim.simulate()
            motif = sim.timestamps[0]
            #print(all_pattern_pools[n,index_pat_kind,j,p,:len(motif)])
            all_pattern_pools[n,index_pat_kind,j,p,:len(motif)] = motif
            pattern_pools_sizes[n,index_pat_kind,j,p] = len(motif)
            sim.reset()

    return osc_pattern_trains

def pattern_pool(n,index_pat_kind):
    
    pattern_train = homogeneous_poisson_process((rate)*pq.Hz,t_start=0*pq.s,t_stop=time_pattern*pq.s,as_array=True)
    pattern_signal = np.zeros(int(time_pattern/delta_pat))
    pattern_trains = []
    #print(pattern_train)
    for i in pattern_train:
        pattern_signal[int(i/delta_pat)]=1/delta_pat
    #print(np.sum(pattern_signal))
    pattern_signal_convolveld = np.convolve(pattern_signal,kern)/np.sum(kern)
    timefunction = TimeFunction((extanded_sample_pattern[:-1], pattern_signal_convolveld), dt=delta_pat)
    #print(np.sum(pattern_signal_convolveld))

    sim = SimuInhomogeneousPoisson([timefunction], end_time=(time_pattern)*2, verbose=False)
    for p in range(pool_size):
        sim.simulate()
        motif = sim.timestamps[0]
        all_pattern_pools[n,index_pat_kind,p,:len(motif)] = motif
        pattern_pools_sizes[n,index_pat_kind,p] = len(motif)
        sim.reset()

    return pattern_trains 


def simulate_no_pattern_neuron(n):

    if not oscillation:
        spiketrain = homogeneous_poisson_process(rate*pq.Hz,t_start=start,t_stop=start+time_seg,as_array=True)
    else:
        sim = SimuInhomogeneousPoisson([sign], end_time=time_seg, verbose=False)
        sim.simulate()
        spiketrain = sim.timestamps[0]+start

    return spiketrain,n,len(spiketrain)


def simulate_pattern_neuron(n):
   
    if not oscillation:
        spiketrain = homogeneous_poisson_process(rate*pq.Hz,t_start=start,t_stop=start+time_seg,as_array=True)
    else:
        sim = SimuInhomogeneousPoisson([sign], end_time=time_seg, verbose=False)
        sim.simulate()
        spiketrain = sim.timestamps[0]+start
    #new_spiketrain = np.empty( int(((rate*time_sim)/nb_segment)*10) ,dtype=np.float32)

    if len(spiketrain)>0:
        if s == 0 :
            for i in range(nb_pattern):
                # make_motif
                if not oscillation:
                    pattern_pool(n,i)
                    
                else :
                    pattern_pool_oscillation(n,i)


 
        actual_spike = outside_pattern(spiketrain,new_spiketrain,pattern_times)
        if not oscillation:
            total_size = pattern_placement(actual_spike,pattern_times,new_spiketrain,motifs_sizes,n,choices_patterns)
        else :
            total_size = pattern_placement_oscillation(actual_spike,pattern_times,new_spiketrain,motifs_sizes,n,choices_patterns)


        return new_spiketrain[:total_size],n,total_size
    
    return spiketrain,n,0

#%%

# parser = argparse.ArgumentParser(description='Generate spike trains')
# parser.add_argument('-rate',type=int,required=True)
# parser.add_argument('-oscillation',action='store_true')
# parser.add_argument('-frequency',type=float,required=False)
# parser.add_argument('-varrate',type=float,required=False)
# parser.add_argument('-timesim',type=float,required=True)
# parser.add_argument('-sampling_rate',type=float,required=False)
# parser.add_argument('-nbneurons',type=int,required=True)
# parser.add_argument('-nbsegment',type=int,required=True)
# parser.add_argument('-outdir',type=str,required=True)
# parser.add_argument('-pattern',action='store_true')
# parser.add_argument('-nbpattern',type=int,required=False)
# parser.add_argument('-timepattern',type=float,required=False)
# parser.add_argument('-patternfrequency',type=int,required=False)
# parser.add_argument('-sparsitypattern',type=float,required=False)
# parser.add_argument('-refpattern',type=float,required=False)
# args = parser.parse_args()
# %%
# rate = args.rate
# var_rate = args.r
# oscillation = args.oscillation
# oscillation_rate = args.oscillation_rate
# time_sim = args.timesim
# sampling_rate = args.samplingvar
# nb_neurons = args.nbneurons
# nb_segment = args.nbsegment
# outdir = args.outdir
# pattern = args.pattern
# if pattern :
#     time_pattern = args.timepattern
#     nb_pattern = args.nbpattern
#     sparsity_pattern = args.sparsitypattern
#     pattern_frequency = args.patternfrequency
#     ref_pattern = args.refpattern
# %%
rate = 10
var_rate = 10
oscillation = False
frequency = 36
time_sim = 1000
sampling_rate = 11
nb_neurons = 10
nb_segment = 1
outdir = "."
pattern = True

if pattern :
    time_pattern = 0.1
    nb_pattern = 1
    sparsity_pattern = 1
    pattern_frequency = 3
    ref_pattern = 0.05
#%%
if pattern:
    Lin_func_kern =lambda x,sigma: np.exp(-np.square(x/sigma))
    delta_pat = 0.002
    pool_size = 1000
    sample_kern = np.linspace(0,time_pattern,int(time_pattern/delta_pat))
    membran_time = 0.01
    kern = Lin_func_kern(sample_kern-time_pattern/2,membran_time)
if oscillation:
    samples = np.linspace(0,time_sim,time_sim*sampling_rate*frequency)
    sin_signal = np.sin(samples*frequency*np.pi*2)
    signal = rate +(var_rate*sin_signal)
    sign = TimeFunction((samples, signal), dt=1/(sampling_rate*frequency))



not_concerned_neurons = set(range(nb_neurons))
concerned_neurons= set()

if pattern:

    sample_pattern = np.linspace(0,time_pattern,int(time_pattern/delta_pat))
    extanded_sample_pattern = np.linspace(0,2*time_pattern,int((time_pattern*2)/delta_pat))
    times_phase = np.array([t for t in np.linspace(0,np.pi*2,sampling_rate)])
    raw_sin_signals_pattern= np.array([np.sin(extanded_sample_pattern*frequency*np.pi*2+t) for t in times_phase]) #replace per sampling rate
    scaled_sin_signals_pattern = ((raw_sin_signals_pattern/2)*((var_rate)/rate))+(1-((var_rate)/rate)/2)
    sin_signals_pattern = dict(zip(times_phase,scaled_sin_signals_pattern))

    new_spiketrain = np.empty( (int(((rate*time_sim)/nb_segment)*10),2 ) ,dtype=np.float32) #change size for longer simulations
    all_motifs = np.empty((nb_neurons,nb_pattern,int(((rate*time_pattern))*20)))
    motifs_sizes = np.empty((nb_neurons,nb_pattern),dtype=int)
    concerned_neurons = set( np.random.choice(range(nb_neurons),int(nb_neurons*sparsity_pattern), replace = False))
    not_concerned_neurons = not_concerned_neurons.difference(concerned_neurons)
    all_pattern_times = np.empty((int(time_sim*pattern_frequency*5),2))
    actual_nb_pattern = 0

    if oscillation:
        all_pattern_pools = np.empty((nb_neurons,nb_pattern,len(times_phase),pool_size,int(((rate*time_pattern))*20)))
        pattern_pools_sizes = np.empty((nb_neurons,nb_pattern,len(times_phase),pool_size),dtype=int)
    else :
        all_pattern_pools = np.empty((nb_neurons,nb_pattern,pool_size,int(((rate*time_pattern))*20)))
        pattern_pools_sizes = np.empty((nb_neurons,nb_pattern,pool_size),dtype=int)

#res = pattern_pool(time_pattern,delta_pat,sin_signals_pattern,extanded_sample_pattern,1000)
#plt.plot(np.array(oscillation_patterns).T)
# %%
datas = np.empty((int(((rate*time_sim*nb_neurons)/nb_segment)*2),3),dtype=np.float32)
time_seg =  (time_sim/nb_segment)*pq.s
# %%
for s in range(nb_segment):
    start = time_seg*s
    if pattern:
        pattern_times = homogeneous_poisson_process(pattern_frequency*pq.Hz,t_start=start,t_stop =time_seg*(s+1),refractory_period = (time_pattern+ref_pattern)*pq.s, as_array=True )
        choices_patterns = np.random.randint(0,nb_pattern,len(pattern_times))
        choices_pool = np.random.randint(0,pool_size,len(pattern_times))

        all_pattern_times[actual_nb_pattern:actual_nb_pattern+len(pattern_times),0]=pattern_times
        all_pattern_times[actual_nb_pattern:actual_nb_pattern+len(pattern_times),1]=choices_patterns
        
        actual_nb_pattern+=len(pattern_times)

    fill_until = 0


    with Pool(processes=36) as pool:

        multiple_thread = [pool.apply_async(simulate_no_pattern_neuron,(n,)) for n in not_concerned_neurons]
        
        for res in multiple_thread:
            final_spike_train,n,total_size=res.get()
            final_spike_train_color = np.array([final_spike_train,np.zeros(len(final_spike_train))]).T
            print(n)
            fill_until = copy_data(datas,fill_until,total_size,final_spike_train,n)
    
    if len(concerned_neurons)>0:

        with Pool(processes=36) as pool:
        
            multiple_thread = [pool.apply_async(simulate_pattern_neuron,(n,)) for n in concerned_neurons]

            for res in multiple_thread:
                final_spike_train,n,total_size=res.get()
                print(n)
                fill_until = copy_data(datas,fill_until,total_size,final_spike_train,n)


# %%
plt.scatter([1,2,3],[1,2,3])

# %%
# start_timer = timer()
# with Pool(processes=36) as pool:
#     multiple_thread = [pool.apply_async(do_simulation) for _ in range(nb_neurons)]
#     multiple_result = [res.get() for res in multiple_thread ]
# end = timer()
# print(end-start_timer)
# #%%
# sim = SimuInhomogeneousPoisson([sign], end_time=time_sim, verbose=False)
# ress = []
# start_timer = timer()
# for i in range(nb_neurons):
#     print(i)
#     sim.simulate()
#     ress.append(sim.timestamps[0])
#     sim.reset()
# end = timer()
# print(end-start_timer)
