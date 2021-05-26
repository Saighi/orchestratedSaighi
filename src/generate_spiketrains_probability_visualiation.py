#%%
from operator import mul
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numba
from tick.hawkes import SimuInhomogeneousPoisson
from tick.base import TimeFunction
from elephant.spike_train_generation import homogeneous_poisson_process
import plotly.express as px
from elephant.statistics import instantaneous_rate,mean_firing_rate,time_histogram
from neo.core import SpikeTrain
import quantities as pq
from elephant import kernels
import plotly.graph_objects as go

#%%
@numba.jit(nopython=True)
def outside_pattern(spiketrain,new_spiketrain,pattern_times):
    actual_pattern = -1
    actual_spike = 0
    for i in range(len(spiketrain)):
        
        while actual_pattern<len(pattern_times)-1 and spiketrain[i]>pattern_times[actual_pattern+1]:
            actual_pattern+=1
        

        if  (actual_pattern == -1 and spiketrain[i]<pattern_times[actual_pattern+1]) or spiketrain[i]>pattern_times[actual_pattern]+time_pattern or spiketrain[i]<pattern_times[actual_pattern]:
            new_spiketrain[actual_spike,0] = spiketrain[i]
            new_spiketrain[actual_spike,1] = 0
            actual_spike+=1

    return actual_spike


@numba.jit(nopython=True)
def pattern_placement(neuron_pattern_pools,neuron_pattern_sizes,actual_spike,pattern_times,new_spiketrain,n,choices_patterns):
    total_size = actual_spike
    for i in range(len(pattern_times)):
        which_pattern_kind = choices_patterns[i]
        which_pattern_pool = choices_pool[i]
        motif = neuron_pattern_pools[which_pattern_kind,which_pattern_pool,:neuron_pattern_sizes[which_pattern_kind,which_pattern_pool]]
        new_spiketrain[total_size:total_size + len(motif),0] = (motif+pattern_times[i])
        new_spiketrain[total_size:total_size + len(motif),1] = np.ones(len(motif))
        total_size += len(motif)

    return total_size

@numba.jit(nopython=True)
def pattern_placement_oscillation(neuron_pattern_pools,neuron_pattern_sizes,actual_spike,pattern_times,new_spiketrain,n,choices_patterns):
    total_size = actual_spike
    
    for i in range(len(pattern_times)):
        
        which_pattern_kind = choices_patterns[i]
        which_pattern_pool = choices_pool[i]
        rephased_time = (pattern_times[i])*(np.pi*2)*frequency
        phase = (rephased_time%(2*np.pi))
        which_phase = np.abs(times_phase-phase).argmin()
        #print(neuron_pattern_pools[which_pattern_kind,which_phase,which_pattern_pool])
        motif = neuron_pattern_pools[which_pattern_kind,which_phase,which_pattern_pool,:neuron_pattern_sizes[which_pattern_kind,which_phase,which_pattern_pool]]
        #print(motif)
        #stock = np.concatenate(stock,motif)
        new_spiketrain[total_size:total_size + len(motif),0] = (motif+pattern_times[i])
        new_spiketrain[total_size:total_size + len(motif),1] = np.ones(len(motif))
        total_size += len(motif)
    #print(np.isnan(stock).any)
    return total_size


def copy_data(datas,fill_until,total_size,new_spiketrain,n):
    datas[fill_until:fill_until+total_size,0] = new_spiketrain[:total_size,0]
    datas[fill_until:fill_until+total_size,1] = np.full(total_size,n)
    datas[fill_until:fill_until+total_size,2] =  new_spiketrain[:total_size,1]
    return fill_until+total_size

def pattern_pool_oscillation():

    neuron_pattern_pools = np.empty((nb_pattern,len(times_phase),pool_size,int(( (rate+var_rate)*time_pattern))*30)) #Changed
    neuron_pattern_pools_sizes = np.empty((nb_pattern,len(times_phase),pool_size),dtype=int)

    for index_pat_kind in range(nb_pattern):

        sim = SimuInhomogeneousPoisson([sign_pattern], end_time=time_pattern, verbose=False)
        sim.simulate()
        pattern_train = sim.timestamps[0]
        pattern_signal = np.zeros(int(time_pattern/delta_pat))

        all_signals=dict()
        #print(pattern_train)
        for i in pattern_train:
            pattern_signal[int(i/delta_pat)]=1/delta_pat
        #print(np.sum(pattern_signal))
        pattern_signal_convolveld = np.convolve(pattern_signal,kern)/np.sum(kern)

        left_born=int((time_pattern/delta_pat)/2)
        right_born=int(((time_pattern)/delta_pat)/2 + (time_pattern/delta_pat))
        
        center_pattern = pattern_signal_convolveld[left_born:right_born]

        reversed_right = np.flip(pattern_signal_convolveld[right_born:])

        padded_reversed_right = np.pad(reversed_right, (len(center_pattern)-len(reversed_right),0), 'constant', constant_values=(0,))
     
        reversed_left = np.flip(pattern_signal_convolveld[:left_born])
        padded_reversed_left = np.pad(reversed_left, (0,len(center_pattern)-len(reversed_left)), 'constant', constant_values=(0,))
      
        pattern_signal_convolveld = pattern_signal_convolveld[left_born:right_born]+padded_reversed_left+padded_reversed_right
        #print(np.sum(pattern_signal_convolveld))
        for t in sin_signals_pattern:
            osc = sin_signals_pattern[t]

            oscillation_pattern = osc*pattern_signal_convolveld
            #print(np.sum(oscillation_pattern))
            all_signals[t] = TimeFunction((sample_pattern, oscillation_pattern), dt=delta_pat)

        for j in range(len(times_phase)):
            t = times_phase[j]
            sim = SimuInhomogeneousPoisson([all_signals[t]], end_time=time_pattern, verbose=False)
            for p in range(pool_size):
                sim.simulate()
                motif = sim.timestamps[0]
                #print(all_pattern_pools[n,index_pat_kind,j,p,:len(motif)])
                neuron_pattern_pools[index_pat_kind,j,p,:len(motif)] = motif
                neuron_pattern_pools_sizes[index_pat_kind,j,p] = len(motif)
                sim.reset()

    return neuron_pattern_pools,neuron_pattern_pools_sizes


def pattern_pool():

    neuron_pattern_pools = np.empty((nb_pattern,pool_size,int((rate)*20)))
    neuron_pattern_pools_sizes = np.empty((nb_pattern,pool_size),dtype=int)
    
    for index_pat_kind in range(nb_pattern):

        #pattern_train = homogeneous_poisson_process(rate*pq.Hz,t_start=0*pq.s,t_stop=time_pattern*pq.s,as_array=True)

        sim = SimuInhomogeneousPoisson([sign_pattern], end_time=time_pattern, verbose=False)
        sim.simulate()
        pattern_train = sim.timestamps[0]


        pattern_signal = np.zeros(int(time_pattern/delta_pat))
        
        #print(pattern_train)
        for i in pattern_train:
            pattern_signal[int(i/delta_pat)]=1/delta_pat
        #print(np.sum(pattern_signal))
        pattern_signal_convolveld = np.convolve(pattern_signal,kern)/np.sum(kern)

        left_born=int((time_pattern/delta_pat)/2)
        right_born=int(((time_pattern)/delta_pat)/2 + (time_pattern/delta_pat))
        
        center_pattern = pattern_signal_convolveld[left_born:right_born]

        reversed_right = np.flip(pattern_signal_convolveld[right_born:])

        padded_reversed_right = np.pad(reversed_right, (len(center_pattern)-len(reversed_right),0), 'constant', constant_values=(0,))
     
        reversed_left = np.flip(pattern_signal_convolveld[:left_born])
        padded_reversed_left = np.pad(reversed_left, (0,len(center_pattern)-len(reversed_left)), 'constant', constant_values=(0,))
      
        pattern_signal_convolveld = pattern_signal_convolveld[left_born:right_born]+padded_reversed_left+padded_reversed_right

        #print(len(pattern_signal_convolveld_test))
        timefunction = TimeFunction((sample_pattern, pattern_signal_convolveld), dt=delta_pat)
        #print(np.sum(pattern_signal_convolveld))

        sim = SimuInhomogeneousPoisson([timefunction], end_time=time_pattern, verbose=False)
        for p in range(pool_size):
            sim.simulate()
            motif = sim.timestamps[0]
            neuron_pattern_pools[index_pat_kind,p,:len(motif)] = motif
            neuron_pattern_pools_sizes[index_pat_kind,p] = len(motif)
            sim.reset()

    return neuron_pattern_pools,neuron_pattern_pools_sizes


def simulate_no_pattern_neuron(n):

    sim = SimuInhomogeneousPoisson([sign],end_time =end_time , verbose=False)
    sim.simulate()
    spiketrain = sim.timestamps[0]

    return spiketrain,n,len(spiketrain)


def simulate_pattern_neuron(n,neuron_pattern_pools,neuron_pattern_sizes):

    new_spiketrain = np.empty( (int(((rate*time_sim)/nb_segment)*10),2 ) ,dtype=np.float64)
    
    sim = SimuInhomogeneousPoisson([sign],end_time =end_time, verbose=False)
    sim.simulate()
    spiketrain = sim.timestamps[0]

    actual_spike = outside_pattern(spiketrain,new_spiketrain,pattern_times)



    if len(spiketrain)>0:

        #print(new_spiketrain[:actual_spike])
        if not oscillation:
            total_size = pattern_placement(neuron_pattern_pools,neuron_pattern_sizes,actual_spike,pattern_times,new_spiketrain,n,choices_patterns)
        else :
            total_size = pattern_placement_oscillation(neuron_pattern_pools,neuron_pattern_sizes,actual_spike,pattern_times,new_spiketrain,n,choices_patterns)
        # print(np.isnan(new_spiketrain[:total_size]).any())
        # print(np.mean(new_spiketrain[:total_size]))
        return new_spiketrain,n,total_size

    return spiketrain,n,0

def do_patterns_neuron(n):
    
    if not oscillation:
        neuron_pattern_pools,neuron_pattern_sizes = pattern_pool()
            
    else :
        neuron_pattern_pools,neuron_pattern_sizes = pattern_pool_oscillation()

    return neuron_pattern_pools,neuron_pattern_sizes,n
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
rate = 25
var_rate = 25
oscillation = True
frequency = 100
time_sim = 10
sampling_rate = 10
nb_neurons = 70
nb_segment = 1
outdir = "."
pattern = True
shift = False
rate_shift=20
start_shift=5

if pattern :
    time_pattern = 0.1
    nb_pattern = 1
    sparsity_pattern = 1
    pattern_frequency = 3
    ref_pattern = 0.05
# %%
datas = np.empty((int(((rate*time_sim*nb_neurons)/nb_segment)*10),3),dtype=np.float64)
time_seg =  (time_sim/nb_segment)
patterns_neurons = dict()
times_patterns_neurons = dict()

if pattern:
    Lin_func_kern =lambda x,sigma: np.exp(-np.square(x/sigma))
    delta_pat = 0.0005
    pool_size = 1000
    sample_kern = np.linspace(0,time_pattern,int(time_pattern/delta_pat))
    membran_time = 0.00666
    kern = Lin_func_kern(sample_kern-time_pattern/2,membran_time)
    if oscillation :
        sign_pattern = TimeFunction(([0,time_pattern], [rate,rate]), dt=time_pattern)
    else:
        sign_pattern = TimeFunction(([0,time_pattern], [rate,rate]), dt=time_pattern)
# if not oscillation :
#     sign = TimeFunction(([0,time_sim], [rate,rate]), dt=time_sim)
if oscillation :
    samples = np.linspace(0,time_sim,int(time_sim*sampling_rate*frequency))
    remove = -len(samples)%nb_segment if len(samples)%nb_segment !=0 else len(samples)
    samples = samples[:remove]
    sin_signal = np.sin((samples*frequency*np.pi*2)) # Ajuster la phase
    signal = rate +(var_rate*sin_signal)
else :
    sampling_no_osc = 1000
    if shift :
        #start_to_shift = start_shift-start
        shift_to_end = time_sim-start_shift
        shift_to_end_sim = time_sim-start_shift
        final_rate = rate_shift/(shift_to_end/shift_to_end_sim)

        start_sign = np.linspace(0,start_shift,int(sampling_no_osc/2))
        start_rate = np.linspace(rate,rate,int(sampling_no_osc/2))

        end_sign = np.linspace(start_shift,time_sim,int(sampling_no_osc/2))
        end_rate = np.linspace(rate,final_rate,int(sampling_no_osc/2))

        samples=np.concatenate((start_sign,end_sign))
        signal=np.concatenate((start_rate,end_rate))
    else :
        samples=np.linspace(0,time_sim,sampling_no_osc)
        signal=np.linspace(rate,rate,sampling_no_osc)
    




not_concerned_neurons = set(range(nb_neurons))
concerned_neurons= set()

if pattern:
    sampling_pattern = 31
    sample_pattern = np.linspace(0,time_pattern,int(time_pattern/delta_pat))
    #extanded_sample_pattern = np.linspace(0,2*time_pattern,int((time_pattern*2)/delta_pat))
    times_phase = np.array([t for t in np.linspace(0,np.pi*2,sampling_pattern)])
    raw_sin_signals_pattern= np.array([np.sin(sample_pattern*frequency*np.pi*2+t) for t in times_phase]) #replace per sampling rate
    scaled_sin_signals_pattern = ((raw_sin_signals_pattern)*((var_rate)/rate))+1 #Changed
    sin_signals_pattern = dict(zip(times_phase,scaled_sin_signals_pattern))

    concerned_neurons = set( np.random.choice(range(nb_neurons),int(nb_neurons*sparsity_pattern), replace = False))
    not_concerned_neurons = not_concerned_neurons.difference(concerned_neurons)
    all_pattern_times = np.empty((int(time_sim*pattern_frequency*5),2))
    actual_nb_pattern = 0

#res = pattern_pool(time_pattern,delta_pat,sin_signals_pattern,extanded_sample_pattern,1000)
#plt.plot(np.array(oscillation_patterns).T)
#pattern_times = homogeneous_poisson_process(pattern_frequency*pq.Hz,t_start=0*pq.s,t_stop =time_seg*(0+1)*pq.s,refractory_period = (time_pattern+ref_pattern)*pq.s, as_array=True )

# %%
reseve = np.array([])
show_data = np.array([])
fir = True
for s in range(nb_segment):

    start = time_seg*s

    start_seg = int((len(samples)/nb_segment)*s)
    if s != nb_segment-1:
        end_seg = int((len(samples)/nb_segment)*(s+1))+1
    else:
        end_seg = int((len(samples)/nb_segment)*(s+1))

    reseve = np.concatenate((reseve,signal[start_seg:end_seg]))
    elapsed_time = samples[end_seg-1]-samples[start_seg]

    sign = TimeFunction((samples[start_seg:end_seg], signal[start_seg:end_seg]), dt=elapsed_time/len(samples[start_seg:end_seg]))
    end_time =samples[end_seg-1]

    if pattern:
        pattern_times = homogeneous_poisson_process(pattern_frequency*pq.Hz,t_start=start*pq.s,t_stop =(time_seg*(s+1)-time_pattern)*pq.s,refractory_period = (time_pattern+ref_pattern)*pq.s, as_array=True )
        choices_patterns = np.random.randint(0,nb_pattern,len(pattern_times))
        choices_pool = np.random.randint(0,pool_size,len(pattern_times))

        all_pattern_times[actual_nb_pattern:actual_nb_pattern+len(pattern_times),0]=pattern_times
        all_pattern_times[actual_nb_pattern:actual_nb_pattern+len(pattern_times),1]=choices_patterns
        
        actual_nb_pattern+=len(pattern_times)

    fill_until = 0


    with Pool(processes=10) as pool:

        multiple_thread = [pool.apply_async(simulate_no_pattern_neuron,(n,)) for n in not_concerned_neurons]
        
        for res in multiple_thread:
            final_spike_train,n,total_size=res.get()
            print(n)
            if len(final_spike_train)>0:
                final_spike_train_color = np.array([final_spike_train,np.zeros(len(final_spike_train))]).T
                fill_until = copy_data(datas,fill_until,total_size,final_spike_train_color,n)
    
    if len(concerned_neurons)>0:
        if s == 0:
            with Pool(processes=10) as pool:
                multiple_thread = [pool.apply_async(do_patterns_neuron,(n,)) for n in concerned_neurons]
                for res in multiple_thread:
                    patterns_neuron,times_patterns_neuron,n = res.get()
                    print(n)
                    patterns_neurons[n] = patterns_neuron
                    times_patterns_neurons[n] = times_patterns_neuron

        with Pool(processes=10) as pool:
        
            multiple_thread = [pool.apply_async(simulate_pattern_neuron,(n,patterns_neurons[n],times_patterns_neurons[n])) for n in concerned_neurons]

            for res in multiple_thread:
                final_spike_train,n,total_size=res.get()
                #print(np.isnan(final_spike_train[:total_size]).any())
                # print(len(final_spike_train[:total_size]))
                print(n)
                if len(final_spike_train)>0:
                    fill_until = copy_data(datas,fill_until,total_size,final_spike_train,n)
   
    fill_data = datas[:fill_until]
    fill_data = fill_data[np.argsort(fill_data[:,0])]
    if fir:
        show_data = fill_data
    else:
        show_data= np.concatenate((show_data,fill_data),axis=0)
    fir = False
    #len(fill_data)
#%%
def f(x, y):
    return np.sin(x*frequency*np.pi*2)
#%%
# if oscillation:
#     fig, ax = plt.subplots()
#     x = np.linspace(0,time_sim+time_pattern,1000)
#     y = np.linspace(0,nb_neurons,1000)
#     X, Y = np.meshgrid(x, y)
#     Z=f(X,Y)
#     ax.scatter(show_data[:,0],show_data[:,1],c=["red" if i == 1 else "blue" for i in show_data[:,2]],s=5,zorder=3)
#     ax.contour(X,Y,Z,100,cmap='Greys')
#     plt.show()
#%%
t_fro = 0
t_to = 10
conv = lambda x: 1024*10*x

to_show = show_data[conv(t_fro):conv(t_to)]
#plt.scatter(to_show[:,0],to_show[:,1],c=["red" if i == 1 else "blue" for i in to_show[:,2]],s=5)
fig = px.scatter(x=to_show[:,0], y=to_show[:,1],color=["red" if i == 1 else "blue" for i in to_show[:,2]])
#fig = px.scatter(x=to_show[:,0], y=to_show[:,1])

fig.show()

#%%
spk = SpikeTrain(to_show[:,0]*pq.s,np.max(to_show[:,0])*pq.s,t_start=np.min(to_show[:,0]))
rte = instantaneous_rate(spk,0.01*pq.s,kernels.GaussianKernel(sigma=10 * pq.ms))
#mean =mean_firing_rate(spk)
#%%
fig = go.Figure(data=go.Scatter(x=rte.times, y=np.array(np.array(rte).ravel())))
fig.show()

# %%
from viziphant.statistics import plot_instantaneous_rates_colormesh
all_spikes_all_neurons = []
for neuron in range(nb_neurons):
    index_pat = 0
    all_spikes = np.array([])
    if oscillation :
        for i in range(len(patterns_neurons[neuron][index_pat][0])):
            #print(times_patterns_neurons[neuron][index_pat][0][i])
            all_spikes = np.concatenate((all_spikes,patterns_neurons[neuron][index_pat][0][i][:times_patterns_neurons[neuron][index_pat][0][i]]))
    else:
        for i in range(len(patterns_neurons[neuron][index_pat])):
            #print(times_patterns_neurons[neuron][index_pat][0][i])
            all_spikes = np.concatenate((all_spikes,patterns_neurons[neuron][index_pat][i][:times_patterns_neurons[neuron][index_pat][i]]))
    
    all_spikes = np.sort(all_spikes)
    if len(all_spikes)>0:
        all_spikes_all_neurons.append(SpikeTrain(all_spikes*pq.s,time_pattern*pq.s,t_start=0))
# %%
#spk_pool = SpikeTrain(all_spikes*pq.s,np.max(all_spikes)*pq.s,t_start=np.min(all_spikes))
rte_pool = instantaneous_rate(all_spikes_all_neurons,0.0005*pq.s,kernels.GaussianKernel(sigma=0.5 * pq.ms))
# %%
plot_instantaneous_rates_colormesh(rte_pool)
# %%
