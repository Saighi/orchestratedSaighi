from operator import mul
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numba
from tick.hawkes import SimuInhomogeneousPoisson
from tick.base import TimeFunction
from elephant.spike_train_generation import homogeneous_poisson_process
from neo.core import SpikeTrain
import quantities as pq
import argparse

@numba.jit(nopython=True)
def outside_pattern(spiketrain,new_spiketrain,pattern_times):
    actual_pattern = -1
    actual_spike = 0
    for i in range(len(spiketrain)):
        
        while actual_pattern<len(pattern_times)-1 and spiketrain[i]>pattern_times[actual_pattern+1]:
            actual_pattern+=1

        if  (actual_pattern == -1 and spiketrain[i]<pattern_times[actual_pattern+1]) or spiketrain[i]>pattern_times[actual_pattern]+time_pattern or spiketrain[i]<pattern_times[actual_pattern] :
            new_spiketrain[actual_spike] = spiketrain[i]
            actual_spike+=1

    return actual_spike


@numba.jit(nopython=True)
def pattern_placement(neuron_pattern_pools,neuron_pattern_sizes,actual_spike,pattern_times,new_spiketrain,n,choices_patterns):
    total_size = actual_spike
    for i in range(len(pattern_times)):
        which_pattern_kind = choices_patterns[i]
        which_pattern_pool = choices_pool[i]
        motif = neuron_pattern_pools[which_pattern_kind,which_pattern_pool,:neuron_pattern_sizes[which_pattern_kind,which_pattern_pool]]
        new_spiketrain[total_size:total_size + len(motif)] = motif+pattern_times[i]
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
        motif = neuron_pattern_pools[which_pattern_kind,which_phase,which_pattern_pool,:neuron_pattern_sizes[which_pattern_kind,which_phase,which_pattern_pool]]
        new_spiketrain[total_size:total_size + len(motif)] = motif+pattern_times[i]
        total_size += len(motif)

    return total_size


def copy_data(datas,fill_until,total_size,new_spiketrain,n):
    datas[fill_until:fill_until+total_size,0] = new_spiketrain[:total_size]
    datas[fill_until:fill_until+total_size,1] = np.full(total_size,n)
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
        for i in pattern_train:
            pattern_signal[int(i/delta_pat)]=1/delta_pat
        pattern_signal_convolveld = np.convolve(pattern_signal,kern)/np.sum(kern)

        left_born=int((time_pattern/delta_pat)/2)
        right_born=int(((time_pattern)/delta_pat)/2 + (time_pattern/delta_pat))
        
        center_pattern = pattern_signal_convolveld[left_born:right_born]

        reversed_right = np.flip(pattern_signal_convolveld[right_born:])

        padded_reversed_right = np.pad(reversed_right, (len(center_pattern)-len(reversed_right),0), 'constant', constant_values=(0,))
     
        reversed_left = np.flip(pattern_signal_convolveld[:left_born])
        padded_reversed_left = np.pad(reversed_left, (0,len(center_pattern)-len(reversed_left)), 'constant', constant_values=(0,))
      
        pattern_signal_convolveld = pattern_signal_convolveld[left_born:right_born]+padded_reversed_left+padded_reversed_right

        for t in sin_signals_pattern:
            osc = sin_signals_pattern[t]

            oscillation_pattern = osc*pattern_signal_convolveld
            all_signals[t] = TimeFunction((sample_pattern, oscillation_pattern), dt=delta_pat)

        for j in range(len(times_phase)):
            t = times_phase[j]
            sim = SimuInhomogeneousPoisson([all_signals[t]], end_time=time_pattern, verbose=False)
            for p in range(pool_size):
                sim.simulate()
                motif = sim.timestamps[0]
                neuron_pattern_pools[index_pat_kind,j,p,:len(motif)] = motif
                neuron_pattern_pools_sizes[index_pat_kind,j,p] = len(motif)
                sim.reset()

    return neuron_pattern_pools,neuron_pattern_pools_sizes


def pattern_pool():

    neuron_pattern_pools = np.empty((nb_pattern,pool_size,int((((rate)*time_pattern))*20)))
    neuron_pattern_pools_sizes = np.empty((nb_pattern,pool_size),dtype=int)
    
    for index_pat_kind in range(nb_pattern):

        sim = SimuInhomogeneousPoisson([sign_pattern], end_time=time_pattern, verbose=False)
        sim.simulate()
        pattern_train = sim.timestamps[0]


        pattern_signal = np.zeros(int(time_pattern/delta_pat))
        
  
        for i in pattern_train:
            pattern_signal[int(i/delta_pat)]=1/delta_pat
    
        pattern_signal_convolveld = np.convolve(pattern_signal,kern)/np.sum(kern)

        left_born=int((time_pattern/delta_pat)/2)
        right_born=int(((time_pattern)/delta_pat)/2 + (time_pattern/delta_pat))
        
        center_pattern = pattern_signal_convolveld[left_born:right_born]

        reversed_right = np.flip(pattern_signal_convolveld[right_born:])

        padded_reversed_right = np.pad(reversed_right, (len(center_pattern)-len(reversed_right),0), 'constant', constant_values=(0,))
     
        reversed_left = np.flip(pattern_signal_convolveld[:left_born])
        padded_reversed_left = np.pad(reversed_left, (0,len(center_pattern)-len(reversed_left)), 'constant', constant_values=(0,))
      
        pattern_signal_convolveld = pattern_signal_convolveld[left_born:right_born]+padded_reversed_left+padded_reversed_right


        timefunction = TimeFunction((sample_pattern, pattern_signal_convolveld), dt=delta_pat)


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
    new_spiketrain = np.empty( (int(((rate*time_sim)/nb_segment)*10),) ,dtype=np.float64)

   
    sim = SimuInhomogeneousPoisson([sign],end_time =end_time, verbose=False)
    sim.simulate()
    spiketrain = sim.timestamps[0]
    actual_spike = outside_pattern(spiketrain,new_spiketrain,pattern_times)

    if len(spiketrain)>0:


        if not oscillation:
            total_size = pattern_placement(neuron_pattern_pools,neuron_pattern_sizes,actual_spike,pattern_times,new_spiketrain,n,choices_patterns)
        else :
            total_size = pattern_placement_oscillation(neuron_pattern_pools,neuron_pattern_sizes,actual_spike,pattern_times,new_spiketrain,n,choices_patterns)

        return new_spiketrain,n,total_size
    
    return spiketrain,n,0

def do_patterns_neuron(n):
    
    if not oscillation:
        neuron_pattern_pools,neuron_pattern_sizes = pattern_pool()
            
    else :
        neuron_pattern_pools,neuron_pattern_sizes = pattern_pool_oscillation()

    return neuron_pattern_pools,neuron_pattern_sizes,n


parser = argparse.ArgumentParser(description='Generate spike trains')
parser.add_argument('-rate',type=int,required=True)
parser.add_argument('-oscillation',action='store_true')
parser.add_argument('-frequency',type=float,required=False)
parser.add_argument('-varrate',type=float,required=False)
parser.add_argument('-timesim',type=float,required=True)
parser.add_argument('-samplingrate',type=float,required=False)
parser.add_argument('-nbneurons',type=int,required=True)
parser.add_argument('-nbsegment',type=int,required=True)
parser.add_argument('-outdir',type=str,required=True)
parser.add_argument('-pattern',action='store_true')
parser.add_argument('-shift',action='store_true')
parser.add_argument('-nbpattern',type=int,required=False)
parser.add_argument('-timepattern',type=float,required=False)
parser.add_argument('-patternfrequency',type=int,required=False)
parser.add_argument('-sparsitypattern',type=float,required=False)
parser.add_argument('-refpattern',type=float,required=False)
parser.add_argument('-membrantime',type=float,required=False)
parser.add_argument('-rateshift',type=float,required=False)
parser.add_argument('-startshift',type=float,required=False)
args = parser.parse_args()

shift = args.shift
rate_shift=args.rateshift
start_shift=args.startshift
rate = args.rate
var_rate = args.varrate
oscillation = args.oscillation
frequency = args.frequency
time_sim = args.timesim
sampling_rate = args.samplingrate
nb_neurons = args.nbneurons
nb_segment = args.nbsegment
outdir = args.outdir
pattern = args.pattern
if pattern :
    time_pattern = args.timepattern
    nb_pattern = args.nbpattern
    sparsity_pattern = args.sparsitypattern
    pattern_frequency = args.patternfrequency
    ref_pattern = args.refpattern
    membran_time = args.membrantime


datas = np.empty((int(((rate*time_sim*nb_neurons)/nb_segment)*2),2),dtype=np.float64)
time_seg =  (time_sim/nb_segment)
patterns_neurons = dict()
times_patterns_neurons = dict()

if pattern:
    Lin_func_kern =lambda x,sigma: np.exp(-np.square(x/sigma))
    delta_pat = 0.001
    pool_size = 100
    sample_kern = np.linspace(0,time_pattern,int(time_pattern/delta_pat))
    membran_time = membran_time
    kern = Lin_func_kern(sample_kern-time_pattern/2,membran_time)
    if oscillation :
        sign_pattern = TimeFunction(([0,time_pattern], [rate,rate]), dt=time_pattern)
    else:
        sign_pattern = TimeFunction(([0,time_pattern], [rate,rate]), dt=time_pattern)

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

    if oscillation:
        times_phase = np.array([t for t in np.linspace(0,np.pi*2,sampling_pattern)])
        raw_sin_signals_pattern= np.array([np.sin(sample_pattern*frequency*np.pi*2+t) for t in times_phase]) #replace per sampling rate
        scaled_sin_signals_pattern = ((raw_sin_signals_pattern/2)*(var_rate/rate))+1 #changed
        sin_signals_pattern = dict(zip(times_phase,scaled_sin_signals_pattern))

    concerned_neurons = set( np.random.choice(range(nb_neurons),int(nb_neurons*sparsity_pattern), replace = False))
    not_concerned_neurons = not_concerned_neurons.difference(concerned_neurons)
    all_pattern_times = np.empty((int(time_sim*pattern_frequency*5),2))
    actual_nb_pattern = 0


for s in range(nb_segment):

    start = time_seg*s
   
    start_seg = int((len(samples)/nb_segment)*s)
    if s != nb_segment-1:
        end_seg = int((len(samples)/nb_segment)*(s+1))+1
    else:
        end_seg = int((len(samples)/nb_segment)*(s+1))

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
                
                fill_until = copy_data(datas,fill_until,total_size,final_spike_train,n)
        
    if len(concerned_neurons)>0:
        if s == 0:
            with Pool(processes=10) as pool:
                multiple_thread = [pool.apply_async(do_patterns_neuron,(n,)) for n in concerned_neurons]
                for res in multiple_thread:
                    patterns_neuron,times_patterns_neuron,n = res.get()
                    patterns_neurons[n] = patterns_neuron
                    times_patterns_neurons[n] = times_patterns_neuron

        with Pool(processes=10) as pool:
        
            multiple_thread = [pool.apply_async(simulate_pattern_neuron,(n,patterns_neurons[n],times_patterns_neurons[n])) for n in concerned_neurons]

            for res in multiple_thread:
                final_spike_train,n,total_size=res.get()
                print(n)
                if len(final_spike_train)>0:
                    fill_until = copy_data(datas,fill_until,total_size,final_spike_train,n)

    fill_data = datas[:fill_until]
    print("sorting...")
    fill_data = fill_data[np.argsort(fill_data[:,0])]
    print("saving...")

    with open(outdir+"/spiketrains_"+str(s),'w') as f:
        fmt = '%.4f %d'
        fmt = '\n'.join([fmt]*fill_data.shape[0])
        data = fmt % tuple(fill_data.ravel())
        f.write(data)

if pattern:
    with open(outdir+"/pattern_times",'w') as f:
        fmt = '%.4f %d'
        fmt = '\n'.join([fmt]*all_pattern_times[:actual_nb_pattern].shape[0])
        data = fmt % tuple(all_pattern_times[:actual_nb_pattern].ravel())
        f.write(data)

