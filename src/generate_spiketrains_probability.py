#%%
import numpy as np
from itertools import accumulate
from neo.core import AnalogSignal
import quantities as pq
import matplotlib.pyplot as plt
from elephant.spike_train_generation import inhomogeneous_poisson_process,homogeneous_poisson_process
from sys import argv
from neo.core import AnalogSignal
from timeit import default_timer as timer
from multiprocessing import Pool, TimeoutError
import numba
from scipy.signal.windows import gaussian
from tick.hawkes import SimuInhomogeneousPoisson
from tick.base import TimeFunction

#%%
Lin_func_kern =lambda x,sigma: np.exp(-np.square(x/sigma))
time = 10000
nb_neurons = 1024
sampling_rate = 10
frequency = 36
delta_pat = 0.002
time_pattern = 0.1
sample_kern = np.linspace(0,time_pattern,int(time_pattern/delta_pat))
membran_time = 0.02
kern = Lin_func_kern(sample_kern-time_pattern/2,membran_time)
rate = 10
var_rate = 10
samples = np.linspace(0,time,time*sampling_rate*frequency)
sin_signal = np.sin(samples*frequency*np.pi*2)
sample_pattern = np.linspace(0,time_pattern,int(time_pattern/delta_pat))
extanded_sample_pattern = np.linspace(0,2*time_pattern,int((time_pattern*2)/delta_pat))
times_phase = [t for t in np.linspace(0,np.pi*2,sampling_rate)]
raw_sin_signals_pattern= np.array([np.sin(extanded_sample_pattern*frequency*np.pi*2+t) for t in times_phase]) #replace per sampling rate
scaled_sin_signals_pattern = ((raw_sin_signals_pattern/2)*((var_rate)/rate))+(1-((var_rate)/rate)/2)

sin_signals_pattern = dict(zip(times_phase,scaled_sin_signals_pattern))

signal = rate +(var_rate*sin_signal)
sign = TimeFunction((samples, signal), dt=1/(sampling_rate*frequency))
# %%
plt.plot(samples[:100],signal[:100])
# %%

def pattern_pool(time_pattern,delta_pat,sin_signals_pattern,extanded_sample_pattern,pool_size):
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

    for t in all_signals:
        sim = SimuInhomogeneousPoisson([all_signals[t]], end_time=(time_pattern)*2, verbose=False)
        for _ in range(pool_size):
            sim.simulate()
            osc_pattern_trains[t].append(sim.timestamps)
            sim.reset()
            
    return osc_pattern_trains

#%%
res = pattern_pool(time_pattern,delta_pat,sin_signals_pattern,extanded_sample_pattern,1000)
#%%
res.keys()
#%%
res[1.5707963267948966]

# %%
plt.plot(np.array(oscillation_patterns).T)
# %%
def do_simulation():
    sim = SimuInhomogeneousPoisson([sign], end_time=time, verbose=False)
    sim.simulate()
    return sim.timestamps[0]

#%%
do_simulation()
# %%
start_timer = timer()
with Pool(processes=36) as pool:
    multiple_thread = [pool.apply_async(do_simulation) for _ in range(nb_neurons)]
    multiple_result = [res.get() for res in multiple_thread ]
end = timer()
print(end-start_timer)
#%%
sim = SimuInhomogeneousPoisson([sign], end_time=time, verbose=False)
ress = []
start_timer = timer()
for i in range(nb_neurons):
    print(i)
    sim.simulate()
    ress.append(sim.timestamps[0])
    sim.reset()
end = timer()
print(end-start_timer)