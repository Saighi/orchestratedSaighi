import numpy as np
from itertools import accumulate
from neo.core import AnalogSignal
import quantities as pq
import matplotlib.pyplot as plt
from elephant.spike_train_generation import inhomogeneous_poisson_process
from sys import argv
from neo.core import AnalogSignal

mean_rate = int(argv[1])
min_rate = int(argv[2])
max_rate = int(argv[3])
speed_change = int(argv[4])
time_sim = int(argv[5])
sampling_var = int(argv[6])
nb_neurons = int(argv[7])
nb_segment = int(argv[8])

rate_var = lambda n,spd : ((np.random.rand(n)*2)-1)*spd
rates = lambda n,mn_rate,spd,minv,maxv : np.array(list(accumulate(rate_var(n,spd),lambda i,j : min(maxv,max(minv,i+j)),initial=mn_rate)))

signal = AnalogSignal(rates(time_sim*sampling_var,mean_rate,speed_change,min_rate,max_rate),units = 'Hz',sampling_rate = sampling_var*(pq.Hz))
time_seg =  (time_sim/nb_segment)*pq.s

rates_over_time = []

for i in range(nb_neurons):
    rates_over_time.append(rates(int(time_sim*sampling_var),mean_rate,speed_change,min_rate,max_rate))

for s in range(nb_segment):
    start = time_seg*s
    datas = np.empty((int(((mean_rate*time_sim*nb_neurons)/nb_segment)*1.5),2),dtype=np.float32)
    fill_until = 0
    for n in range(nb_neurons):
        sign = AnalogSignal(rates_over_time[n][int(start*sampling_var):int((start+time_seg)*sampling_var)],units = 'Hz',sampling_rate = sampling_var*(pq.Hz),t_start=start,t_stop=start+time_seg)
        spiketrain = inhomogeneous_poisson_process(sign,as_array=True)
        print(n)
        datas[fill_until:fill_until+len(spiketrain)] = np.column_stack((spiketrain,np.full(len(spiketrain),n)))
        fill_until = fill_until+len(spiketrain)

    print("sorting...")
    datas = datas[:fill_until]
    datas = datas[np.argsort(datas[:,0])]
    print("saving...")
    np.savetxt("spiketrains_"+str(s),datas)
