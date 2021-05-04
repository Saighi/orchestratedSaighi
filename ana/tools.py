import numpy as np
from neo.core import SpikeTrain
from quantities import s


def inPatternOld(my_ras, dure_simu, duree_pattern, delay, multWindow=1,
                 time_step=0.0001):
    batch = duree_pattern+delay
    tile = np.append(np.ones(int(duree_pattern//time_step)),
                     np.zeros(int(delay//time_step)))
    signal = np.tile(tile, int(dure_simu//(batch))+1)
    rs_ntime = []

    for t in my_ras:
        rs_ntime.append(int(t//time_step))

    sizeSimb = int(dure_simu//batch)//multWindow
    batchSim = np.zeros(sizeSimb+1)
    batchCpt = np.zeros(sizeSimb+1)
    nbatch = int(batch//time_step)*multWindow
    for nt in rs_ntime:
        batchCpt[int(nt//nbatch)] += 1
        if signal[nt] == 1:
            batchSim[int(nt//nbatch)] += 1

    return batchSim/np.where(batchCpt == 0, 1, batchCpt)


def inOrOutPattern(my_ras, dure_simu, duree_pattern, delay, multWindow=1,
                   time_step=0.0001):
    batch = duree_pattern+delay
    tile = np.append(np.ones(int(duree_pattern//time_step)),
                     np.zeros(int(delay//time_step)))
    signal = np.tile(tile, int(dure_simu//(batch))+1)
    rs_ntime = []

    for t in my_ras:
        rs_ntime.append(int(t//time_step))

    sizeSimb = int(dure_simu//batch)//multWindow
    batchSim = np.zeros(sizeSimb+1)
    batchCpt = np.zeros(sizeSimb+1)
    nbatch = int(batch//time_step)*multWindow
    for nt in rs_ntime:
        batchCpt[int(nt//nbatch)] += 1
        if signal[nt] == 1:
            batchSim[int(nt//nbatch)] += 1

    return np.max([batchSim/batchCpt, 1-batchSim/batchCpt], axis=0)


def extractTrainsFiles(namefiles, nb_neurons, neo=False):
    if type(namefiles) == list:
        trainsStim = np.concatenate([np.loadtxt(name) for name in namefiles])
    else:
        trainsStim = np.loadtxt(namefiles)

    ListTrains = [[] for _ in range(nb_neurons)]
    for spike in range(len(trainsStim)):
        ListTrains[int(trainsStim[:, 1][spike])].append(
            trainsStim[:, 0][spike])
    if neo:
        ListNeoTrains = []
        for train in ListTrains:
            ListNeoTrains.append(SpikeTrain(train*s, t_stop=max(train)))
        return ListNeoTrains
    else:
        return ListTrains


def extractTrainFiles(spiketrain, nb_neurons):

    ListTrains = [[] for _ in range(nb_neurons)]
    for spike in range(len(spiketrain)):
        ListTrains[int(spiketrain[:, 1][spike])].append(
            spiketrain[:, 0][spike])

    return ListTrains


def TakeSampleTrain(namefiles, time):
    spikes = np.concatenate([np.loadtxt(name) for name in namefiles])
    spikes = spikes[spikes[:, 0].argsort()]
    return spikes[:np.argmax(spikes[:, 0] > time)]


def SpikesDistFromPatOld(spikeTrain, duree_pattern, delay):
    times = []
    dists = []
    for time in spikeTrain:
        times.append(time)
        dists.append(time % (duree_pattern+delay))
    return times, dists


""" spikeTrains and signal_times need to be ordered """


def SpikesDistFromPat(spikeTrain, duree_pattern, signal_times, window=0.5):
    times = []
    dists = []
    r = 0

    for event in signal_times:
        for spike_i in range(r, len(spikeTrain)):
            if spikeTrain[spike_i] > event+window/2:
                break
            if spikeTrain[spike_i] < event-window/2:
                r += 1
            else:
                dists.append(spikeTrain[spike_i]-event)
                times.append(spikeTrain[spike_i])

    return times, dists

def SpikesDistNeurones(spikeTrainN, duree_pattern, signal_times, window=0.5,offset=0, nb_neuron = 4096):
    
    spikeTrain = spikeTrainN[:,0]
    neurones = spikeTrainN[:,1]
    
    listdistN = [[] for _ in range(nb_neuron)] 
    
    r = 0

    for event in signal_times:
        for spike_i in range(r, len(spikeTrain)):
            if spikeTrain[spike_i] > (event+window/2)+offset:
                break
            if spikeTrain[spike_i] < (event-window/2)+offset:
                r += 1
            else:
                listdistN[int(neurones[spike_i])].append((spikeTrain[spike_i]-event,spikeTrain[spike_i]))


    return listdistN


def in_pattern(spikeTrain, duree_pattern, signal_times):
    in_pat= np.zeros(len(spikeTrain))


    r = 0
    for i in range(len(spikeTrain)):
        time = spikeTrain[i][0]
        distance_event2 = signal_times[1+r]-time

        if distance_event2 < 0 and r < len(signal_times)-2:
            while signal_times[1+r]-time < 0:
                r += 1
                if r > len(signal_times)-2:
                    break

        distance_event1 = time-signal_times[0+r]

        if distance_event1 < duree_pattern and distance_event1>0:
            in_pat[i] = 1

    return in_pat

