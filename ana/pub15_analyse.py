# %%
import numpy as np
sys.path.append(os.path.expanduser("~/auryn/tools/python/"))
from auryntools import *
import pandas as pd
import matplotlib.pyplot as plt
# %%
datadir = os.path.expanduser("~/data/sim_network/sim_one_excitatory_neuron") # Set this to your data path
prefix = "rf1"

# %%
spkfiles  = "%s/%s.0.ext.spk"%(datadir,prefix)
sfo = AurynBinarySpikeView(spkfiles)
# %%
spikes_ext = np.array(sfo.get_spikes())
# %%
spikes_ext
# %%
membrane_ex= pd.read_csv("%s/%s.0.e.mem"%(datadir,prefix),delimiter=' ').values
sse_w= pd.read_csv("%s/%s.0.sse"%(datadir,prefix),delimiter=' ').values
sse_x_u= pd.read_csv("%s/%s.u_x"%(datadir,prefix),delimiter=' ').values
# %%
plt.plot(membrane_ex[:,1])
# %%
plt.plot(sse[:,1])

# %%
plt.plot(sse_x_u[:,1])

# %%
plt.plot(sse_x_u[:,2])

# %%
plt.plot(sse_w[:,1])
# %%
sse_w