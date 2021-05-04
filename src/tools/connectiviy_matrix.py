#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
size = 1000
nb_neurons = 4096

x_coordinates_e = np.random.uniform(low=0, high=size, size=(nb_neurons,))
y_coordinates_e = np.random.uniform(low=0, high=size, size=(nb_neurons,))
x_coordinates_i = np.random.uniform(low=0, high=size, size=(nb_neurons/4,))
y_coordinates_i = np.random.uniform(low=0, high=size, size=(nb_neurons/4,))

plt.scatter(x_coordinates,y_coordinates)
# %%
