# %%
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
# %%
size_x = 100
size_y = 100
nb_neurons_e = 4096
nb_neurons_i = 1024
connectivity = 0.05
start_with = 0.5
nb_connected_to_e = int(connectivity*nb_neurons_e)
nb_connected_to_i = int(connectivity*nb_neurons_i)
# %%
x_exc=np.random.uniform(low=0, high=size_x, size=(nb_neurons_e,))
y_exc=np.random.uniform(low=0, high=size_y, size=(nb_neurons_e,))
pos_exc = np.array([x_exc,y_exc]).T
x_inh=np.random.uniform(low=0, high=size_x, size=(nb_neurons_i,))
y_inh=np.random.uniform(low=0, high=size_y, size=(nb_neurons_i,))
pos_inh = np.array([x_inh,y_inh]).T

# %%
def creat_connection_mat(connections,value):
    mat = np.array(np.empty(shape=[0, 3]))
    for n in range(len(connections)):
        inside = np.array([np.full(len(connections[n]),n)+1,np.array(connections[n])+1,np.full(len(connections[n]),value)]).T
        mat = np.concatenate((mat,inside))
    return mat

def save_connectivity(matrice,name,outdir,nb_from,nb_to):
    
    with open(outdir+"/connectivity_mat_"+name+".wmat",'w') as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write("%\n")
        f.write(str(nb_from)+" "+str(nb_to)+" "+str(matrice.shape[0])+"\n")
        fmt = '%d %d %g'
        fmt = '\n'.join([fmt]*matrice.shape[0])
        data = fmt % tuple(matrice.ravel())
        f.write(data)

def ideal_T(T,probability,distances):
    return probability-np.sum(cliped_negative_exp(distances,T))

def cliped_negative_exp(distances,T):
    clipped = np.clip(np.exp(-T*distances),None,1)
    return clipped[np.where(clipped>0.0001)]

def nearest_neig(point,all_points,size_x,size_y):
    return np.sqrt(np.square(all_points[:,0]-point[0])+np.square(all_points[:,1]-point[1]))

def nearest_neig_thorus(point,all_points,size_x,size_y):

    usual_x_dist = np.abs(all_points[:,0]-point[0])
    other_side_x_dist = size_x-(usual_x_dist)
    thorus_x_dist =  np.min(np.array([usual_x_dist,other_side_x_dist]),axis=0)

    usual_y_dist = np.abs(all_points[:,1]-point[1])
    other_side_y_dist = size_y-(usual_y_dist)
    thorus_y_dist =np.min(np.array([usual_y_dist,other_side_y_dist]),axis=0)

    thorus_dist = np.sqrt(np.square(thorus_x_dist)+np.square(thorus_y_dist))

    return thorus_dist 

# %%
distances = np.array([0.1,1,2,3])
print(cliped_negative_exp(distances,2))
# %%
plt.scatter(x_inh,y_inh)
# %%
all_nee_i = []
all_nee_d = []
all_nee_p = []
optimal_T = 0.1

for n in pos_exc:
    dists = nearest_neig_thorus(n,pos_exc,size_x,size_y)
    all_nee_i.append(np.argsort(dists))
    all_nee_d.append(dists[all_nee_i[-1]])
    optimal_T = fsolve(ideal_T,optimal_T,(nb_connected_to_e,all_nee_d[-1]))
    all_nee_p.append( cliped_negative_exp(all_nee_d[-1],optimal_T) )

all_nee_i = np.array(all_nee_i)
all_nee_d = np.array(all_nee_d)
plateau = int(connectivity*nb_neurons_e)
connection_nee = all_nee_i[:,:plateau]
# %%
all_nie_i = []
all_nie_d = []
all_nie_p = []

for n in pos_inh:
    dists = nearest_neig_thorus(n,pos_exc,size_x,size_y)
    all_nie_i.append(np.argsort(dists))
    all_nie_d.append(dists[all_nie_i[-1]])

all_nie_i = np.array(all_nie_i)
all_nie_d = np.array(all_nie_d)
plateau = int(connectivity*nb_neurons_e)
connection_nie = all_nie_i[:,:plateau]
# %%
all_nii_i = []
all_nii_d = []
all_nii_p = []

for n in pos_inh:
    dists = nearest_neig_thorus(n,pos_inh,size_x,size_y)
    all_nii_i.append(np.argsort(dists))
    all_nii_d.append(dists[all_nii_i[-1]])

all_nii_i = np.array(all_nii_i)
all_nii_d = np.array(all_nii_d)
plateau = int(connectivity*nb_neurons_i)
connection_nii = all_nii_i[:,:plateau]
# %%
all_nei_i = []
all_nei_d = []
all_nei_p = []

for n in pos_exc:
    dists = nearest_neig_thorus(n,pos_inh,size_x,size_y)
    all_nei_i.append(np.argsort(dists))
    all_nei_d.append(dists[all_nei_i[-1]])

all_nei_i = np.array(all_nei_i)
all_nei_d = np.array(all_nei_d)
plateau = int(connectivity*nb_neurons_i)
connection_nei = all_nei_i[:,:plateau]
# %%
connection_nei_mat = creat_connection_mat(connection_nei,0.72)
connection_nii_mat = creat_connection_mat(connection_nii,0.08)
connection_nee_mat = creat_connection_mat(connection_nee,0.1)
connection_nie_mat = creat_connection_mat(connection_nie,0.08)
# %%
save_connectivity(connection_nei_mat,"nei","./data",nb_neurons_e,nb_neurons_i)
save_connectivity(connection_nii_mat,"nii","./data",nb_neurons_i,nb_neurons_i)
save_connectivity(connection_nee_mat,"nee","./data",nb_neurons_e,nb_neurons_e)
save_connectivity(connection_nie_mat,"nie","./data",nb_neurons_i,nb_neurons_e)
# %%
# mon_n_c = connection_nei[10]
# # %%
# plt.scatter(x_inh,y_inh, c = ["red" if n in mon_n_c else "blue" for n in range(nb_neurons_i)],alpha = .4)
# %%
# stritcly_connected_prop = 1/4
# plateau = int((connectivity*nb_neurons_e)*stritcly_connected_prop)
# probablity_rest = connectivity - (connectivity*stritcly_connected_prop)
# nb_connection_rest = probablity_rest*nb_neurons_e
#%%
#considered = 2000
#distances = all_nee_d[0][:2000]
#distances = list(range(2000))
# %%
#connection_nee = all_nee[:,:plateau]
#%%
# all_connection_nee = []
# for n in range(len(pos_exc)):
#     new_connections_nee = []
#     print(n)
#     for i in range(len(all_nee_p[n])):
#         p = all_nee_p[n][i]
#         n2 = all_nee_i[n][i]
#         tirage = np.random.uniform()
#         if tirage < p :
#             new_connections_nee.append(n2)

#     all_connection_nee.append(np.concatenate((connection_nee[n],new_connections_nee)))
# %%
# mon_n_c = all_connection_nee[1]
# # %%
# plt.scatter(x_exc,y_exc, c = ["red" if n in mon_n_c else "blue" for n in range(nb_neurons_e)],alpha = .4)