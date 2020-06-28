"""
# simulation data analysis
"""
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

raw = np.load('simu_all_ronch.npy')
ue = np.load('simu_ue.npy')

sc = SpectralClustering(n_clusters=7,random_state=0).fit(ue)
label = sc.labels_

n = label.max()
unique = list(set(label))
colors = [plt.cm.tab10(float(i)/max(unique)) for i in unique]

plt.figure(1)
for i, u in enumerate(unique):
    xi = [ue[j,0] for j  in range(len(ue[:,0])) if label[j] == u]
    yi = [ue[j,1] for j  in range(len(ue[:,1])) if label[j] == u]
    plt.scatter(xi, yi, c=colors[i], label=str(u))
plt.legend()
plt.show()
plt.title('UMAP manifold pattern')

plt.figure(2)
plt.imshow(label.reshape(37,64),cmap='tab10')
plt.title('spatial distribution of cluster labels')

mean_ronch = np.zeros((n+1,16800))
fig = plt.figure(figsize=(10, 10))
for i in range(n+1):
    ax = fig.add_subplot(3,3,(i+1))
    ind = np.argwhere(label==i).flatten()
    mean_ronch[i,:] = raw[ind,:].mean(axis=0)
    plt.imshow(mean_ronch[i,:].reshape(120,140),cmap='inferno')
    plt.title('cluster_%s'%i)
    plt.axis('off')
    plt.colorbar(format='%.0e')
plt.suptitle('mean ronch of cluster')

std_ronch = np.zeros((n+1,16800))
fig = plt.figure(figsize=(10, 10))
for i in range(n+1):
    ax = fig.add_subplot(3,3,(i+1))
    ind = np.argwhere(label==i).flatten()
    std_ronch[i,:] = raw[ind,:].std(axis=0)
    plt.imshow(std_ronch[i,:].reshape(120,140),cmap='inferno')
    plt.title('cluster_%s'%i)
    plt.axis('off')
    plt.colorbar(format='%.0e')
plt.suptitle('ronch std of cluster')

    
L = np.zeros((n+1,2368))
fig = plt.figure(figsize=(20, 10))
for i in range(n+1):
    ax = fig.add_subplot(3,3,(i+1))
    L[i,:]= ans=euclidean_distances(raw, mean_ronch[i,:].reshape(1,16800)).flatten()
    plt.imshow((1/L[i,:]).reshape(37,64),cmap=plt.cm.viridis)
    plt.title('cluster_%s'%i)
    plt.axis('off')
    plt.colorbar()
plt.suptitle('similarity loadings')


