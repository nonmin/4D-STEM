"""
# simulatin defocus data analysis
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

raw = np.load('simu_defocus_all_ronch.npy')

ue_root = np.load('simu_defocus_ue.npy')
ue_1b = np.load('simu_defocus_ue_1b.npy')
label = np.load('simu_defocus_ue_1b_label.npy')

n = label.max()
unique = list(set(label))
colors = [plt.cm.Set1(float(i)/max(unique)) for i in unique]

plt.figure(1)
for i, u in enumerate(unique):
    xi = [ue_1b[j,0] for j  in range(len(ue_1b[:,0])) if label[j] == u]
    yi = [ue_1b[j,1] for j  in range(len(ue_1b[:,1])) if label[j] == u]
    plt.scatter(xi, yi, c=colors[i], label=str(u))
plt.legend()
plt.title('Bootstrapped UMAP manifold pattern')
plt.show()

plt.figure(2)
plt.imshow(label.reshape(32,64),cmap='Set1')
plt.title('spatial distribution of cluster labels')

plt.figure(3)
for i, u in enumerate(unique):
    xi = [ue_root[j,0] for j  in range(len(ue_root[:,0])) if label[j] == u]
    yi = [ue_root[j,1] for j  in range(len(ue_root[:,1])) if label[j] == u]
    plt.scatter(xi, yi, c=colors[i], label=str(u))
plt.legend()
plt.show()
plt.title('UMAP manifold pattern')

bs = np.zeros((n+1,16800))
fig = plt.figure(figsize=(10, 10))
for i in range(n+1):
    ax = fig.add_subplot(3,3,(i+1))
    ind = np.argwhere(label==i).flatten()
    bs[i,:] = raw[ind,:].mean(axis=0)
    plt.imshow(np.flip(bs[i,:].reshape(120,140),0),cmap='inferno',vmin=0.00047, vmax=0.00057 )
    plt.title('cluster_%s'%i)
    plt.axis('off')
    plt.axis('square')
    plt.axis('tight')
    plt.colorbar(format='%.2e')
plt.suptitle('mean ronch of cluster')

bstd = np.zeros((n+1,16800))
fig = plt.figure(figsize=(10, 10))
for i in range(n+1):
    ax = fig.add_subplot(3,3,(i+1))
    ind = np.argwhere(label==i).flatten()
    bstd[i,:] = raw[ind,:].std(axis=0)
    plt.imshow(np.flip(bstd[i,:].reshape(120,140),0),cmap='inferno')
    plt.title('cluster_%s'%i)
    plt.axis('off')
    plt.axis('square')
    plt.axis('tight')
    plt.colorbar(format='%.2e')
plt.suptitle('ronch std of cluster')
    
    
L = np.zeros((n+1,2048))
fig = plt.figure(figsize=(20, 10))
for i in range(n+1):
    ax = fig.add_subplot(3,3,(i+1))
    L[i,:]= ans=euclidean_distances(raw, bs[i,:].reshape(1,16800)).flatten()
    plt.imshow(np.flip((1/L[i,:]).reshape(32,64),0),cmap=plt.cm.viridis)
    plt.title('cluster_%s'%i)
    plt.axis('off')
    plt.axis('square')
    plt.axis('tight')
    plt.colorbar()
plt.suptitle('similarity loadings')

