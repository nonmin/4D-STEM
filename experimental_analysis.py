"""
#experimental data analysis
"""
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import hdbscan
from scipy.optimize import curve_fit

raw = np.load('expr_all_ronch.npy')

ue_root = np.load('expr_ue_root.npy')
ue_1b = np.load('expr_ue_1b.npy')

nk = 130
krange=np.linspace(20,149,num=nk).astype(np.uint32)
cn=np.zeros([1,nk])
Label_trend=[None]*nk
for k in range(130):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=k+20).fit(ue_1b)
    cluster_persistence=clusterer.cluster_persistence_
    cluster_label=clusterer.labels_
    cn[0,k]=len(clusterer.cluster_persistence_)
    Label_trend[k]=cluster_label

def func(x,c,t,b):
    return c*np.exp(-t * x)+b
 
popt, pcov = curve_fit(func, (krange-20).flatten(), cn.flatten())

cn_fit = func(krange-20,*popt)
tau = round(1/popt[1])

plt.figure(0)
plt.plot(krange, cn.flatten(), 'ro',label="Original Data")
plt.plot(krange, cn_fit, label=r"Fitted Curve, $\tau$ = %s"%tau) #same as line above \/ 
plt.legend(loc='upper right')
plt.xlabel("Minimum cluster size, k", fontsize=15)
plt.ylabel("Number of Cluster", fontsize=15)
plt.show()

label = Label_trend[66]

n = label.max()

plt.figure(1)
plt.imshow(label.reshape(64,64),cmap='Paired')
plt.axis('off')
plt.title('spatial distribution of cluster labels')

plt.figure(2)
for i in np.arange(-1,n+1):
    ind = np.argwhere(label==i)
    color = plt.cm.Paired(float(i+1)/(n+1))
    plt.scatter(ue_root[ind,0],ue_root[ind,1],c=color,label=str(i))
plt.legend(loc=2)
plt.show()
plt.title('UMAP manifold pattern')

plt.figure(3)
for i in np.arange(-1,n+1):
    ind = np.argwhere(label==i)
    color = plt.cm.Paired(float(i+1)/(n+1))
    plt.scatter(ue_1b[ind,0],ue_1b[ind,1],c=color,label=str(i))
plt.legend(loc=2)
plt.show()
plt.title('Bootstrapped UMAP manifold pattern')

amean = raw.mean(axis=0)
mean_ronch_subtracted = np.zeros((n+1,32400))
fig = plt.figure(figsize=(15, 10))
for i in range(n+1):
    ax = fig.add_subplot(3,4,(i+1))
    ind = np.argwhere(label==i).flatten()
    mean_ronch_subtracted[i,:] = raw[ind,:].mean(axis=0)-amean
    plt.imshow(np.flip(mean_ronch_subtracted[i,:].reshape(180,180).T,0),cmap='inferno')
    plt.colorbar()
    plt.title('cluster_%s'%i)
    plt.axis('off')
plt.suptitle('mean ronch of cluster subtracted by mean of all ronch')

mean_ronch = np.zeros((n+1,32400))
fig = plt.figure(figsize=(15, 10))
for i in range(n+1):
    ax = fig.add_subplot(3,4,(i+1))
    ind = np.argwhere(label==i).flatten()
    mean_ronch[i,:] = raw[ind,:].mean(axis=0)
    plt.imshow(np.flip(mean_ronch[i,:].reshape(180,180).T,0),cmap='inferno',vmin=8000,vmax=9800)
    plt.colorbar()
    plt.title('cluster_%s'%i)
    plt.axis('off')
plt.suptitle('mean ronch of cluster with mannually tuned colorbar range')

L = np.zeros((n+1,4096))
fig = plt.figure(figsize=(15, 10))
for i in range(n+1):
    L[i,:]=euclidean_distances(raw, mean_ronch[i,:].reshape(1,32400)).flatten()
    ax = fig.add_subplot(3,4,(i+1))
    plt.imshow((1/L[i,:]).reshape(64,64),cmap='viridis')
    plt.title('cluster_%s'%i)
    plt.axis('off')
plt.suptitle('similarity loadings')
