 

# ## Community

# [KMeans](https://gist.github.com/bistaumanga/6023692)
# 
# [KMeans Reinforce](https://gist.github.com/tvwerkhoven/4fdc9baad760240741a09292901d3abd)

# In[210]:


import numpy as np

def kMeans(X, K, maxIters = 10, plot_progress = None):
    
    centroids = X[np.random.choice(np.arange(len(X)), K)]
    for i in range(maxIters):
        if i%100==0:
            print("Iter: ",i)
        # Cluster Assignment step
        C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
        # Ensure we have K clusters, otherwise reset centroids and start over
        # If there are fewer than K clusters, outcome will be nan.
        if (len(np.unique(C)) < K):
            print("Reset")
            centroids = X[np.random.choice(np.arange(len(X)), K)]
        else:
            # Move centroids step 
            centroids = [X[C == k].mean(axis = 0) for k in range(K)]
        if plot_progress != None: plot_progress(X, C, np.array(centroids))
    return np.array(centroids) , C


# In[211]:


D1=graph.sum(axis=0).astype("float")
D1[D1==0]=0.1
D=np.diag(D1)
W=graph.copy().astype("float")


# ### Ratio Cut
# 
# $D-W$选择最小的k个特征向量$n\times k$ 

# In[11]:


#无需重新运行
w_r,v_r=np.linalg.eigh(D-W)
np.save("./data/graph/w_r.npy",w_r)
np.save("./data/graph/v_r.npy",v_r)


# In[212]:


w_r=np.load("./data/graph/w_r.npy")
v_r=np.load("./data/graph/v_r.npy")


# In[ ]:


#无需重跑
K=50 #K Clusters
k=100 #n*k
Miter=1000
centroids_r,C_r=kMeans(v_r[:,np.argsort(w_r)[:k]], K, maxIters = Miter, plot_progress = None)
np.save("./data/graph/C_r-50.npy",C_r)


# In[52]:


C_r=np.load("./data/graph/C_r-50.npy")
Counter(C_r).most_common()


# In[58]:


size=6000
choice=np.random.choice(name_num,size,replace=False)
g1=graph[choice][:,choice]
choice1=(g1.sum(axis=1)!=0)
G=nx.Graph(g1[choice1][:,choice1])
#G=nx.Graph(graph[:size,:size])
values=C_r[choice[choice1]]

options = {
    'node_color': 'red',
    'node_size': 0.1,
    'line_color': 'grey',
    #'linewidths': 0,
    'width': 0.1,
    'cmap':plt.get_cmap('jet'),
    'node_color':values
}
nx.draw(G,**options)
#plt.figure(figsize=(30,70))
plt.savefig("./graph/ratioCut"+str(size)+"-random.png",dpi=1000)
plt.show()


# ### Normalized Cut
# 
# $D^{-\frac{1}{2}}(D-W)D^{-\frac{1}{2}}$选择最小的k个特征向量$n\times k$

# In[13]:


#无需重新运行
w_n,v_n=np.linalg.eigh((np.diag(D1**(-1/2)))@(D-W)@(np.diag(D1**(-1/2))))
np.save("./data/graph/w_n.npy",w_n)
np.save("./data/graph/v_n.npy",v_n)


# In[59]:


w_n=np.load("./data/graph/w_n.npy")
v_n=np.load("./data/graph/v_n.npy")


# In[60]:


#无需重跑
K=20 #K Clusters
k=100 #n*k
Miter=1000
centroids_n,C_n=kMeans(v_n[:,np.argsort(w_n)[:k]], K, maxIters = Miter, plot_progress = None)
np.save("./data/graph/C_n.npy",C_n)


# In[ ]:


C_n=np.load("./data/graph/C_n.npy")
Counter(C_n).most_common()


# In[62]:


size=6000
choice=np.random.choice(name_num,size,replace=False)
g1=graph[choice][:,choice]
choice1=(g1.sum(axis=1)!=0)
G=nx.Graph(g1[choice1][:,choice1])
#G=nx.Graph(graph[:size,:size])
values=C_n[choice[choice1]]

options = {
    'node_color': 'red',
    'node_size': 0.1,
    'line_color': 'grey',
    #'linewidths': 0,
    'width': 0.1,
    'cmap':plt.get_cmap('jet'),
    'node_color':values,
    #'pos':nx.spectral_layout(G)
}
nx.draw(G,**options)
#plt.figure(figsize=(30,70))
plt.savefig("./graph/normalizedCut"+str(size)+"-random.png",dpi=1000)
plt.show()


# ### Modularity
# 
# $B=A-dd^\top /2m$ 选择最大的k个特征向量$n\times k$

# In[15]:


#无需重新运行
w_m,v_m=np.linalg.eigh(W-D1.reshape(name_num,1)@D1.reshape(1,name_num)/W.sum())
np.save("./data/graph/w_m.npy",w_m)
np.save("./data/graph/v_m.npy",v_m)


# In[63]:


w_m=np.load("./data/graph/w_m.npy")
v_m=np.load("./data/graph/v_m.npy")


# In[64]:


#无需重跑
K=20 #K Clusters
k=100 #n*k
Miter=1000
centroids_m,C_m=kMeans(v_m[:,np.argsort(w_m)[-k:]], K, maxIters = Miter, plot_progress = None)
np.save("./data/graph/C_m.npy",C_m)


# In[65]:


C_m=np.load("./data/graph/C_m.npy")
Counter(C_m).most_common()


# In[66]:


size=6000
choice=np.random.choice(name_num,size,replace=False)
g1=graph[choice][:,choice]
choice1=(g1.sum(axis=1)!=0)
G=nx.Graph(g1[choice1][:,choice1])
#G=nx.Graph(graph[:size,:size])
values=C_m[choice[choice1]]

options = {
    'node_color': 'red',
    'node_size': 0.1,
    'line_color': 'grey',
    #'linewidths': 0,
    'width': 0.1,
    'cmap':plt.get_cmap('jet'),
    'node_color':values
}
nx.draw(G,**options)
#plt.figure(figsize=(30,70))
plt.savefig("./graph/modularity"+str(size)+"-random.png",dpi=1000)
plt.show()


# In[ ]:




