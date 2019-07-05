 

# ## Influence

# ### Degree Centrality

# In[7]:


degree=graph.sum(axis=0)
seq2name[degree.argmax()]


# ### Eigenvector Centrality

# In[23]:


#无需重新运行
w2,v2=np.linalg.eigh(graph.astype("float"))

np.save("./data/graph/w2.npy",w2)
np.save("./data/graph/v2.npy",v2)


# In[37]:


w2=np.load("./data/graph/w2.npy")
v2=np.load("./data/graph/v2.npy")
seq2name[v2[:,w2.argmax()].argmax()]


# ### Closeness Centrality

# In[48]:


#无需重新运行
#近似算法，计算距离
#下面的D可以从内存取出，迭代
D=np.full((name_num,name_num),name_num)
D[graph]=1
index=np.arange(name_num)
for k in range(10):
    print("====== ",k," ======")
    for i in range(n):
        if i%1000==0:
            print(i)
        nextP=index[D[i]!=name_num]
        for j in range(n):
            for p in nextP:
                D[i,j]=min(D[i,j],D[i,p]+D[p,j])
np.save("./data/graph/D.npy",D)


# In[51]:


Distance=np.load("./data/graph/D.npy")
D_avg=(name_num-1)/(Distance.sum(axis=1))


# In[52]:


seq2name[D_avg.argmax()]

