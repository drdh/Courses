 

# In[209]:


size=3000
choice=np.random.choice(name_num,size,replace=False)
g1=graph[choice][:,choice]
choice=(g1.sum(axis=1)!=0)
choice
graph


# In[45]:


size=6000
choice=np.random.choice(name_num,size,replace=False)
g1=graph[choice][:,choice]
choice=(g1.sum(axis=1)!=0)
G=nx.Graph(g1[choice][:,choice])
get_ipython().run_line_magic('matplotlib', 'inline')
options = {
    'node_color': 'red',
    'node_size': 0.1,
    'line_color': 'grey',
    #'linewidths': 0,
    'width': 0.1,
}
nx.draw(G,**options)
#plt.figure(figsize=(30,70))
plt.savefig("./graph/node"+str(size)+".png",dpi=1000)
plt.show()


# In[46]:


#!pip install python-louvain 
#思考是否可用动态的/ice and fire/knowledge graph
from community import community_louvain
size=4000
choice=np.random.choice(name_num,size,replace=False)
g1=graph[choice][:,choice]
choice=(g1.sum(axis=1)!=0)
G=nx.Graph(g1[choice][:,choice])
#G=nx.Graph(graph[:size,:size])
part =community_louvain.best_partition(G)
values = [part.get(node) for node in G.nodes()]

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
plt.savefig("./graph/node"+str(size)+"best_partition-random.png",dpi=1000)
plt.show()

