 
#!/usr/bin/env python
# coding: utf-8

# # Part II-2

# In[203]:


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter


# ## Graph Visualization

# [networkx](https://networkx.github.io/documentation/stable/tutorial.html)
# 
# [networkx 笔记](https://hui-liu.github.io/blog/networkx-%E7%AC%94%E8%AE%B0/)

# In[204]:


f=open("./data/graph/Erdos1_modified")
line=f.readlines()


# In[205]:


data=[]
temp=[]
count=0
for i in line:
    if i!='\n':
        s=i.strip("\t\n 1234567890*^:").lower()
        if s=="erdos, peter l.":
            count+=1
        else:
            temp.append(s)
    else:
        data.append(temp)
        temp=[]
data.append(temp)
print(count)


# In[206]:


name2seq=dict()
count=0
max_co=0
for i in data:
    if len(i)>max_co:
        max_co=len(i)
    for j in i:
        if not j in name2seq:
            name2seq[j]=count
            count+=1
name_num=len(name2seq)
print(name_num,max_co)


# In[207]:


seq2name={}
for i in name2seq.items():
    seq2name[i[1]]=i[0]


# In[208]:


graph=np.full((name_num,name_num),False)
for i in data:
    x=name2seq[i[0]]
    for j in i[1:]:
        y=name2seq[j]
        graph[x,y]=True
        graph[y,x]=True

