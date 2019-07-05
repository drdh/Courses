# ## Preprocessing

# In[1]:


import numpy as np


# In[380]:


metadata=np.loadtxt("./data/HT_Sensor_UCIsubmission/HT_Sensor_metadata.dat",skiprows=1,dtype=str)


# In[3]:


metadata[metadata[:,2]=="background",2]=2
metadata[metadata[:,2]=="banana",2]=0
metadata[metadata[:,2]=="wine",2]=1

metadata=np.array(metadata[:,[0,2,3,4]],dtype=float)


# In[4]:


dataset = np.loadtxt('./data/HT_Sensor_UCIsubmission/HT_Sensor_dataset.dat', skiprows=1)
datasetID=np.array(dataset[:,0],dtype=int)


# In[5]:


# 仅仅挑选ID为banana与wine的
# 以及时间恰好有这两者存在的时间区间
selected = np.logical_and(metadata[datasetID,1]!=2,dataset[:,1]>0,dataset[:,1]<metadata[datasetID,3])
data=dataset[selected]
dataID=np.array(data[:,0],dtype=int)
data[:,0]=metadata[dataID,1]

#混淆in-place
np.random.shuffle(data)
#归一化
dataID=np.array(data[:,0],dtype=int)
data=(data-data.mean(axis=0))/data.std(axis=0)
data[:,0]=dataID
np.save("./data/HT_Sensor_UCIsubmission/data.npy",data)

