 

# ## Preprocessing

# [Electrical Grid Stability Simulated Data Data Set ](https://archive.ics.uci.edu/ml/datasets/Electrical+Grid+Stability+Simulated+Data+)

# In[154]:


import numpy as ny
from collections import Counter
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[132]:


data=np.loadtxt("./data/PartI/Data_for_UCI_named.csv",delimiter=",",skiprows=1,dtype=str)
#"tau1","tau2","tau3","tau4","p1","p2","p3","p4","g1","g2","g3","g4","stab","stabf"
#0       1      2       3     4    5    6    7    8     9   10   11   12     13

data=data[:,[0,1,2,3,5,6,7,8,9,10,11,12]].copy()
#"tau1","tau2","tau3","tau4","p2","p3","p4","g1","g2","g3","g4","stab",
#   0     1     2      3      4    5    6    7    8    9    10   11


# In[133]:


#随机化
np.random.shuffle(data)
X=data[:,:11].copy().astype("float")
y=data[:,11].astype("float")>0
X=(X-X.mean(axis=0))/X.std(axis=0)

np.save("./data/PartI/X.npy",X)
np.save("./data/PartI/y.npy",y)

