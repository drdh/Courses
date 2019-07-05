
 
# ## KNN

# In[6]:


import numpy as np
from collections import Counter


# In[207]:


data=np.load("./data/HT_Sensor_UCIsubmission/data.npy")
test_size=data.shape[0]//2
#分割train test 10%
X_test=data[:test_size,2:10]
y_test=data[:test_size,0].astype('int')

X_train=data[test_size:,2:10]
y_train=data[test_size:,0].astype('int')

print(X_test.shape,y_test.shape,X_train.shape,y_train.shape,data.shape)


# In[151]:


class KNN():
    def __init__(self, k=5):
        self.k = k

    def predict(self, X_test, X_train, y_train):
        y_pred=np.empty(X_test.shape[0],dtype=int)
        for i,X in enumerate(X_test):
            if i%100 ==0:
                print(i)
            y_pred[i]=np.bincount(y_train[np.argsort(np.linalg.norm(X-X_train,axis=1))[:self.k]]).argmax()
        return y_pred


# In[152]:


model=KNN(1)
y_pred=model.predict(X_test[:],X_train[:],y_train)


# In[155]:


result=Counter(np.equal(y_pred[:],y_test[:])).most_common()
result


# ```python
# result=Counter(np.equal(y_pred[:],y_test[:])).most_common()
# result
# ```
# ```python
# [(True, 41638), (False, 7)]
# ```

