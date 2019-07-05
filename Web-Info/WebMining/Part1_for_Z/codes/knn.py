 

# ## KNN

# In[137]:


X=np.load("./data/PartI/X.npy")
y=np.load("./data/PartI/y.npy")

test_size=X.shape[0]//10
#分割train test 10%
X_test=X[:test_size]
y_test=y[:test_size]

X_train=X[test_size:]
y_train=y[test_size:]

print(X_test.shape,y_test.shape,X_train.shape,y_train.shape)


# In[143]:


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


# In[147]:


model=KNN(3)
y_pred=model.predict(X_test[:],X_train[:],y_train)


# In[148]:


result=Counter(np.equal(y_pred[:],y_test[:])).most_common()
result


# In[152]:


result=[]
for i in range(1,52,2):
    model=KNN(i)
    y_pred=model.predict(X_test[:],X_train[:],y_train)
    temp_result=Counter(np.equal(y_pred[:],y_test[:])).most_common()
    print(i,"\t",temp_result)
    result.append(temp_result)


# In[171]:


plt.figure(1)
plt.plot(range(1,52,2),[i[0][1]/1000 for i in result],'r')
#plt.plot(range(1,52,2),[i[1][1]for i in result],'g')
plt.title("KNN Precision of K")
plt.ylabel("Precision")
plt.xlabel("K")
plt.grid(True)
plt.savefig("./graph/PartI/KNN_Precision.png",dpi=1000)
plt.show()

