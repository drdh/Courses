 

# ## Logistic Regression

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


# In[167]:


LOSS=[]
class LogisticRegression():
    def __init__(self,lr=0.1):
        self.lr=lr
        
    def sigmoid(self,Z):
        return 1/(1+np.exp(-Z))
    
    def loss(self,y,y_hat):
        return -np.mean(y * np.log(y_hat)+(1-y)*np.log(1-y_hat))
    
    def fit(self,X_train,y_train,epochs=5000):
        limit=1/math.sqrt(X_train.shape[1])
        self.W=np.random.uniform(-limit,limit,(X_train.shape[1],))
        
        for i in range(epochs):
            y_hat=self.sigmoid(X_train @ self.W)
            self.W -= self.lr * (X_train.T @ (y_hat - y_train) / y_train.shape[0])
            temp_loss=self.loss(y_train,y_hat)
            LOSS.append((i,temp_loss))
            if i %100 ==0:
                print(i,temp_loss)
                
    def predict(self,X_test):
        y_pred=self.sigmoid(X_test @ self.W)>0.5
        return y_pred.astype('bool')


# In[168]:


model=LogisticRegression(0.25)
model.fit(X_train,y_train,5000)
y_pred=model.predict(X_test)


# In[169]:


Counter(y_pred == y_test).most_common()


# In[170]:


plt.figure(1)
plt.plot([i[0] for i in LOSS],[i[1] for i in LOSS],'r')
plt.title("Logistic Regression Loss")
plt.grid(True)
plt.ylabel("Loss")
plt.xlabel("Epoches")
plt.savefig("./graph/PartI/LR.png",dpi=1000)
plt.show()

