 

# ## Extension

# [推荐系统之矩阵分解及其Python代码实现](https://www.cnblogs.com/shenxiaolin/p/8637794.html)

# In[178]:


#需要用到最上面的数据
print(train_matrix.shape,test_matrix.shape,rating_matrix.shape,movie_profile.shape)


# In[188]:


def matrix_factorization(R,P,Q,K,steps=5000,alpha=0.0002,beta=0.02):
    Q=Q.T  # .T操作表示矩阵的转置
    result=[]
    for step in range(steps):
        print(step)
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j]>0:
                    eij=R[i][j]-np.dot(P[i,:],Q[:,j]) # .dot(P,Q) 表示矩阵内积
                    for k in range(K):
                        P[i][k]=P[i][k]+alpha*(2*eij*Q[k][j]-beta*P[i][k])
                        Q[k][j]=Q[k][j]+alpha*(2*eij*P[i][k]-beta*Q[k][j])
        eR=np.dot(P,Q)
        e=0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j]>0:
                    e=e+pow(R[i][j]-np.dot(P[i,:],Q[:,j]),2)
                    for k in range(K):
                        e=e+(beta/2)*(pow(P[i][k],2)+pow(Q[k][j],2))
        result.append(e)
        if e<0.001:
            break
    return P,Q.T,result


# In[186]:


R=train_matrix.copy()
K=18 #设为电影的种类
P=np.random.rand(R.shape[0],K)
Q=np.random.rand(R.shape[1],K)


# In[191]:


nP,nQ,result=matrix_factorization(R,P,Q,K,steps=40)


# In[200]:


np.save("./data/ml-1m/nP.npy",nP)
np.save("./data/ml-1m/nQ.npy",nQ)
np.save("./data/ml-1m/MF_Result.npy",result)


# In[196]:


plt.plot(range(len(result)),result,color="r",linewidth=3)
plt.title("Convergence Curve")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("./graph/Extended_Convergence_Curve.png",dpi=1000)
plt.show()


# In[193]:


R_MF=np.dot(nP,nQ.T)


# In[194]:


ext_pred=R_MF[:test_user_size,:test_movie_size]


# In[195]:


# RMSE 0.010776504401721263
# MAE 0.8694166710904666
# P 0.46589403973509935 
# R 0.47469635627530365
result_matrix2=ext_pred.copy()
result_matrix2[test_matrix==0]=0
result_matrix=result_matrix2-test_matrix

RMSE=np.linalg.norm(result_matrix)/np.count_nonzero(test_matrix) 
MAE=np.sum(np.absolute(result_matrix))/np.count_nonzero(test_matrix)

N=10 #TopN
recommend_rate=3
pred=np.argsort(result_matrix2,axis=1)[:,-N:]
#truth=np.argsort(test_matrix,axis=1)[:,-N:]

Total_Right=0
index=np.arange(test_movie_size)
for i in range(test_user_size):
    #Total_Right+=np.intersect1d(pred[i],truth[i]).shape[0]
    
    Total_Right+=np.intersect1d(pred[i],index[test_matrix[i]>recommend_rate]).shape[0]
    
precision=Total_Right/(N*test_user_size)
recall=Total_Right/test_matrix[test_matrix>recommend_rate].shape[0]

print(RMSE,MAE,precision,recall)


# In[201]:


(0.4995033112582781 -0.46589403973509935)/0.46589403973509935


# In[202]:


(0.5089406207827261-0.47469635627530365)/0.47469635627530365

