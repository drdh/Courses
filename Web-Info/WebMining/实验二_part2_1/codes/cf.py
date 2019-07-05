
# ## Collaborative Filtering

# In[99]:


#需要用到上面的数据
print(train_matrix.shape,test_matrix.shape,rating_matrix.shape)


# In[104]:


movie_sim=np.zeros((movie_num+1,movie_num+1))
movie_bar=np.zeros(movie_num+1)


# In[7]:


#无需重新运行
#item相似性 movie[i,j]表示电影i,j的相似性
for i in range(movie_num+1):
    ri=train_matrix[:,i]
    nonzero=np.count_nonzero(ri)
    if nonzero!=0:
        movie_bar[i]=ri.sum()/nonzero


for i in range(1,movie_num+1):
    print(i)
    for j in range(1,movie_num+1):
        if i>j:
            movie_sim[i,j]=movie_sim[j,i]
            continue
        ri=train_matrix[:,i]
        rj=train_matrix[:,j]
        
        rri=np.zeros(user_num+1)
        rrj=np.zeros(user_num+1)
        selected=np.logical_and(ri,rj)
        rri[selected]=ri[selected]-movie_bar[i]
        rrj[selected]=rj[selected]-movie_bar[j]
        
        norm=np.linalg.norm(rri)*np.linalg.norm(rrj)
        if norm==0:
            movie_sim[i,j]=0
        else:
            movie_sim[i,j]=(rri@rrj)/(norm)

np.save("./data/ml-1m/movie_sim.npy",movie_sim)


# In[105]:


movie_sim=np.load("./data/ml-1m/movie_sim.npy")
user_bar=np.zeros(user_num+1)
for i in range(user_num+1):
    ri=train_matrix[i]
    nonzero=np.count_nonzero(ri)
    if nonzero!=0:
        user_bar[i]=ri.sum()/nonzero


# In[106]:


miu=train_matrix.sum()/np.count_nonzero(train_matrix)
bxi=user_bar.reshape(user_num+1,1)+movie_bar-miu


# In[108]:


k=5
cf_pred=np.zeros((test_user_size,test_movie_size))

for x in range(1,test_user_size):
    if x%100==0:
        print(x)
    for i in range(1,test_movie_size):
        index=np.arange(movie_num+1)[train_matrix[x].astype("bool")]
        Nix=index[np.argsort(movie_sim[i,train_matrix[x].astype("bool")])[-k:]]
        
        sumij=movie_sim[i,Nix].sum()
        if sumij!=0:
            cf_pred[x,i]=bxi[x,i]+(movie_sim[i,Nix]@(train_matrix[x,Nix]-bxi[x,Nix]))/sumij


# In[164]:


#result_matrix=np.zeros((test_user_size,test_movie_size))
#for i in range(test_user_size):
#    for j in range(test_movie_size):
#        if test_matrix[i,j]!=0:
#            result_matrix[i,j]=cf_pred[i,j]-test_matrix[i,j]
result_matrix2=cf_pred.copy()
result_matrix2[test_matrix==0]=0
result_matrix=result_matrix2-test_matrix


# In[165]:


#RMSE
np.linalg.norm(result_matrix)/np.count_nonzero(test_matrix)


# In[166]:


#MAE
np.sum(np.absolute(result_matrix))/np.count_nonzero(test_matrix)


# In[167]:


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

print(precision,recall)


# In[168]:


def Pred_of_N(N):
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
    
    return precision,recall


# In[169]:


PR=[Pred_of_N(N) for N in range(1,50)]


# In[175]:


plt.figure(1)
plt.plot(range(1,50),[i[0] for i in PR],label="Precision",color="g")
plt.plot(range(1,50),[i[1] for i in PR],label="Recall",color="r")
plt.plot(range(1,50),[2*(i[0]*i[1])/(i[0]+i[1]) for i in PR],label="F1",color="y")
plt.title("Collaborative Filtering P/R/F1")
plt.xlabel("N")
plt.ylabel("Ratio")
plt.grid(True)
plt.legend()
plt.savefig("./graph/Collaborative_Filter-PRF1_of_N.png")
plt.show()


 
