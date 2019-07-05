 

# ## Content-Based

# In[6]:


#需要用到上面的数据
user_profile=np.zeros((test_user_size,user_num+1))
for i in range(test_user_size):
    user_profile[i]=np.sum(train_matrix*train_matrix[i],axis=1)
    if i%100==0:
        print(i)


# In[7]:


cb_pred=np.zeros((test_user_size,test_movie_size))
for i in range(1,test_user_size):
    for j in range(1,test_movie_size):
        norm=np.linalg.norm(train_matrix[:,j])*np.linalg.norm(user_profile[i])
        if norm == 0:
            cb_pred[i,j]=0
        else:
            cb_pred[i,j]=((train_matrix[:,j]@user_profile[i])/(norm))*5


# In[149]:


#result_matrix=np.zeros((test_user_size,test_movie_size))
#result_matrix2=np.zeros((test_user_size,test_movie_size))
#for i in range(test_user_size):
#    for j in range(test_movie_size):
#        if test_matrix[i,j]!=0:
#            result_matrix[i,j]=cb_pred[i,j]-test_matrix[i,j]

result_matrix2=cb_pred.copy()
result_matrix2[test_matrix==0]=0
result_matrix=result_matrix2-test_matrix


# $$
# RMSE=\sqrt{\frac{1}{n}\sum_{j=1}^n (y_j-\bar{y}_j)^2}
# $$

# In[150]:


#RMSE
np.linalg.norm(result_matrix)/np.count_nonzero(test_matrix)


# $$
# MAE=\frac{1}{n}\sum_{j=1}^n|y_j-\bar{y}_j|
# $$

# In[151]:


#MAE
np.sum(np.absolute(result_matrix))/np.count_nonzero(test_matrix)


# In[154]:


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


# In[155]:


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


# In[156]:


PR=[Pred_of_N(N) for N in range(1,50)]


# In[174]:


plt.figure(1)
plt.plot(range(1,50),[i[0] for i in PR],label="Precision",color="g")
plt.plot(range(1,50),[i[1] for i in PR],label="Recall",color="r")
plt.plot(range(1,50),[2*(i[0]*i[1])/(i[0]+i[1]) for i in PR],label="F1",color="y")
plt.title("Content-Based P/R/F1")
plt.xlabel("N")
plt.ylabel("Ratio")
plt.grid(True)
plt.legend()
plt.savefig("./graph/Content_based-PRF1_of_N.png")
plt.show()

