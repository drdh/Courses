 
# ## Improved

# 基于Collaborative Filter

# In[120]:


#需要用到最上面的数据
print(train_matrix.shape,test_matrix.shape,rating_matrix.shape)


# In[121]:


movies=np.zeros((movie_num+1,3),dtype="<100U")
with open("./data/ml-1m/movies.dat",encoding = "ISO-8859-1") as f:
    line=f.readline()
    while line:
        temp=line.strip().split("::")
        movies[int(temp[0]),:]=temp
        line=f.readline()


# In[122]:


users=np.zeros((user_num+1,5),dtype="<20U")
with open("./data/ml-1m/users.dat",encoding = "ISO-8859-1") as f:
    line=f.readline()
    while line:
        temp=line.strip().split("::")
        users[int(temp[0]),:]=temp
        line=f.readline()


# In[123]:


genre={"Action":0,
"Adventure":1,
"Animation":2,
"Children's":3,
"Comedy":4,
"Crime":5,
"Documentary":6,
"Drama":7,
"Fantasy":8,
"Film-Noir":9,
"Horror":10,
"Musical":11,
"Mystery":12,
"Romance":13,
"Sci-Fi":14,
"Thriller":15,
"War":16,
"Western":17
      }

movie_profile=np.zeros((movie_num+1,18))
for i in range(movie_num+1):
    temp=movies[i,2].split('|')
    for g in temp:
        if g in genre:
            movie_profile[i,genre[g]]=1


# In[124]:


user_profile=np.zeros((test_user_size,18))
for i in range(test_user_size):
    user_profile[i]=np.sum(movie_profile.T*train_matrix[i],axis=1)


# In[125]:


#逐渐变化alpha得到一个图
ex_pred=np.zeros((test_user_size,test_movie_size))
alpha=0.01
for x in range(1,test_user_size):
    if x%100 == 0:
        print(x)
    for i in range(1,test_movie_size):
        norm=np.linalg.norm(movie_profile[i])*np.linalg.norm(user_profile[x])
        if norm != 0:
            ex_pred[x,i]=0
            index=np.arange(movie_num+1)[train_matrix[x].astype("bool")]
            Nix=index[np.argsort(movie_sim[i,train_matrix[x].astype("bool")])[-k:]]
        
            sumij=movie_sim[i,Nix].sum()
            
            if sumij!=0:
                ex_pred[x,i]=bxi[x,i]+(1-alpha)*(movie_sim[i,Nix]@(train_matrix[x,Nix]-bxi[x,Nix]))/sumij+alpha*((movie_profile[i]@user_profile[x])/(norm))*5


# In[137]:


#result_matrix=np.zeros((test_user_size,test_movie_size))
#for i in range(test_user_size):
#    for j in range(test_movie_size):
#        if test_matrix[i,j]!=0:
#            result_matrix[i,j]=ex_pred[i,j]-test_matrix[i,j]
result_matrix2=ex_pred.copy()
result_matrix2[test_matrix==0]=0
result_matrix=result_matrix2-test_matrix


# In[138]:


np.linalg.norm(result_matrix)/np.count_nonzero(test_matrix) 


# In[139]:


np.sum(np.absolute(result_matrix))/np.count_nonzero(test_matrix)


# In[ ]:


# RMSE 0.010776504401721263
# MAE 0.8694166710904666
# P 0.46589403973509935 
# R 0.47469635627530365


# In[140]:


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


# In[141]:


#逐渐变化alpha得到一个图
#alpha=0.01
k=5 #表示最邻近的k个点。由于是基于Collaborative Filter
def EX_PRED(alpha):
    print(alpha)
    ex_pred=np.zeros((test_user_size,test_movie_size))
    
    for x in range(1,test_user_size):
        #if x%100 == 0:
        #    print(x)
        for i in range(1,test_movie_size):
            norm=np.linalg.norm(movie_profile[i])*np.linalg.norm(user_profile[x])
            if norm != 0:
                ex_pred[x,i]=0
                index=np.arange(movie_num+1)[train_matrix[x].astype("bool")]
                Nix=index[np.argsort(movie_sim[i,train_matrix[x].astype("bool")])[-k:]]

                sumij=movie_sim[i,Nix].sum()

                if sumij!=0:
                    ex_pred[x,i]=bxi[x,i]+(1-alpha)*(movie_sim[i,Nix]@(train_matrix[x,Nix]-bxi[x,Nix]))/sumij+alpha*((movie_profile[i]@user_profile[x])/(norm))*5
            
        
    #result_matrix=np.zeros((test_user_size,test_movie_size))
    #for i in range(test_user_size):
    #    for j in range(test_movie_size):
    #        if test_matrix[i,j]!=0:
    #            result_matrix[i,j]=ex_pred[i,j]-test_matrix[i,j]
    result_matrix2=ex_pred.copy()
    result_matrix2[test_matrix==0]=0
    result_matrix=result_matrix2-test_matrix
    
    RMSR=np.linalg.norm(result_matrix)/np.count_nonzero(test_matrix) 
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

    print(RMSR,MAE,precision,recall)
    return RMSR,MAE,precision,recall


# In[142]:


# RMSE 0.010776504401721263
# MAE 0.8694166710904666
# P 0.46589403973509935 
# R 0.47469635627530365
alpha=np.linspace(0,1,26)
result=[EX_PRED(a) for a in alpha]


# In[143]:


result2=[EX_PRED(a) for a in np.linspace(0,0.08,21)]


# In[144]:


result3=[EX_PRED(a) for a in np.linspace(0,0.04,21)]


# In[180]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(1)
plt.subplot(3,3,1)
plt.plot(np.linspace(0,1,26),[i[0] for i in result],label="RMSR",color='r')
plt.title('RMSR')
plt.grid(True)


plt.subplot(3,3,2)
plt.plot(np.linspace(0,1,26),[i[1] for i in result],label="MAE",color='g')
plt.title('MAE')
plt.grid(True)

plt.subplot(3,3,3)
plt.plot(np.linspace(0,1,26),[i[2] for i in result],label="P",color='g')
plt.plot(np.linspace(0,1,26),[i[3] for i in result],label="R",color='r')
plt.title('P & R')
plt.grid(True)


plt.subplot(3,3,4)
plt.plot(np.linspace(0,0.08,21),[i[0] for i in result2],label="RMSR",color='r')
plt.grid(True)

plt.subplot(3,3,5)
plt.plot(np.linspace(0,0.08,21),[i[1] for i in result2],label="MAE",color='g')
plt.grid(True)

plt.subplot(3,3,6)
plt.plot(np.linspace(0,0.08,21),[i[2] for i in result2],label="P",color='g')
plt.plot(np.linspace(0,0.08,21),[i[3] for i in result2],label="R",color='r')
#plt.title('P & R')
plt.grid(True)

plt.subplot(3,3,7)
plt.plot(np.linspace(0,0.04,21),[i[0] for i in result3],label="RMSR",color='r')
plt.grid(True)
plt.xlabel("alpha")

plt.subplot(3,3,8)
plt.plot(np.linspace(0,0.04,21),[i[1] for i in result3],label="MAE",color='g')
plt.grid(True)
plt.xlabel("alpha")

plt.subplot(3,3,9)
plt.plot(np.linspace(0,0.04,21),[i[2] for i in result3],label="P",color='g')
plt.plot(np.linspace(0,0.04,21),[i[3] for i in result3],label="R",color='r')
#plt.title('P & R')
plt.xlabel("alpha")
plt.grid(True)


plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
plt.savefig("./graph/PartII-1-Extension.png",dpi=1000)
plt.show()

