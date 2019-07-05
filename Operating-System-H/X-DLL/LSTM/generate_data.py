
# coding: utf-8

# In[89]:


import numpy as np
import string
import random


# In[52]:


c=string.printable[0:36]
char2index={i:j for i,j in zip(c,range(37))}
index2char={j:i for i,j in char2index.items()}
names=['data','pic','text','paper','salary']
for name in names:
    print(name,end=' ')
    for c in name:
        print(char2index[c],end=' ')
    print('\n')


# In[102]:


name=[]
id=[]
extension=[]
directory=[]
size=[]
protection=[]
owner=[]
created=[]
modified=[]
access=[]
operation=[]

data_name=[]
for i in range(1,501):
    data_name.append([13,10,29,10,i])
    name.append([13,10,29,10,i])
    id.append(i)
    extension.append(0)
    directory.append([1,1,i//100,(i-i//100*100)//10,i-i//10*10])
    size.append(random.randint(450,500))
    protection.append(0)
    owner.append(0)
    created.append([18,6,2,i%24])
    modified.append([18,6,2,i%24])
    access.append([18,6,2,i%24])
    operation.append(0)

pic_name=[]
for i in range(1,1000,2):
    pic_name.append([25,18,12,i])
    name.append([25,18,12,i])
    id.append(i)
    extension.append(1)
    directory.append([2,3,i//100,(i-i//100*100)//10,i-i//10*10])
    size.append(random.randint(900,920))
    protection.append(1)
    owner.append(i%5)
    created.append([18,6,i%30+1,0])
    modified.append([18,6,i%30+1,0])
    access.append([18,6,i%30+1,0])
    operation.append(0)

text_name=[]
for i in range(1,1000,3):
    text_name.append([29,14,33,29,i])
    name.append([29,14,33,29,i])
    id.append(i)
    extension.append(2)
    directory.append([4,i//100,(i-i//100*100)//10,i-i//10*10,0])
    size.append(random.randint(100,110))
    protection.append(2)
    owner.append(i%5)
    created.append([18,6,i%30+1,0])
    modified.append([18,6,i%30+1,0])
    access.append([18,6,i%30+1,0])
    operation.append(1)

paper_name=[]
for i in range(1,1000,4):
    paper_name.append([25,10,25,21,10,27,i])
    name.append([25,10,25,21,10,27,i])
    id.append(i)
    extension.append(3)
    directory.append([9,i//100,(i-i//100*100)//10,i-i//10*10,0])
    size.append(random.randint(800,900))
    protection.append(2)
    owner.append(i%5)
    created.append([18,i%12+1,2,0])
    modified.append([18,i%12+1,2,0])
    access.append([18,i%12+1,2,0])
    operation.append(1)

salary_name=[]
for i in range(1,1000,5):
    salary_name.append([28,10,21,10,27,34,1])
    name.append([28,10,21,10,27,34,1])
    id.append(i)
    extension.append(4)
    directory.append([i//100,(i-i//100*100)//10,i-i//10*10,0,0])
    size.append(random.randint(10,13))
    protection.append(1)
    owner.append(i%5)
    created.append([i%19,6,2,0])
    modified.append([i%19,6,2,0])
    access.append([i%19,6,2,0])
    operation.append(0)


# In[114]:


name_n=np.array(name)
id_n=np.array(id)
extension_n=np.array(extension)
directory_n=np.array(directory)
size_n=np.array(size)
protection_n=np.array(protection)
owner_n=np.array(owner)
created_n=np.array(created)
modified_n=np.array(modified)
access_n=np.array(access)
operation_n=np.array(operation)

name_n.dump("../data/generate_data/name")
id_n.dump("../data/generate_data/id")
extension_n.dump("../data/generate_data/extension")
directory_n.dump("../data/generate_data/directory")
size_n.dump("../data/generate_data/size")
protection_n.dump("../data/generate_data/protection")
owner_n.dump("../data/generate_data/owner")
created_n.dump("../data/generate_data/created")
modified_n.dump("../data/generate_data/modified")
access_n.dump("../data/generate_data/access")
operation_n.dump("../data/generate_data/operation")
