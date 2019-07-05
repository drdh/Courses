# 仅仅只提供了借口，具体的实现正在补充
# 以下构建的函数借口不一定是最终版本，可能会删掉或者加上一些输入输出，因此在使用接口的时候请考虑可拓展性
# 参数更加详细的解释请看本目录README
# 为了传参方便，使用了整数与整数列表
# 以下均为python3,但是暂时并不需要耗费时间学习，只需要明白接口的作用。
# 以下全局变量在每次调用函数的时候均会改变
# 如果使用java或者其他非python语言 请先用本目录下的test.py进行练习，以确保是带状态的调用
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np

name_global=[]
#id_global=[]
#extension_global=[]
#directory_global=[]
#size_global=[]
#protection_global=[]
#owner_global=[]
created_global=[]
modified_global=[]
access_global=[]
#operation_global=[]

def network(name,id,extension,directory,size,protection,owner,created,modified,access,operation):
# name　整数列表，最大为10个字符，只考虑a-z(大小写不敏感),０－９,其他的自动忽略
#       0-9:0-9, a-z:10-35
#       例如: abc3 编码为name=[10,11,12,3]
# id 整数，文件的唯一identifier, 范围为0-999
#    例如: id=36
# extension 整数，0-19,文件拓展名，UNK:0 doc:1 pdf:2 jpg:3 docx:4 html:5 txt:6 xls:7 png:8 gif:9
#                         avi:10 md:11 c:1 cc:13 java:14 py:15 tex:16 rar:17 tar:18 zip:19
#　　　　　　　例如:　extension=16
# directory　整数列表，且元素个数最多为５，每个元素范围0-9,表示最多５层目录，每层目录最多10个子目录
# 　　　　　　例如: /src/local/network.py     directory=[1,4]假设src编码为１，local编码为４
# size 整数，1-1024，表示小文件
#      例如: size=34
# protection 整数，executing 0，reading 1，writting 2
#            例如: protection=2 表示可写文件
# owner 整数，０－4
# 　　　 例如　owner=4
# created 整数列表，元素为４个
#         例如: 2018/6/2 17:07 created=[18,6,2,17]
# modified 同上
# access 同上
# operation 整数0-1,read 0,write 1
#           例如: operation=0 表示当前操作为读。



# 以下返回值是类型实例
# 均返回整数或者整数列表
# 返回均为模式的增量
# 比如，id增量为１
    name_return=[0,0,0,1,2,0,0,0,0,0]
    id_return=id_predict(id) # return delta id
    extension_return=extension_predict(extension) #return next extension
    directory_return=directory_predict(directory)
    size_return=size_predict(size) # return delta size
    protection_return=protection_predict(protection) # return next protection
    owner_return=owner_predict(owner) # return next owner
    created_return=[0,0,0,0]
    modified_return=[0,0,0,1]
    access_return=[0,0,1,0]
    operation_return=operation_predict(operation) # return next operation

    return (name_return,
    id_return,
    extension_return,
    directory_return,
    size_return,
    protection_return,
    owner_return,
    created_return,
    modified_return,
    access_return,
    operation_return)

# import network
# a=network.network([10,11,12,3],36,16,[1,4],34,0,4,[18,6,2,17],[18,6,2,7],[18,6,2,7],0)
# print(a)
# ([0, 0, 0, 1, 2, 0, 0, 0, 0, 0],
# 1,
# 0,
# [0, 0, 1, 0, 0],
# 0,
# 0,
# 0,
# [0, 0, 0, 0],
# [0, 0, 0, 1],
# [0, 0, 1, 0],
# 1)

# id: int 0-999
id_global=[0]*10
id_model=load_model("../model/id_SimpleRNN.h5")
def id_predict(id):
    global id_global,id_model
    id_global=id_global[1:]
    id_global.append(id)
    y=np.array(id_global)
    y=y.reshape(1,10,1)
    y=np.divide(y,1000)
    pred=id_model.predict(y,verbose=0)[0,0]
    return int(pred*1000)


#extension: int 0-19
extension_global=[0]*10 #10个初始值，每次都是平移
extension_model=load_model("../model/extension_SimpleRNN.h5")
def extension_predict(extension):
    global extension_global,extension_model
    extension_global=extension_global[1:]
    extension_global.append(extension)
    y=to_categorical([extension_global],num_classes=20)
    pred=extension_model.predict(y,verbose=0)[0]
    return np.argmax(pred)

# directory: int list [1,2,3,4,5],0-9
directory_global=[[0]*5]*10
directory_model=load_model("../model/directory_LSTM.h5")
def directory_predict(directory):
    global directory_global,directory_model
    directory_global=directory_global[1:]
    directory_global.append(np.array(directory)-np.array(directory_global[8]))
    #y=to_categorical(directory_global,num_classes=10)
    y=np.array(directory_global)
    y=y.reshape(1,10,5)
    pred=directory_model.predict(y,verbose=0)[0]
    #p=np.argmax(pred)
    #return [p//10000,p//1000-p//10000*10,p//100-p//1000*10,p//10-p//100*10,p-p//10*10]
    ret=[]
    for i in pred:
        ret.append(int(i))
    return ret

#size: int 1-1024
size_global=[0]*10
size_model=load_model("../model/size_SimpleRNN.h5")
def size_predict(size):
    global size_global,size_model
    size_global=size_global[1:]
    size_global.append(size)
    y=np.array(size_global)
    y=y.reshape(1,10,1)
    y=np.divide(y,1024)
    pred=size_model.predict(y,verbose=0)[0,0]
    return int(pred*1024)


# protection: int 0,1,2
protection_global=[0]*10
protection_model=load_model("../model/protection_SimpleRNN.h5")
def protection_predict(protection):
    global protection_global,protection_model
    protection_global=protection_global[1:]
    protection_global.append(protection)
    y=to_categorical([protection_global],num_classes=3)
    pred=protection_model.predict(y,verbose=0)[0]
    return np.argmax(pred)


# owner: int 0,1,2,3,4
owner_global=[0]*10
owner_model=load_model("../model/owner_SimpleRNN.h5")
def owner_predict(owner):
    global owner_global,owner_model
    owner_global=owner_global[1:]
    owner_global.append(owner)
    y=to_categorical([owner_global],num_classes=5)
    pred=owner_model.predict(y,verbose=0)[0]
    return np.argmax(pred)

# operation: int 0,1
operation_global=[0]*10
operation_model=load_model("../model/operation_SimpleRNN.h5")
def operation_predict(operation):
    global operation_global,operation_model
    operation_global=operation_global[1:]
    operation_global.append(operation)
    y=to_categorical([operation_global],num_classes=2)
    pred=operation_model.predict(y,verbose=0)[0]
    return np.argmax(pred)
