from keras.models import load_model
from keras.utils import to_categorical
from collections import Counter
import numpy as np

md=load_model("../model/dynamic_model.h5")

ID_MAP_DICT={}
ID_MAP_DICT2={}
ID_MAP_TOTAL=1

SEQLEN=5
NAME_TYPE=32

CUR_MAPPD_ID=1

DELTA_COUNTER=Counter()

CUR_DELTA=[1]*5


# 输入为int
# 输出为当前预测的int
# 输出说明：　0: 表示没有预测成功
#           其他: 表示当前预测的int,而且此int 之前一定出现过
# 例如: ypred=dynamic_predict(12)

def dynamic_predict(id_in):
    global ID_MAP_DICT,ID_MAP_TOTAL,ID_MAP_DICT2,md

    id_map=1
    if id_in in ID_MAP_DICT:
        id_map=ID_MAP_DICT[id_in]
    else:
        ID_MAP_DICT[id_in]=ID_MAP_TOTAL
        ID_MAP_DICT2[ID_MAP_TOTAL]=id_in
        id_map=ID_MAP_TOTAL
        ID_MAP_TOTAL+=1

    global CUR_MAPPD_ID,DELTA_COUNTER,SEQLEN,NAME_TYPE,CUR_DELTA
    delta=id_map-CUR_MAPPD_ID
    CUR_MAPPD_ID=id_map
    DELTA_COUNTER[delta]+=1

    delta_map={}
    delta_map2={}
    total=1
    for i in DELTA_COUNTER.most_common(31):
        delta_map[i[0]]=total
        delta_map2[total]=i[0]
        total+=1

    cur_delta=[]
    for i in CUR_DELTA:
        if i in delta_map:
            cur_delta.append(delta_map[i])
        else:
            cur_delta.append(0)

    CUR_DELTA=CUR_DELTA[1:]
    CUR_DELTA.append(delta)

    if delta in delta_map:
        delta=delta_map[delta]
    else:
        delta=0


    x=to_categorical([cur_delta],num_classes=NAME_TYPE)
    y=to_categorical([delta],num_classes=NAME_TYPE)

    md.fit(x,y,batch_size=1,epochs=1,verbose=0)


    cur_delta=[]
    for i in CUR_DELTA:
        if i in delta_map:
            cur_delta.append(delta_map[i])
        else:
            cur_delta.append(0)

    x=to_categorical([cur_delta],num_classes=NAME_TYPE)

    pred=md.predict(x,verbose=0)[0]
    ypred=np.argmax(pred)


    if ypred in delta_map2:
        temp=delta_map2[ypred]+CUR_MAPPD_ID
        if temp in ID_MAP_DICT2:
            return ID_MAP_DICT2[temp]

        else: return 0
    else: return 0
