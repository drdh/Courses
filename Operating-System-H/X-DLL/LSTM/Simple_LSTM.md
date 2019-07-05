** 注：本节内容来自《Deep Learning with Keras》 **

** 阅读本节之前请先阅读[搭建keras](./搭建keras.md) **
## SimpleRNN with Keras -- Generating text
#### 下载文本数据
[Alice in Wonderland](http://www.gutenberg.org/files/11/11-0.txt)
```
wget http://www.gutenberg.org/files/11/11-0.txt
```
并且放在 /data/alice_in_wonderland

#### 目标
在《爱丽丝梦游仙境》的基础上，训练一个基于字符的语言模型，给10个之前的字符，预测下一个字符，之所以基于字符而不是基于基于单词，是因为字符有更小的词汇表，训练起来更加容易。

```
jupyter notebook
```

#### import necessary modules
```
from keras.layers import Dense,Activation
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
from keras.utils import plot_model
import numpy as np
```

#### 做一些清除工作
```
fin=open("../data/alice_in_wonderland.txt","rb")
lines=[]
for line in fin:
    line=line.strip().lower()
    line=line.decode("ascii","ignore")
    if len(line)==0:
        continue
    lines.append(line)
fin.close()
text=" ".join(lines)
```

#### 构建字符/数字映射
```
chars=set([c for c in text])
nb_chars=len(chars)
char2index=dict((c,i) for i,c in enumerate(chars))
index2char=dict((i,c) for i,c in enumerate(chars))
```

#### 构建输入数据
input_chars为字符串数组，每个字符串长10，label_char为字符数组，两者结合起来表示，前面10个字符，来预测后一个字符
```
SEQLEN=10
STEP=1

input_chars=[]
label_chars=[]
for i in range(0,len(text)-SEQLEN,STEP):
    input_chars.append(text[i:i+SEQLEN])
    label_chars.append(text[i+SEQLEN])
```

#### 输入数据向量化
使用one-hot编码
```
X=np.zeros((len(input_chars),SEQLEN,nb_chars),dtype=np.bool)
y=np.zeros((len(input_chars),nb_chars),dtype=np.bool)
for i,input_char in enumerate(input_chars):
    for j,ch in enumerate(input_char):
        X[i,j,char2index[ch]]=1
    y[i,char2index[label_chars[i]]]=1
```

#### 构建模型
```
HIDDEN_SIZE=128
BATCH_SIZE=128
NUM_ITERATIONS=25
NUM_EPOTCHS_PER_ITERATION=1
NUM_PREDS_PER_EPOCH=100

model=Sequential()
model.add(SimpleRNN(HIDDEN_SIZE,return_sequences=False,input_shape=(SEQLEN,nb_chars),unroll=True))
model.add(Dense(nb_chars))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy",optimizer="rmsprop")
```

#### 模型可视化
在这之前，需要先安装pydot和graphviz
```
pip3 install pydot
sudo apt install graphviz
```
然后才能输出
```
plot_model(model,to_file="SimpleRNN.png",show_shapes=True)
```
![SimpleRNN](img/SimpleRNN.png)

#### 训练预测
```
for iteration in range(NUM_ITERATIONS):
    print("=" *50)
    print("Iteration #: %d"%(iteration))
    model.fit(X,y,batch_size=BATCH_SIZE,epochs=NUM_EPOCHS_PER_ITERATION)

    test_idx=np.random.randint(len(input_chars))
    test_chars=input_chars[test_idx]
    print("Gerationg from seed: %s"%(test_chars))
    print(test_chars,end="")
    for i in range(NUM_PREDS_PER_EPOCH):
        Xtest=np.zeros((1,SEQLEN,nb_chars))
        for i,ch in enumerate(test_chars):
            Xtest[0,i,char2index[ch]]=1
        pred=model.predict(Xtest,verbose=0)[0]
        ypred=index2char[np.argmax(pred)]
        print(ypred,end="")
        test_chars=test_chars[1:]+ypred
    print('\n')
```
这里的训练方式不同于普通的，外部循环相当于训练的epoch，内部的循环是训练一次之后进行预测的次数，预测不影响下一次的训练。这样设置的目的是，方便观察每次训练的结果。

#### 结果

截取最后三次训练的结果如下
```
==================================================
Iteration #: 22
Epoch 1/1
158773/158773 [==============================] - 13s 82us/step - loss: 1.3097
Gerationg from seed:  and down
 and down the caterpillar the baby what i should think the beginning to sinder that the mouse to the other the

==================================================
Iteration #: 23
Epoch 1/1
158773/158773 [==============================] - 11s 68us/step - loss: 1.3083
Gerationg from seed: to win, th
to win, the dormouse she had got to the poor little shriek a right to the white rabbit the caterpillar. alice

==================================================
Iteration #: 24
Epoch 1/1
158773/158773 [==============================] - 11s 70us/step - loss: 1.3066
Gerationg from seed: , we went
, we went on, what a moment the mock turtle read of the work or the work or the work or the work or the work o
```


## LSTM with Keras -sentiment analysis
#### 目的
many-to-one RNN

#### 数据
使用kaggle上面的数据，首先安装kaggle
```
pip3 install kaggle
```
然后注册账户等等，[参考资料](https://github.com/Kaggle/kaggle-api)
> To use the Kaggle API, sign up for a Kaggle account at https://www.kaggle.com. Then go to the 'Account' tab of your user profile (https://www.kaggle.com/<username>/account) and select 'Create API Token'. This will trigger the download of kaggle.json, a file containing your API credentials. Place this file in the location ~/.kaggle/kaggle.json

> For your security, ensure that other users of your computer do not have read access to your credentials. On Unix-based systems you can do this with the following command:

> chmod 600 ~/.kaggle/kaggle.json

下载之前，需要先接受规则。下载到本地目录
```
kaggle competitions download -c si650winter11 -p .
```


#### import
```
from keras.layers.core import Activation,Dense,Dropout,SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
from keras.utils import plot_model
```

#### 探索分析
先下载一个包
```
import nltk
nltk.download("punkt")
```

```
maxlen=0
word_freqs=collections.Counter()
num_recs=0
ftrain=open(os.path.join("../data","umich-sentiment-test.txt"),'rb')
for line in ftrain:
    label,sentence =line.strip().split(b'\t')
    words=nltk.word_tokenize(sentence.decode("ascii","ignore").lower())
    if(len(words)>maxlen):
        maxlen=len(words)
    for word in words:
        word_freqs[word]+=1
    num_recs+=1
ftrain.close()
```

得到maxlen=42,len(word_freqs)=2313

进行设置
```
MAX_FEATURES=2000
MAX_SENTENCE_LENGTH=40
```

#### 建立查找表
其中也设置了空项
```
vocab_size=min(MAX_FEATURES,len(word_freqs))+2
word2index={x[0]:i+2 for i,x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"]=0
word2index["unk"]=1
index2word={v:k for k,v in word2index.items()}
```

#### 构建input
```
X=np.empty((num_recs,),dtype=list)
y=np.zeros((num_recs,))
i=0
ftrain=open(os.path.join("../data","umich-sentiment-test.txt"),'rb')
for line in ftrain:
    label,sentence =line.strip().split(b'\t')
    words=nltk.word_tokenize(sentence.decode("ascii","ignore").lower())
    seqs=[]
    for word in words:
        if word in word2index:
            seqs.append(word2index[word])
        else:
            seqs.append(word2index["UNK"])
    X[i]=seqs
    y[i]=int(label)
    i+=1
ftrain.close()
X=sequence.pad_sequences(X,maxlen=MAX_SENTENCE_LENGTH)
```
分割数据
```
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=42)
```

#### 构建模型
```
EMBEDDING_SIZE=128
HIDDEN_LAYER_SIZE=64
BATCH_SIZE=32
NUM_EPOCHS=10

model=Sequential()
model.add(Embedding(vocab_size,EMBEDDING_SIZE,input_length=MAX_SENTENCE_LENGTH))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(HIDDEN_LAYER_SIZE,dropout=0.2,recurrent_dropout=0.3))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
```
输出图形
```
plot_model(model,to_file="LSTM_Sentiment.png",show_shapes=True)
```
![LSTM_Sentiment](./img/LSTM_Sentiment.png)

#### 训练
```
history=model.fit(Xtrain,ytrain,batch_size=BATCH_SIZE,epochs=NUM_EPOCHS,validation_data=(Xtest,ytest))
```

#### 结果
```
VTrain on 5668 samples, validate on 1418 samples
Epoch 1/10
5668/5668 [==============================] - 12s 2ms/step - loss: 0.2474 - acc: 0.8904 - val_loss: 0.0657 - val_acc: 0.9781
Epoch 2/10
5668/5668 [==============================] - 11s 2ms/step - loss: 0.0294 - acc: 0.9899 - val_loss: 0.0509 - val_acc: 0.9774
Epoch 3/10
5668/5668 [==============================] - 10s 2ms/step - loss: 0.0107 - acc: 0.9975 - val_loss: 0.0396 - val_acc: 0.9873
Epoch 4/10
5668/5668 [==============================] - 11s 2ms/step - loss: 0.0075 - acc: 0.9977 - val_loss: 0.0521 - val_acc: 0.9838
Epoch 5/10
5668/5668 [==============================] - 11s 2ms/step - loss: 0.0044 - acc: 0.9981 - val_loss: 0.0478 - val_acc: 0.9887
Epoch 6/10
5668/5668 [==============================] - 11s 2ms/step - loss: 0.0055 - acc: 0.9988 - val_loss: 0.0758 - val_acc: 0.9795
Epoch 7/10
5668/5668 [==============================] - 11s 2ms/step - loss: 0.0013 - acc: 0.9996 - val_loss: 0.0550 - val_acc: 0.9887
Epoch 8/10
5668/5668 [==============================] - 11s 2ms/step - loss: 5.8718e-04 - acc: 0.9998 - val_loss: 0.0559 - val_acc: 0.9901
Epoch 9/10
5668/5668 [==============================] - 11s 2ms/step - loss: 0.0025 - acc: 0.9996 - val_loss: 0.0526 - val_acc: 0.9866
Epoch 10/10
5668/5668 [==============================] - 11s 2ms/step - loss: 0.0024 - acc: 0.9995 - val_loss: 0.0525 - val_acc: 0.9845
```

作图
```
plt.subplot(211)
plt.title("Accuracy")
plt.plot(history.history["acc"],color="g",label="Train")
plt.plot(history.history["val_acc"],color="b",label="Validation")
plt.legend(loc="best")

plt.subplot(212)
plt.title("Loss")
plt.plot(history.history["loss"],color="g",label="Train")
plt.plot(history.history["val_loss"],color="b",label="Validation")
plt.legend(loc="best")

plt.tight_layout()
plt.savefig("LSTM_Sentiment_plot.png")
```
![LSTM_Sentiment_plot](./img/LSTM_Sentiment_plot.png)

#### 测试
```
score,acc=model.evaluate(Xtest,ytest,batch_size=BATCH_SIZE)
print("Test score : %.3f, Accuracy : %.3f"%(score,acc))

for i in range(5):
    idx=np.random.randint(len(Xtest))
    xtest=Xtest[idx].reshape(1,40)
    ylabel=ytest[idx]
    ypred=model.predict(xtest)[0][0]
    sent=" ".join([index2word[x] for x in xtest[0].tolist() if x!=0])
    print("%.0f %d %s"%(ypred,ylabel,sent))
```

```
1418/1418 [==============================] - 0s 294us/step
Test score : 0.053, Accuracy : 0.984
0 0 is it just me , or does harry potter suck ? ...
1 1 i love being a sentry for mission impossible and a station for bonkers .
0 0 da vinci code = up , up , down , down , left , right , left , right , b , a , suck !
1 1 i love harry potter .
1 1 he 's like , 'yeah i got acne and i love brokeback mountain '..
```

## GRU with Keras -- POS tagging
#### 目的
对英语词汇进行标记,many-to-many,句子对应一组标记

#### 数据
```
import nltk
nltk.download()
d treebank
```

#### import
```
from keras.layers.core import Activation,Dense,Dropout,RepeatVector,SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
import collections
import nltk
import numpy as np
import os
```

#### download data
```
fedata=open(os.path.join("../data","treebank_sents.txt"),"w")
ffdata=open(os.path.join("../data","treebank_poss.txt"),"w")

sents=nltk.corpus.treebank.tagged_sents()
for sent in sents:
    words,poss=[],[]
    for word,pos in sent:
        if pos =="-NONE-":
            continue
        words.append(word)
        poss.append(pos)
    fedata.write("{}\n".format(" ".join(words)))
    ffdata.write("{}\n".format(" ".join(poss)))
fedata.close()
ffdata.close()
```

#### 探索
```
def parse_sentences(filename):
    word_freqs=collections.Counter()
    num_recs,maxlen=0,0
    fin=open(filename,"rb")
    for line in fin:
        words=line.strip().lower().split()
        for word in words:
            word_freqs[word]+=1
        if(len(words)>maxlen):
            maxlen=len(words)
        num_recs+=1
    fin.close()
    return word_freqs,maxlen,num_recs

s_wordfreqs,s_maxlen,s_numrecs=parse_sentences(os.path.join("../data","treebank_sents.txt"))
t_wordfreqs,t_maxlen,t_numrecs=parse_sentences(os.path.join("../data","treebank_poss.txt"))
print(len(s_wordfreqs),s_maxlen,s_numrecs)
print(len(t_wordfreqs),t_maxlen,t_numrecs)
```
输出为
```
10947 249 3914
45 249 3914
```

#### 构建索引
```
MAX_SEQLEN=250
S_MAX_FEATURES=5000
T_MAX_FEATURES=45

s_vocabsize=min(len(s_wordfreqs),S_MAX_FEATURES)+2
s_word2index={x[0]:i+2 for i,x in enumerate(s_wordfreqs.most_common(S_MAX_FEATURES))}
s_word2index["PAD"]=0
s_word2index["UNK"]=1
s_index2word={v:k for k,v in s_word2index.items()}

t_vocabsize=len(t_wordfreqs)+1
t_word2index={x[0]:i for i,x in enumerate(t_wordfreqs.most_common(T_MAX_FEATURES))}

t_word2index["PAD"]=0
t_index2word={v:k for k,v in t_word2index.items()}
```

#### 构建input
其中tag使用了one-hot编码
```
def build_tensor(filename,numrecs,word2index,maxlen,make_categorical=False,num_classes=0):
    data=np.empty((numrecs,),dtype=list)
    fin=open(filename,"r")
    i=0
    for line in fin:
        wids=[]
        for word in line.strip().lower().split():
            if word in word2index:
                wids.append(word2index[word])
            else:
                wids.append(word2index["UNK"])
        if make_categorical:
            data[i]=np_utils.to_categorical(wids,num_classes=num_classes)
        else:
            data[i]=wids
        i+=1
    fin.close()
    pdata=sequence.pad_sequences(data,maxlen=maxlen)
    return pdata

X=build_tensor(os.path.join("../data","treebank_sents.txt"),s_numrecs,s_word2index,MAX_SEQLEN)
Y=build_tensor(os.path.join("../data","treebank_poss.txt"),t_numrecs,t_word2index,MAX_SEQLEN,True,t_vocabsize)
```
分割数据
```
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.2,random_state=42)
```

#### 构建模型
```
EMBED_SIZE=128
HIDDEN_SIZE=64
BATCH_SIZE=32
NUM_EPOCHS=1

model=Sequential()
model.add(Embedding(s_vocabsize,EMBED_SIZE,input_length=MAX_SEQLEN))
model.add(SpatialDropout1D(0.2))
model.add(GRU(HIDDEN_SIZE,dropout=0.2,recurrent_dropout=0.2))
model.add(RepeatVector(MAX_SEQLEN))
model.add(GRU(HIDDEN_SIZE,return_sequences=True))
model.add(TimeDistributed(Dense(t_vocabsize)))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
```
输出
```
plot_model(model,to_file="GRU_POS.png",show_shapes=True)
```
![GRU_POS](./img/GRU_POS.png)

#### 训练
```
model.fit(Xtrain,Ytrain,batch_size=BATCH_SIZE,epochs=NUM_EPOCHS,validation_data=[Xtest,Ytest])
```
结果为
```
Train on 3131 samples, validate on 783 samples
Epoch 1/1
3131/3131 [==============================] - 43s 14ms/step - loss: 0.3016 - acc: 0.7295 - val_loss: 0.2929 - val_acc: 0.9157
```
```
score,acc=model.evaluate(Xtest,Ytest,batch_size=BATCH_SIZE)
print(score,acc)
```
```
783/783 [==============================] - 2s 3ms/step
0.2928801686294843 0.9157496759261208
```

#### 使用Bidirectional
```
from keras.layers.wrappers import Bidirectional

model=Sequential()
model.add(Embedding(s_vocabsize,EMBED_SIZE,input_length=MAX_SEQLEN))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(GRU(HIDDEN_SIZE,dropout=0.2,recurrent_dropout=0.2)))
model.add(RepeatVector(MAX_SEQLEN))
model.add(Bidirectional(GRU(HIDDEN_SIZE,return_sequences=True)))
model.add(TimeDistributed(Dense(t_vocabsize)))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

plot_model(model,to_file="Bidirectional_POS.png",show_shapes=True)

model.fit(Xtrain,Ytrain,batch_size=BATCH_SIZE,epochs=NUM_EPOCHS,validation_data=[Xtest,Ytest])

score,acc=model.evaluate(Xtest,Ytest,batch_size=BATCH_SIZE)
print(score,acc)
```
![Bidirectional_POS](./img/Bidirectional_POS.png)
此处使用GRU的Bidirectional比使用LSTM的Bidirectional有更好的准确率.
单独GRU也比LSTM有更高的准确率

## Stateful LSTM with Keras -- predicting electricity consumption
#### 目的
使用带有状态的LSTM预测用户用电量

####　数据
```
https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
```

#### 探索
```
import numpy as np
import matplotlib.pyplot as plt
import os
import re

fld=open(os.path.join("../data","LD2011_2014.txt"),"r")
data=[]
cid=250
while True:
    line=fld.readline()
    if not line:
        break
    if line.startswith('"";'):
        continue
    cols=[float(re.sub(",",".",x)) for x in line.strip().split(";")[1:]]
    data.append(cols[cid])
```

```
NUM_ENTRIES=1000
plt.plot(range(NUM_ENTRIES),data[0:NUM_ENTRIES])
plt.ylabel("electricity consumption")
plt.xlabel("time (1pt = 15 mins)")
plt.savefig("Electricity_Consumption.png")

np.save(os.path.join("../data","LD_250.npy"),np.array(data))
```

![Electricity_Consumption](./img/Electricity_Consumption.png)

#### import
```
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
import os
from keras.utils import plot_model
```

#### 读取数据
```
data=np.load(os.path.join("../data","LD_250.npy"))
data=data.reshape(-1,1)
scaler=MinMaxScaler(feature_range=(0,1),copy=False)
data=scaler.fit_transform(data)
```

```
NUM_TIMESTEPS=20
HIDDEN_SIZE=10
BATCH_SIZE=96

X=np.zeros((data.shape[0],NUM_TIMESTEPS))
Y=np.zeros((data.shape[0],1))
for i in range(len(data)-NUM_TIMESTEPS-1):
    X[i]=data[i:i+NUM_TIMESTEPS].T
    Y[i]=data[i+NUM_TIMESTEPS+1]
X=np.expand_dims(X,axis=2)
```

#### 分割数据
```
sp=int(0.7*len(data))
Xtrain,Xtest,Ytrain,Ytest=X[0:sp],X[sp:],Y[0:sp],Y[sp:]
```

#### 训练
```
#stateless
NUM_EPOCHS=5
model=Sequential()
model.add(LSTM(HIDDEN_SIZE,input_shape=(NUM_TIMESTEPS,1),return_sequences=False))
model.add(Dense(1))

model.compile(loss="mean_squared_error",optimizer="adam",metrics=["mean_squared_error"])

model.fit(Xtrain,Ytrain,epochs=NUM_EPOCHS,batch_size=BATCH_SIZE,validation_data=(Xtest,Ytest))
```

```
#stateful
NUM_EPOCHS=5
model=Sequential()
model.add(LSTM(HIDDEN_SIZE,stateful=True,batch_input_shape=(BATCH_SIZE,NUM_TIMESTEPS,1),return_sequences=False))
model.add(Dense(1))

model.compile(loss="mean_squared_error",optimizer="adam",metrics=["mean_squared_error"])

train_size=(Xtrain.shape[0] // BATCH_SIZE ) * BATCH_SIZE
test_size=(Xtest.shape[0] // BATCH_SIZE) * BATCH_SIZE
Xtrain,Ytrain=Xtrain[0:train_size],Ytrain[0:train_size]
Xtest,Ytest=Xtest[0:test_size],Ytest[0:test_size]

for i in range(NUM_EPOCHS):
    print("Epoch {:d}/{:d}".format(i+1,NUM_EPOCHS))
    model.fit(Xtrain,Ytrain,batch_size=BATCH_SIZE,epochs=1,validation_data=(Xtest,Ytest),shuffle=False)
    model.reset_states()
```
