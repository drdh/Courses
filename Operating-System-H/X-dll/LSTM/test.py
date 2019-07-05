# 请使用如下函数测试你选择的语言调用python
#　python3

import numpy as np

a_global=0
b_global=[]
c_global=[]

def count(a,b,c):
    global a_global
    global b_global
    global c_global
    a_global+=a
    b_global.append(b)
    c_global.append(c)

    n=np.array(b_global)

    return (a_global,n.mean(),max(c_global))


# 调用结果
# 此处全局变量有作用
# import test
# test.count(1,2,[1,2])
# (1, 2.0, [1, 2])
# test.count(1,3,[2,2])
# (2, 2.5, [2, 2])
# test.count(1,4,[1,3])
# (3, 3.0, [2, 2])
