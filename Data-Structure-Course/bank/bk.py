from ctypes import *
import os
bk2=cdll.LoadLibrary(os.getcwd()+'/bk2.so')

def bank(closetime,time,Q1,Q2,Total,TotalNum,TotalTime):
    c_time=(c_int * 400)(*time)
    c_Q1=(c_int * 200)(*Q1)
    c_Q2=(c_int *200)(*Q2)
    c_Total=(c_int *400)(*Total)
    c_TotalNum=(c_int *400)(*TotalNum)
    c_TotalTime=(c_int *400)(*TotalTime)
    return bk2.Bank(closetime,c_time,c_Q1,c_Q2,c_Total,c_TotalNum,c_TotalTime),c_time,c_Q1,c_Q2,c_Total,c_TotalNum,c_TotalTime
    

