import os
from dynamic_predict import dynamic_predict

#0表示下载，1表示上传
def getFilename(command):
    cmd = command.split()
    if cmd[2] == "-get":
        return (0,cmd[3],cmd[4])
    elif cmd[2] == "-put":
        return (1,cmd[3],cmd[4])
    else: return (-1,None,None)

def addToMap(name):
	if name_id_map[name] == None:
		return
    length = len(name_id_map)
    name_id_map[name] = length + 1
    id_name_map[length + 1] = name

def nameToId(name):
    return name_id_map[name]

def idToName(id):
    return id_name_map(id)

def check_here(src,dst):           #检测本地临时文件夹是否已经有这个文件
    p = src.split("/")
    length = len(p)
    nm = p[length - 1]
    if os.path.exists(tmp_here + nm) == False:
        return False
    else:
        os.system("mv " + tmp_here + nm + " " + dst)

def check_hdfs(src,dst):           #检测HDFS临时文件夹是否已经有这个文件
    p = src.split("/")
    length = len(p)
    nm = p[length - 1]
    result = os.popen(path + "hadoop fs -ls " + tmp_hdfs)
    if nm in result:
        os.system(path + "hadoop fs -mv " + tmp_hdfs + " " + dst)
    else:
        return False

name_id_map = {}
id_name_map = {}
path = "/home/linan/hadoop-2.7.6/bin/"
tmp_here = "/home/linan/tmp/"
tmp_hdfs = "/tmp/"
while True:
    command = input()
    if command == "exit":
        break
    elif command[0:6] == "hadoop":              #是hadoop相关的命令
        command = path + command
        result = getFilename(command)         #如果是上传或者下载命令，获取源文件路径
        mode = result[0]
        src = result[1]
        dst = result[2]
        if mode == 0:                    #下载命令，检查本地临时文件夹是否已有该文件
            if check_here(src,dst) == True:
                continue
        if mode == 1:                    #上传命令，检查HDFS临时文件夹是否已有该文件
            if check_hdfs(src,dst) == True:
                continue

        print(os.popen(command).read())         #执行该命令，并打印出结果
        if mode == -1:
            continue
        else:                                   #下面是预测部分
            addToMap(src)                       # 将当前文件路径加到map中
            id_in = nameToId(src)               #当前文件的id
            print(id_in)
            id = dynamic_predict(id_in)         #预测得到要预取或预存的文件的id
            if id == 0:                         #预测失败
                continue
            else:
                name = idToName(id)             #要预取或预存的文件名
                if mode == 0:                   #预取或预存到临时文件夹
                    cmd = path + "hadoop fs -get " + name + " " + tmp_here
                    os.system(cmd)
                else:
                    cmd = path + "hadoop fs -put " + name + " " + tmp_hdfs
                    os.system(cmd)
    else:
        continue
