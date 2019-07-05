import os
from dynamic_predict import dynamic_predict
from multiprocessing import Pipe,Process,Queue

#0表示下载，1表示上传
def getFilename(command):
	cmd = command.split()
	if cmd[2] == "-get":
		return (0,cmd[3],cmd[4])
	elif cmd[2] == "-put":
		return (1,cmd[3],cmd[4])
	else: return (-1,None,None)

def addToMap(name,name_id_map,id_name_map):
	if name_id_map[name] != None:
		return
	length = len(name_id_map)
	name_id_map[name] = length + 1
	id_name_map[length + 1] = name

def nameToId(name,name_id_map):
	return name_id_map[name]

def idToName(id,id_name_map):
	return id_name_map(id)

def check_here(src,dst):		   #检测本地临时文件夹是否已经有这个文件
	p = src.split("/")
	length = len(p)
	nm = p[length - 1]
	if os.path.exists(tmp_here + nm) == False:
		return False
	else:
		os.system("mv " + tmp_here + nm + " " + dst)

def check_hdfs(src,dst):		   #检测HDFS临时文件夹是否已经有这个文件
	p = src.split("/")
	length = len(p)
	nm = p[length - 1]
	result = os.popen(path + "hadoop fs -ls " + tmp_hdfs)
	if nm in result:
		os.system(path + "hadoop fs -mv " + tmp_hdfs + " " + dst)
	else:
		return False

def pre_download(q_download):				   #还可以再加一个检测CPU利用率的函数，决定是将该进程休眠还是读取队列
	while True:
		name = q_download.get()
		if name == None:
			exit(0)
		cmd = path + "hadoop fs -get " + name + " " + tmp_here
		os.system(cmd)

def pre_upload(q_upload):
	while True:
		print("upload")
		name = q_upload.get()	
		if name == None:
			exit(0)
			cmd = path + "hadoop fs -put " + name + " " + tmp_here
		os.system(cmd)

def predict(p):					 #调用预测模块，单独作为一个进程，以提高性能
	name_id_map = {}
	id_name_map = {}
	q_download = Queue()
	p_download = Process(target=pre_download, args=(q_download,))
	p_download.start()			  #创建下载进程
	
	q_upload = Queue()
	print("123")
	p_upload = Process(target=pre_upload(), args=(q_upload,))
	p_upload.start()				# 创建上传进程
	
	while True:
		print("predict")
		src = p.recv()
		
		if src == None:
			q_download.put(None)
			q_upload.put(None)
			exit(0)

		addToMap(src,name_id_map,id_name_map)			   # 将当前文件路径加到map中
		id_in = nameToId(src,name_id_map)	   # 当前文件的id
		id = dynamic_predict(id_in)  # 预测得到要预取或预存的文件的id
		if id == 0:  # 预测失败
			continue
		else:
			name = idToName(id,id_name_map)		 # 要预取或预存的文件名
			if mode == 0:			   # 预取或预存到临时文件夹
				q_download.put(name)
			else:
				q_upload.put(name)

path = "/home/linan/hadoop-2.7.6/bin/"
tmp_here = "/home/linan/tmp/"
tmp_hdfs = "/tmp/"

parent_side,child_side = Pipe()
p = Process(target=predict,args=(child_side,))
p.start()

while True:
	command = input()
	if command == "exit":
		parent_side.send(None)
		break
	elif command[0:6] == "hadoop":			  #是hadoop相关的命令
		command = path + command
		result = getFilename(command)		 #如果是上传或者下载命令，获取源文件路径
		mode = result[0]
		src = result[1]
		dst = result[2]
		if mode == 0:					#下载命令，检查本地临时文件夹是否已有该文件
			if check_here(src,dst) == True:
				continue
		if mode == 1:					#上传命令，检查HDFS临时文件夹是否已有该文件
			if check_hdfs(src,dst) == True:
				continue

		print(os.popen(command).read())		 #执行该命令，并打印出结果
		if mode == -1:
			continue
		else:								   #下面是预测部分
			parent_side.send(src)			   #将文件名传递给预测进程
	else:
		continue
