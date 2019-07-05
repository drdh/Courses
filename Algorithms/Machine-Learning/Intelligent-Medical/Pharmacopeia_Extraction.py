import csv
import os
import re
files=os.listdir("./药典/")

count=0
with open("Extraction.csv","w") as csvfile: 
    writer = csv.writer(csvfile)
    writer.writerow(["index","药名","症状","不明症状","用法","每次用量","每日用量","症状原文","用法用量原文"])
    for file in files:
        with open("./药典/"+file,"r") as f:
            #print(count,file)
            line=""
            origin_sym=""
            origin_use=""
            while True:
                line=f.readline()
                origin_sym=line
                if not line:
                    break
                if "功能与主治" in line:
                    break
            
            #到这里分割出"【功能与主治】"行
            count+=1
            name=file.split(".")[0]
            symptom=""
            unclear_symptom=""
            #表意不清症状
            unclear=[""]
            usage=""
            dosage1=""
            dosage2=""
            
            
            if line !="" and line:
                #print(line.strip())
                line=line.strip().split("。")[1].replace("用于","")#此处分割后只有后面的"用于"开头的语句
                line=line.split("；")[0] # "；"后面是"慢性鼻炎、过敏鼻炎、鼻窦炎见上述证候者", 类似的，除去
                line=line.split("，")#症状整体使用"，"分隔，内部可能涉及有“、”分隔
                for i in range(len(line)):
                    line[i]=re.sub(r'.*所致',"",line[i]) #去除各个症状中间"所致的"部分
                    line[i]=re.sub(r'.*的',"",line[i])#去除原因
                    line[i]=line[i].replace("症见","") #去除“症见”关键词
                    line[i]=line[i].replace("或","、") #"或"表达的是并列关系
                    line[i]=line[i].replace("及","、") #同上
                    line[i]=line[i].replace("和","、")
                    line[i]=line[i].replace(" ","") #去除内部空格
                    
                    line[i]=line[i].strip("、：")#去除首尾无关紧要的符号
                    
                    #以下内容是否去除,不明。单列一项，便于之后的处理                    
                    # 此症状需要上述所有症状的信息，不可单独罗列，故提取
                    if "上述证候者" in line[i] or( re.search(r'证$',line[i]) and not re.search(r"淋证$",line[i]))\
                            or re.search("者$",line[i]) \
                            or re.search(r'证者$',line[i]) \
                            or re.search(r"证候者$",line[i]) \
                            or re.search("乳房肿块",line[i]) \
                            or re.search("乳房结节",line[i]) \
                            or re.search("乳房结块",line[i]) \
                            or re.search("Ⅰ、Ⅱ",line[i]) \
                            or re.search("I、II",line[i])\
                            or re.search("【功能与主治】",line[i])\
                            or re.search("【用法与用量】口服",line[i]):
                            
                        unclear.append(line[i])
                            
                    
                if(len(unclear)>0):
                    for un in unclear:
                        if un in line:
                            line.remove(un)
                    #print("unclear",end=" ") #显示那些标示为unclear的项
                    #print(unclear)
                
                
                #第二趟扫描
                for i in range(len(line)):
                    #由"，"分割后，有很大部分是有"、"来分割的，部分并不能去除
                    noComma=line[i].split("、")
                    if(len(noComma)>1):
                        # 分割后如果有字数为1的项，一定不合理
                        noComma_ok=True
                        for item in noComma:
                            if len(item)==1:
                                unclear.append(line[i])
                                #print("noComma-WRONG",end="") #查看去除”、“后不合理的项
                                #print(line[i])
                                noComma_ok=False
                                break
                        if noComma_ok:
                            line.pop(i)
                            line.extend(noComma)
                            #for item in noComma.reverse():
                            #    line.insert(i,item)
                            #line[i]=noComma
                            #print("noComma-OK",end="") #查看其他的去除"、"是否可行
                            #print(line[i])
                #第二次去除            
                if(len(unclear)>0):
                    for un in unclear:
                        if un in line:
                            line.remove(un)
                    #print("unclear",end=" ") #显示那些标示为unclear的项
                    #print(unclear)    
                
                symptom=" ".join(line)
                unclear_symptom=" ".join(unclear)
                #print(symptom)
                #print(unclear_symptom)
                
                
                #以上均为处理【症状】
                #以下处理【用法与用量】
                line=f.readline()
                line=f.readline()
                
                
                if re.search(r"【.*用法.*】",line):
                    origin_use=line
                    line=re.sub(r"【.*用法.*】","",line).strip("。 \n")
                    if "口服" in line:
                        #usage.insert(0,"口服")
                        usage="口服"
                        line=line.replace("口服","")
                    elif "开水冲服" in line:
                        #usage.insert(0,"开水冲服")
                        usage="开水冲服"
                        line=line.replace("开水冲服","")
                    elif "外用" in line:
                        #usage.insert(0,"外用")
                        usage="外用"
                        line=line.replace("外用。","")
                    elif "外用适量" in line:
                        #usage.insert(0,"外用适量")
                        usage="外用适量"
                        line=line.replace("外用适量","")
                    
                    
                    line=line.replace(" ","")
                    pattern1=re.compile(r'(一次)?([0-9\.]*～[0-9\.]*|[0-9\.]*)(ml|mL|g|袋|片|丸|粒|块|揿|滴|锭|贴|枚)')
                    pattern2=re.compile(r'(—日|一日|每日)([0-9\.]*(～|~|-)[0-9\.]*|[0-9\.]*)次')
                    result1=pattern1.findall(line)
                    line=re.sub(pattern1,"",line)
                    result2=pattern2.findall(line)
                    line=re.sub(pattern2,"",line)                    
                    
                    print(usage)
                    print(result1,end="")
                    print(result2,end="")
                    line=line.strip("。，；")
                    print(line)
                    #if len(result1)==0 and len(result2)==0:
                    #    print(line)
                    
                    #writer.writerow(["index","药名","症状","不明症状","用法","每次用量","每日用量","症状原文","用法用量原文"])
                    if(len(result1)!=0):
                        for item in result1:
                            if item[1]!="":
                                dosage1=dosage1+"".join(item)+";"
                    
                    if(len(result2)!=0):
                        for item in result2:
                            if item[1]!="":
                                dosage2=dosage2+"".join(item[0:2])+"次;"    
                        
                    
                    
            writer.writerow([count,name,symptom,unclear_symptom,usage,dosage1,dosage2,origin_sym,origin_use])
                        
                    
                
            #print(count,name,content)
            #writer.writerow([count,name,content,usage])
            
