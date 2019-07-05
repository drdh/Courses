# 1

| doc/term | 2010 | 世博会 | 中国 | 举行 | 2005 | 1970 | 日本 |
| -------- | ---- | ------ | ---- | ---- | ---- | ---- | ---- |
| $d_1$    | lg 2 | 0      | lg 2 | 0    | 0    | 0    | 0    |
| $d_2$    | 0    | 0      | 0    | 0    | lg 2 | lg 2 | lg 2 |
| $q$      | lg 2 | 0      | lg 2 | 0    | 0    | 0    | 0    |

$$
Sim(q,d_1)=0.707 \\
sim(q,d_2)=0
$$

说明$q$与$d_1$更相关

# 2

- 无法描述词项之间的关系，词项之间并不完全独立，词语之间的关系可能实际上影响文档的相关性
- 需要扫描所有的文档才能计算

# 3

## 算法思路

### 基本假设1

一个热门的微博会被很多活跃用户转发或发

### 基本假设2

一个活跃用户会转发或发很多热门微博

## 伪代码

```pseudocode
//m weibis, n users and their posting relationships
//post[i][j]==1 means user[i] posting or reposting weibo[j]
input: post[n][m]
output: hot level, user[n],weibo[m]

//initialize
for i=1 to n:
	user[i]=1
for j=1 to m:
	weibo[j]=1

//iteration
repeat 
	//procedure I
	for j=1 to m:
		for i=1 to n:
			if post[i][j]==1:
				weibo[j]+=user[i]
	//procedure O
	for i=1 to n:
		for j=1 to m:
			if post[i][j]==1
				user[i]+=weibo[j]
until user and weibo converge
```





