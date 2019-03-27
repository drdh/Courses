先从Host上下载相应的文件，解压，然后网络调整到Host-Only,使用Lab1的共享传输文件。

需要[修改权限](<https://blog.csdn.net/tingyuyiye01/article/details/49903297>)

```cmd
net share helloc=C: /grant:win,full
```

然后在host上

```cmd
net use Y: \\192.168.86.101\helloc
```









winwinpass