## 使用`nmap`扫描端口

需要先按照文档配置好网络，host-only, 然后ubuntu和win都需要使用adapter (#1)的网络

win下使用`ipconfig`来查看网络。

ubuntu下使用`ifconfig`来查看网络。

需要关闭win下的网络防火墙才能被ping通。

在ubuntu下使用`nmap 192.168.86.101`



## net命令的使用

在win下使用

```powershell
net user win win /add
net share helloc=C:
net share helloc /del
```

host下使用

```cmd
net use Y: \\192.168.86.101\helloc
#win win
net use Y: /del
```



如果出现不安全的信息，在启用或关闭Windows功能 中打开`SMB 1.0/CIFS 文件共享支持`的`客户端`