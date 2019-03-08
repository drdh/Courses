[manjaro中安装virtualbox](https://wiki.manjaro.org/index.php?title=VirtualBox)

查看版本

```bash
uname -r
```

安装

```bash
sudo pacman -S virtualbox
```

将VirtualBox Module安装进内核，可以用重启实现，也可以用下面的命令

```bash
sudo vboxreload
```

然后从[该网站](https://www.oracle.com/technetwork/server-storage/virtualbox/downloads/index.html)下载Extension

然后在虚拟机的Preference里面添加Extension

安装好的Ubuntu可以通过更改Displays=>Resolution来扩大屏幕