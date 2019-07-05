# NachOS 
### Install
```
$ sudo apt install build-essential gcc-multilib g++-multilib
(build.linux/ make clean & make depend
 test make clean)

$ wget http://mll.csie.ntu.edu.tw/course/os_f08/assignment/nachos_40.tar.gz 

$ wget http://mll.csie.ntu.edu.tw/course/os_f08/assignment/mips-decstation.linux-xgcc.gz

$ wget http://mll.csie.ntu.edu.tw/course/os_f08/assignment/nachos-gcc.diff

$ tar -zxvf nachos_40.tar.gz
$ tar zxvf mips-decstation.linux-xgcc.gz

nachos $ patch -p0 < nachos-gcc.diff
nashos/NachOS-4.0/code/build.linux $ make depend
```

Change all the "IsInList()" of "lib/list.cc" into "this->IsInList()"

Vim Search

```
nashos/NachOS-4.0/code/build.linux $ make
```

```
nashos/NachOS-4.0/code/build.linux $ ./nashos
```

```
nashos/NachOS-4.0/coff2noff $ make
```

```
nachos/NachOS-4.0/code/test $ make
nachos/NachOS-4.0/code/test $ ../build.linux/nachos -x halt
```


### lab 1
-   
