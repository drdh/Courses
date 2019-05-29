

<https://github.com/ajayrandhawa/PyQt-Sqlite-Project-CURD>

<https://github.com/mbaser/pyqt-sqlite>



<https://www.yiibai.com/sqlite/insert-query.html>

<http://www.runoob.com/sqlite/sqlite-update.html>



<https://pythonspot.com/pyqt5/>

<http://code.py40.com/pyqt5/18.html>





​		





oracle apex account

用户名：dh_129@163.com

密码: drdhapex

workspace: BANK_DH





oracle account

dh_129@163.com

drdhApex2





环境变量设置

![1556780896633](README.assets/1556780896633.png)

按[设置](<https://tutel.me/c/dba/questions/211436/can+not+install+oracle+apex+on+gnulinux+db+12c>)

`validate_con_names: ORCLPDB is not open`

```plsql
SHUTDOWN IMMEDIATE;
STARTUP UPGRADE;
ALTER PLUGGABLE DATABASE ALL OPEN UPGRADE;
select name,open_mode from v$pdbs; --MIGRATE
```



按[Oracle Apex 5.0安装教程](<https://blog.csdn.net/sunansheng/article/details/74196149>)

```
sqlplus /nolog

@apxremov.sql
 
@apexins.sql sysaux sysaux temp /i/

@apex_epg_config.sql D:\exe\apex_5.1.1_en

alter user anonymous account unlock;

@apxconf.sql
username:ADMIN [default/return]
email:[empty/return]
passward:abc#123ABC
```

浏览器打开`http://localhost:8080/apex/apex_admin`

ADMIN/abc#123ABC