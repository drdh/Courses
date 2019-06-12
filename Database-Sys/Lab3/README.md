Bugs

- [x] responsible/depositor不能删除,设置了relation的column不能查找，与relation设置相关

  > 去掉所有的`setRelation`. 因为这的确是个麻烦项，似乎设计的目的只是for view

- [x] depositor不能插入，因为account表没有更新

  > 使用触发器解决

- [x] 其他的需求可能可以使用query直接用sql语句或者视图解决

  > 使用触发器解决

- [x] statistics设计

  > 使用了多种图表示

- [x] 设计添加管理员界面/mainWindow，美化主界面

  > 添加管理界面
  >
  > 添加toolbar/tips
  >
  > 添加背景图片

- [ ] 按dialog分成不同的文件存放



ref

<https://github.com/ajayrandhawa/PyQt-Sqlite-Project-CURD>

<https://github.com/mbaser/pyqt-sqlite>

<https://github.com/skinex/CRUD-SQLite->



[易百sqlite3](https://www.yiibai.com/sqlite/insert-query.html)

[runoob sqlite3](http://www.runoob.com/sqlite/sqlite-update.html)



[PyQt5 中文教程](http://code.py40.com/pyqt5/18.html)

[PyQt5 Doc](https://doc.bccnsoft.com/docs/PyQt5)

[PyQt Chart](<https://doc.qt.io/qt-5/qtcharts-index.html>)

[PyQt界面编程应用与实践](https://www.cnblogs.com/jinjiangongzuoshi/p/5636960.html)

[QSqlQuery](<https://blog.csdn.net/baidu_33570760/article/details/71740554>)

[PyQt 测试案例](https://github.com/PyQt5/PyQt)

<https://pythonspot.com/pyqt5/>

<https://maicss.gitbooks.io/pyqt5/content/>

<http://shouce.jb51.net/qt-beginning/>



安装与使用

```bash
pip install PyQtChart --user
python main.py
#superuser: lx
```



- Main_Window
- Administer
  - Login_Dialog
  - Register
- Infrastructure
  - Branch_Dialog
  - Employee_Dialog
- Client
  - Customer_Dialog
  - Responsible_Dialog
- Transaction
  - Account
    - Saving_Dialog
    - Check_Dialog
    - Depositor_Dialog
  - Debt
    - Loan_Dialog
    - Payment_Dialog
    - Borrow_Dialog
- Statistics
  - Overall_Dialog
  - Seasonly_Dialog
  - Monthly_Dialog

​		



Others

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