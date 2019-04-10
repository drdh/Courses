安装`oracle12c`,`pl/sql developer`, `power designer`

为了便于复现，请传到rec上

`pl/sql developer`在网上找[注册码](https://www.cnblogs.com/shizilukou123/p/9149358.html)即可

[Power Designer 破解](https://www.fujieace.com/software/powerdesigner.html)



用户名oracle

passwd: drdhoracle



全局数据库名: orcl

口令P: drdhoracle



sql plus中

用户名: system

passwd: drdhoracle





[带外键的change](<https://blog.csdn.net/u014030117/article/details/46333751>)





Procedure

```plsql
Execute ChangeBookID('b10','b12');
select * from book;
```

Trigger

```plsql
insert into borrow(book_ID,reader_ID,borrow_date)values('b3','r1',to_date('01-03-2019','dd-mm-yy'));

select * from book;

update borrow set return_date=to_date('01-03-2019','dd-mm-yy') where reader_ID='r2' and book_id='b3';

select * from book;
```



实体完整性

```plsql
insert into book(ID,name,author)values('b9','fun','drdh');
insert into book(name,author)values('fun','drdh');--error
```

参照完整性

```plsql
delete from book where ID='b1';
```

用户自定义完整性

```plsql
insert into book(ID,author)values('b9','drdh');
```



