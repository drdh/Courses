

`bank.sql`

[数据类型](<https://www.runoob.com/sqlite/sqlite-data-types.html>)统一如下

```
id: integer
money: real
date: text
other: text
```

触发器设计

```sqlite
create trigger insert_saving after insert on saving_account
begin
insert into account("Account_id","branch_name")
	values (new.Account_id,new.branch_name);
end;

create trigger insert_check after insert on check_account
begin
insert into account("Account_id","branch_name")
	values (new.Account_id,new.branch_name);
end;

create trigger delete_saving after delete on saving_account
begin
delete from account 
	where Account_id=old.Account_id;
end;

create trigger delete_check after delete on check_account
begin
delete from account 
	where Account_id=old.Account_id;
end;
```



使用`bank.sql`生成数据库初始表


```bash
rm test.db -rf
sqlite3 test.db < ../bank.sql
```

进入数据库

```
sqlite3 test.db
.tables
.schema
```

导出数据库(可以每次更新的时候都导出，可以一键生成数据库)

```bash
sqlite3 test.db .dump > test.sql
#sqlite3 test.db < test.sql
```


格式化输出

```sqlite
.header on
.mode column
--.timer on
```

开启外键约束

```sqlite
PRAGMA foreign_keys=ON;
```



插入数据

```sqlite
--见test_data_generator.ipynb
```

更新数据

```sqlite
update branch
	set asset=23456
	where branch_name="South_Bank";
```












