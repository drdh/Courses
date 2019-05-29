

`bank.sql`

[数据类型](<https://www.runoob.com/sqlite/sqlite-data-types.html>)统一如下

```
id: integer
money: real
date: text
other: text
```

使用`bank.sql`生成数据库初始表


```bash
rm data/test.db -rf
sqlite3 data/test.db < bank.sql
```

进入数据库

```
sqlite3 data/test.db
.tables
.schema
```

导出数据库(可以每次更新的时候都导出，可以一键生成数据库)

```bash
sqlite3 data/test.db .dump > data/test.sql
#sqlite3 data/test.db < data/test.sql
```


格式化输出

```sqlite
.header on
.mode column
--.timer on
```

插入数据

```sqlite
insert into branch (branch_name,city,asset)
	values ("North_Bank","North",12345);
	
insert into branch (branch_name,city,asset)
	values ("South_Bank","South",23456);
	
insert into branch (branch_name,city,asset)
	values ("East_Bank","East",34567);

insert into branch (branch_name,city,asset)
	values ("West_Bank","West",45678);

/*	
insert into branch 
	values ("North_Bank","North",12345);
*/
```

更新数据

```sqlite
update branch
	set asset=23456
	where branch_name="South_Bank";
```












