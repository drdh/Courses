

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
rmtest.db -rf
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

插入数据

```sqlite
-- branch
insert into branch (branch_name,city,asset)
	values ("North_Bank","North",12345);
	
insert into branch (branch_name,city,asset)
	values ("South_Bank","South",23456);
	
insert into branch (branch_name,city,asset)
	values ("East_Bank","East",34567);

insert into branch (branch_name,city,asset)
	values ("West_Bank","West",45678);

-- customer
insert into customer (customer_id,name,phone,address,contact_name,contact_phone,contact_email,contact_relation)
values 
(1,"ab","123456","ustc","cd","234567","ab@mail.ustc.edu.cn","friends");

insert into customer (customer_id,name,phone,address,contact_name,contact_phone,contact_email,contact_relation)
values 
(2,"cd","234567","ustc","ef","234567","cd@mail.ustc.edu.cn","parents");

insert into customer (customer_id,name,phone,address,contact_name,contact_phone,contact_email,contact_relation)
values 
(3,"ef","345678","ustc","ab","123456","ef@mail.ustc.edu.cn","other");

-- employee
insert into employee
(employee_id,branch_name,manager_id,name,address,phone,start_date)
values
(1,"North_Bank",2,"a","ustc","123","2019-04-25 11:25:13.333");

insert into employee
(employee_id,branch_name,manager_id,name,address,phone,start_date)
values
(2,"North_Bank",2,"b","ustc","234","2019-04-24 11:25:13.333");
```

更新数据

```sqlite
update branch
	set asset=23456
	where branch_name="South_Bank";
```












