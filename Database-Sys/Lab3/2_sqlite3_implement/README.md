

`bank.sql`

[数据类型](<https://www.runoob.com/sqlite/sqlite-data-types.html>)统一如下

```
id: integer
money: real
date: text
other: text
```

使用`bank.sql`生成数据库


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

导出数据库

```
sqlite3 data/test.db .dump > data/test.sql
```







```

```

