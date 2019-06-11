PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "Account" 
(
   "Account_id"         integer               not null,
   "branch_name"        text                  not null,
--   "balance"            real,
--   "open_date"          text,
   constraint PK_Account primary key ("Account_id")
--   constraint FK_ACCOUNT_OPEN_ACCO_BRANCH foreign key ("branch_name")
--      references "Branch" ("branch_name")
);
INSERT INTO Account VALUES(0,'South_Bank');
INSERT INTO Account VALUES(1,'North_Bank');
INSERT INTO Account VALUES(2,'East_Bank');
INSERT INTO Account VALUES(3,'South_Bank');
INSERT INTO Account VALUES(4,'West_Bank');
INSERT INTO Account VALUES(5,'West_Bank');
INSERT INTO Account VALUES(6,'West_Bank');
INSERT INTO Account VALUES(7,'South_Bank');
INSERT INTO Account VALUES(8,'South_Bank');
INSERT INTO Account VALUES(9,'North_Bank');
INSERT INTO Account VALUES(10,'West_Bank');
INSERT INTO Account VALUES(11,'North_Bank');
INSERT INTO Account VALUES(12,'South_Bank');
INSERT INTO Account VALUES(13,'South_Bank');
INSERT INTO Account VALUES(14,'South_Bank');
INSERT INTO Account VALUES(15,'East_Bank');
INSERT INTO Account VALUES(16,'North_Bank');
INSERT INTO Account VALUES(17,'South_Bank');
INSERT INTO Account VALUES(18,'North_Bank');
INSERT INTO Account VALUES(19,'West_Bank');
INSERT INTO Account VALUES(20,'North_Bank');
INSERT INTO Account VALUES(21,'East_Bank');
INSERT INTO Account VALUES(22,'South_Bank');
INSERT INTO Account VALUES(23,'North_Bank');
INSERT INTO Account VALUES(24,'North_Bank');
INSERT INTO Account VALUES(25,'West_Bank');
INSERT INTO Account VALUES(26,'North_Bank');
INSERT INTO Account VALUES(27,'East_Bank');
INSERT INTO Account VALUES(28,'West_Bank');
INSERT INTO Account VALUES(29,'East_Bank');
INSERT INTO Account VALUES(30,'East_Bank');
INSERT INTO Account VALUES(31,'West_Bank');
INSERT INTO Account VALUES(32,'South_Bank');
INSERT INTO Account VALUES(33,'West_Bank');
INSERT INTO Account VALUES(34,'North_Bank');
INSERT INTO Account VALUES(35,'West_Bank');
INSERT INTO Account VALUES(36,'South_Bank');
INSERT INTO Account VALUES(37,'South_Bank');
INSERT INTO Account VALUES(38,'West_Bank');
INSERT INTO Account VALUES(39,'East_Bank');
INSERT INTO Account VALUES(40,'West_Bank');
INSERT INTO Account VALUES(41,'South_Bank');
INSERT INTO Account VALUES(42,'East_Bank');
INSERT INTO Account VALUES(43,'North_Bank');
INSERT INTO Account VALUES(44,'South_Bank');
INSERT INTO Account VALUES(45,'East_Bank');
INSERT INTO Account VALUES(46,'East_Bank');
INSERT INTO Account VALUES(47,'East_Bank');
INSERT INTO Account VALUES(48,'North_Bank');
INSERT INTO Account VALUES(49,'East_Bank');
INSERT INTO Account VALUES(50,'East_Bank');
INSERT INTO Account VALUES(51,'North_Bank');
INSERT INTO Account VALUES(52,'East_Bank');
INSERT INTO Account VALUES(53,'South_Bank');
INSERT INTO Account VALUES(54,'East_Bank');
INSERT INTO Account VALUES(55,'West_Bank');
INSERT INTO Account VALUES(56,'North_Bank');
INSERT INTO Account VALUES(57,'South_Bank');
INSERT INTO Account VALUES(58,'West_Bank');
INSERT INTO Account VALUES(59,'East_Bank');
INSERT INTO Account VALUES(60,'East_Bank');
INSERT INTO Account VALUES(61,'East_Bank');
INSERT INTO Account VALUES(62,'West_Bank');
INSERT INTO Account VALUES(63,'West_Bank');
INSERT INTO Account VALUES(64,'North_Bank');
INSERT INTO Account VALUES(65,'West_Bank');
INSERT INTO Account VALUES(66,'South_Bank');
INSERT INTO Account VALUES(67,'East_Bank');
INSERT INTO Account VALUES(68,'South_Bank');
INSERT INTO Account VALUES(69,'South_Bank');
INSERT INTO Account VALUES(70,'West_Bank');
INSERT INTO Account VALUES(71,'West_Bank');
INSERT INTO Account VALUES(72,'North_Bank');
INSERT INTO Account VALUES(73,'South_Bank');
CREATE TABLE IF NOT EXISTS "Branch" 
(
   "branch_name"        text                 not null,
   "city"               text,
   "asset"              real,
   constraint PK_BRANCH primary key ("branch_name")
);
INSERT INTO Branch VALUES('North_Bank','North',12345.0);
INSERT INTO Branch VALUES('South_Bank','South',23456.0);
INSERT INTO Branch VALUES('East_Bank','East',34567.0);
INSERT INTO Branch VALUES('West_Bank','West',45678.0);
CREATE TABLE IF NOT EXISTS "Check_Account" 
(
   "Account_id"         integer               not null,
   "branch_name"        text                 not null,
   "balance"            real,
   "open_date"          text     "2019-6-11 19:23:45",
   "overdraft"          real,
   constraint PK_CHECK_ACCOUNT primary key ("Account_id"),
   constraint FK_CHECK_AC_Account_IN_Account foreign key ("Account_id")
      references "Account" ("Account_id"),
   constraint FK_Check_ACCOUNT_OPEN_ACCO_BRANCH foreign key ("branch_name")
      references "Branch" ("branch_name")
);
INSERT INTO Check_Account VALUES(37,'South_Bank',1.0,'2019-9-12 12:23:34',145.0);
INSERT INTO Check_Account VALUES(38,'West_Bank',0.0,'2019-1-12 12:23:34',100.99999999999999999);
INSERT INTO Check_Account VALUES(39,'East_Bank',56.999999999999999998,'2019-11-12 12:23:34',167.99999999999999999);
INSERT INTO Check_Account VALUES(40,'West_Bank',51.999999999999999998,'2019-2-12 12:23:34',176.99999999999999999);
INSERT INTO Check_Account VALUES(41,'South_Bank',40.0,'2019-10-12 12:23:34',140.99999999999999999);
INSERT INTO Check_Account VALUES(42,'East_Bank',24.0,'2019-3-12 12:23:34',120.0);
INSERT INTO Check_Account VALUES(43,'North_Bank',100.0,'2019-3-12 12:23:34',132.0);
INSERT INTO Check_Account VALUES(44,'South_Bank',73.000000000000000001,'2019-9-12 12:23:34',123.0);
INSERT INTO Check_Account VALUES(45,'East_Bank',65.0,'2019-8-12 12:23:34',163.0);
INSERT INTO Check_Account VALUES(46,'East_Bank',75.999999999999999999,'2019-9-12 12:23:34',175.99999999999999999);
INSERT INTO Check_Account VALUES(47,'East_Bank',53.000000000000000001,'2019-11-12 12:23:34',120.0);
INSERT INTO Check_Account VALUES(48,'North_Bank',76.999999999999999998,'2019-7-12 12:23:34',114.99999999999999999);
INSERT INTO Check_Account VALUES(49,'East_Bank',100.0,'2019-5-12 12:23:34',190.99999999999999999);
INSERT INTO Check_Account VALUES(50,'East_Bank',73.000000000000000001,'2019-4-12 12:23:34',130.99999999999999999);
INSERT INTO Check_Account VALUES(51,'North_Bank',59.0,'2019-1-12 12:23:34',196.0);
INSERT INTO Check_Account VALUES(52,'East_Bank',27.0,'2019-8-12 12:23:34',100.99999999999999999);
INSERT INTO Check_Account VALUES(53,'South_Bank',48.000000000000000001,'2019-12-12 12:23:34',116.99999999999999999);
INSERT INTO Check_Account VALUES(54,'East_Bank',53.000000000000000001,'2019-12-12 12:23:34',117.99999999999999999);
INSERT INTO Check_Account VALUES(55,'West_Bank',32.0,'2019-9-12 12:23:34',175.0);
INSERT INTO Check_Account VALUES(56,'North_Bank',13.999999999999999999,'2019-2-12 12:23:34',142.99999999999999999);
INSERT INTO Check_Account VALUES(57,'South_Bank',50.0,'2019-3-12 12:23:34',167.99999999999999999);
INSERT INTO Check_Account VALUES(58,'West_Bank',20.0,'2019-9-12 12:23:34',198.99999999999999999);
INSERT INTO Check_Account VALUES(59,'East_Bank',88.999999999999999996,'2019-4-12 12:23:34',127.99999999999999999);
INSERT INTO Check_Account VALUES(60,'East_Bank',85.0,'2019-6-12 12:23:34',147.0);
INSERT INTO Check_Account VALUES(61,'East_Bank',70.999999999999999999,'2019-8-12 12:23:34',153.99999999999999999);
INSERT INTO Check_Account VALUES(62,'West_Bank',75.0,'2019-8-12 12:23:34',121.0);
INSERT INTO Check_Account VALUES(63,'West_Bank',27.999999999999999999,'2019-10-12 12:23:34',117.99999999999999999);
INSERT INTO Check_Account VALUES(64,'North_Bank',4.0,'2019-7-12 12:23:34',154.99999999999999999);
INSERT INTO Check_Account VALUES(65,'West_Bank',3.0,'2019-11-12 12:23:34',165.0);
INSERT INTO Check_Account VALUES(66,'South_Bank',44.0,'2019-12-12 12:23:34',172.0);
INSERT INTO Check_Account VALUES(67,'East_Bank',45.0,'2019-2-12 12:23:34',137.99999999999999999);
INSERT INTO Check_Account VALUES(68,'South_Bank',44.0,'2019-6-12 12:23:34',112.0);
INSERT INTO Check_Account VALUES(69,'South_Bank',37.0,'2019-2-12 12:23:34',134.0);
INSERT INTO Check_Account VALUES(70,'West_Bank',30.0,'2019-7-12 12:23:34',177.99999999999999999);
INSERT INTO Check_Account VALUES(71,'West_Bank',58.000000000000000001,'2019-1-12 12:23:34',161.0);
INSERT INTO Check_Account VALUES(72,'North_Bank',50.999999999999999999,'2019-10-12 12:23:34',108.0);
INSERT INTO Check_Account VALUES(73,'South_Bank',56.999999999999999998,'2019-12-12 12:23:34',171.0);
CREATE TABLE IF NOT EXISTS "Customer" 
(
   "customer_id"        integer               not null,
   "name"               text,
   "phone"              text,
   "address"            text,
   "contact_name"       text,
   "contact_phone"      text,
   "contact_email"      text,
   "contact_relation"   text,
   constraint PK_CUSTOMER primary key ("customer_id")
);
INSERT INTO Customer VALUES(0,'c0','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(1,'c1','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(2,'c2','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(3,'c3','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(4,'c4','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(5,'c5','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(6,'c6','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(7,'c7','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(8,'c8','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(9,'c9','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(10,'c10','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(11,'c11','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(12,'c12','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(13,'c13','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(14,'c14','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(15,'c15','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(16,'c16','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(17,'c17','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(18,'c18','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(19,'c19','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(20,'c20','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(21,'c21','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(22,'c22','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(23,'c23','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(24,'c24','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(25,'c25','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(26,'c26','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(27,'c27','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(28,'c28','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(29,'c29','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(30,'c30','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(31,'c31','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(32,'c32','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(33,'c33','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(34,'c34','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(35,'c35','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
INSERT INTO Customer VALUES(36,'c36','345678','ustc','ab','123456','ab@mail.ustc.edu.cn','other');
CREATE TABLE IF NOT EXISTS "Employee" 
(
   "employee_id"        integer               not null,
   "branch_name"        text                 not null,
   "manager_id"         integer,
   "name"               text,
   "address"            text,
   "phone"              text,
   "start_date"         text,
   constraint PK_EMPLOYEE primary key ("employee_id"),
   constraint FK_EMPLOYEE_MANAGE_EMPLOYEE foreign key ("manager_id")
      references "Employee" ("employee_id"),
   constraint FK_EMPLOYEE_WORK_FOR_BRANCH foreign key ("branch_name")
      references "Branch" ("branch_name")
);
INSERT INTO Employee VALUES(0,'North_Bank',1,'e1','ustc','123','2019-04-25 11:25:13');
INSERT INTO Employee VALUES(1,'North_Bank',1,'e1','ustc','123','2019-04-25 11:25:13');
INSERT INTO Employee VALUES(2,'North_Bank',1,'e2','ustc','234','2019-04-24 11:25:13');
INSERT INTO Employee VALUES(3,'North_Bank',1,'e3','ustc','234','2019-04-26 11:25:13');
INSERT INTO Employee VALUES(4,'South_Bank',4,'e4','ustc','234','2019-04-26 11:25:13');
INSERT INTO Employee VALUES(5,'South_Bank',4,'e5','ustc','234','2019-04-25 11:25:13');
INSERT INTO Employee VALUES(6,'South_Bank',4,'e6','ustc','234','2019-04-27 11:25:13');
INSERT INTO Employee VALUES(7,'East_Bank',7,'e7','ustc','234','2019-04-27 11:25:13');
INSERT INTO Employee VALUES(8,'East_Bank',7,'e8','ustc','234','2019-04-27 11:25:13');
INSERT INTO Employee VALUES(9,'East_Bank',7,'e9','ustc','234','2019-04-27 11:25:13');
INSERT INTO Employee VALUES(10,'West_Bank',10,'e10','ustc','234','2019-04-27 11:25:13');
INSERT INTO Employee VALUES(11,'West_Bank',10,'e11','ustc','234','2019-04-27 11:25:13');
INSERT INTO Employee VALUES(12,'West_Bank',10,'e11','ustc','234','2019-04-27 11:25:13');
CREATE TABLE IF NOT EXISTS "Loan" 
(
   "loan_id"            integer           not null,
   "branch_name"        text                 not null,
   "amount"             real,
   "total"              real     default 0,
   constraint PK_LOAN primary key ("loan_id"),
   constraint FK_LOAN_BRANCH_LO_BRANCH foreign key ("branch_name")
      references "Branch" ("branch_name")
);
INSERT INTO Loan VALUES(0,'East_Bank',107.0,29.0);
INSERT INTO Loan VALUES(1,'North_Bank',129.99999999999999999,22.999999999999999999);
INSERT INTO Loan VALUES(2,'North_Bank',137.0,34.0);
INSERT INTO Loan VALUES(3,'East_Bank',138.99999999999999999,7.0);
INSERT INTO Loan VALUES(4,'North_Bank',141.99999999999999999,15.0);
INSERT INTO Loan VALUES(5,'East_Bank',132.0,3.0);
INSERT INTO Loan VALUES(6,'North_Bank',125.99999999999999999,29.0);
INSERT INTO Loan VALUES(7,'East_Bank',114.99999999999999999,19.0);
INSERT INTO Loan VALUES(8,'West_Bank',126.99999999999999999,30.999999999999999999);
INSERT INTO Loan VALUES(9,'East_Bank',111.0,30.0);
INSERT INTO Loan VALUES(10,'East_Bank',135.0,29.0);
INSERT INTO Loan VALUES(11,'North_Bank',145.0,22.0);
INSERT INTO Loan VALUES(12,'North_Bank',105.99999999999999999,37.0);
INSERT INTO Loan VALUES(13,'South_Bank',125.99999999999999999,25.0);
INSERT INTO Loan VALUES(14,'South_Bank',100.0,12.0);
INSERT INTO Loan VALUES(15,'North_Bank',124.0,13.999999999999999999);
INSERT INTO Loan VALUES(16,'West_Bank',110.0,11.0);
INSERT INTO Loan VALUES(17,'West_Bank',140.99999999999999999,17.999999999999999999);
INSERT INTO Loan VALUES(18,'South_Bank',126.99999999999999999,15.0);
INSERT INTO Loan VALUES(19,'West_Bank',108.0,11.0);
INSERT INTO Loan VALUES(20,'West_Bank',130.99999999999999999,17.0);
INSERT INTO Loan VALUES(21,'East_Bank',129.99999999999999999,39.0);
INSERT INTO Loan VALUES(22,'West_Bank',109.0,20.0);
INSERT INTO Loan VALUES(23,'North_Bank',137.99999999999999999,24.0);
INSERT INTO Loan VALUES(24,'West_Bank',132.0,27.999999999999999999);
INSERT INTO Loan VALUES(25,'North_Bank',144.0,20.999999999999999999);
INSERT INTO Loan VALUES(26,'South_Bank',138.99999999999999999,12.0);
INSERT INTO Loan VALUES(27,'North_Bank',134.0,12.999999999999999999);
INSERT INTO Loan VALUES(28,'South_Bank',102.99999999999999999,27.0);
INSERT INTO Loan VALUES(29,'South_Bank',120.0,37.999999999999999999);
INSERT INTO Loan VALUES(30,'East_Bank',102.99999999999999999,11.0);
INSERT INTO Loan VALUES(31,'East_Bank',128.99999999999999999,3.0);
INSERT INTO Loan VALUES(32,'West_Bank',116.99999999999999999,12.0);
INSERT INTO Loan VALUES(33,'North_Bank',148.0,22.999999999999999999);
INSERT INTO Loan VALUES(34,'North_Bank',102.99999999999999999,17.0);
INSERT INTO Loan VALUES(35,'East_Bank',115.99999999999999999,12.0);
INSERT INTO Loan VALUES(36,'West_Bank',119.0,20.999999999999999999);
CREATE TABLE IF NOT EXISTS "Payment" 
(
   "loan_id"            integer           not null,
   "payment_id"         integer               not null,
   "payment_date"       text,
   "amount"             real,
   constraint PK_PAYMENT primary key ("loan_id", "payment_id"),
   constraint FK_PAYMENT_PAY_LOAN foreign key ("loan_id")
      references "Loan" ("loan_id")
);
INSERT INTO Payment VALUES(0,73,'2019-4-12 12:23:34',15.0);
INSERT INTO Payment VALUES(0,0,'2019-6-12 12:23:34',11.0);
INSERT INTO Payment VALUES(1,1,'2019-11-12 12:23:34',6.0);
INSERT INTO Payment VALUES(2,2,'2019-1-12 12:23:34',17.999999999999999999);
INSERT INTO Payment VALUES(3,3,'2019-11-12 12:23:34',3.0);
INSERT INTO Payment VALUES(4,4,'2019-3-12 12:23:34',0.0);
INSERT INTO Payment VALUES(5,5,'2019-9-12 12:23:34',1.0);
INSERT INTO Payment VALUES(6,6,'2019-7-12 12:23:34',17.0);
INSERT INTO Payment VALUES(7,7,'2019-10-12 12:23:34',17.999999999999999999);
INSERT INTO Payment VALUES(8,8,'2019-9-12 12:23:34',11.0);
INSERT INTO Payment VALUES(9,9,'2019-9-12 12:23:34',17.0);
INSERT INTO Payment VALUES(10,10,'2019-7-12 12:23:34',17.999999999999999999);
INSERT INTO Payment VALUES(11,11,'2019-12-12 12:23:34',10.0);
INSERT INTO Payment VALUES(12,12,'2019-3-12 12:23:34',17.999999999999999999);
INSERT INTO Payment VALUES(13,13,'2019-3-12 12:23:34',12.0);
INSERT INTO Payment VALUES(14,14,'2019-3-12 12:23:34',5.0);
INSERT INTO Payment VALUES(15,15,'2019-1-12 12:23:34',11.0);
INSERT INTO Payment VALUES(16,16,'2019-2-12 12:23:34',9.0);
INSERT INTO Payment VALUES(17,17,'2019-1-12 12:23:34',17.0);
INSERT INTO Payment VALUES(18,18,'2019-2-12 12:23:34',12.0);
INSERT INTO Payment VALUES(19,19,'2019-12-12 12:23:34',7.0);
INSERT INTO Payment VALUES(20,20,'2019-5-12 12:23:34',17.0);
INSERT INTO Payment VALUES(21,21,'2019-11-12 12:23:34',20.0);
INSERT INTO Payment VALUES(22,22,'2019-8-12 12:23:34',16.0);
INSERT INTO Payment VALUES(23,23,'2019-7-12 12:23:34',12.999999999999999999);
INSERT INTO Payment VALUES(24,24,'2019-11-12 12:23:34',17.999999999999999999);
INSERT INTO Payment VALUES(25,25,'2019-2-12 12:23:34',8.0);
INSERT INTO Payment VALUES(26,26,'2019-11-12 12:23:34',8.0);
INSERT INTO Payment VALUES(27,27,'2019-3-12 12:23:34',5.0);
INSERT INTO Payment VALUES(28,28,'2019-12-12 12:23:34',11.0);
INSERT INTO Payment VALUES(29,29,'2019-11-12 12:23:34',19.0);
INSERT INTO Payment VALUES(30,30,'2019-1-12 12:23:34',11.0);
INSERT INTO Payment VALUES(31,31,'2019-4-12 12:23:34',0.0);
INSERT INTO Payment VALUES(32,32,'2019-2-12 12:23:34',10.0);
INSERT INTO Payment VALUES(33,33,'2019-9-12 12:23:34',19.0);
INSERT INTO Payment VALUES(34,34,'2019-3-12 12:23:34',2.0);
INSERT INTO Payment VALUES(35,35,'2019-12-12 12:23:34',2.0);
INSERT INTO Payment VALUES(36,36,'2019-7-12 12:23:34',1.0);
INSERT INTO Payment VALUES(0,37,'2019-7-12 12:23:34',3.0);
INSERT INTO Payment VALUES(1,38,'2019-12-12 12:23:34',17.0);
INSERT INTO Payment VALUES(2,39,'2019-2-12 12:23:34',16.0);
INSERT INTO Payment VALUES(3,40,'2019-10-12 12:23:34',4.0);
INSERT INTO Payment VALUES(4,41,'2019-6-12 12:23:34',15.0);
INSERT INTO Payment VALUES(5,42,'2019-7-12 12:23:34',2.0);
INSERT INTO Payment VALUES(6,43,'2019-11-12 12:23:34',12.0);
INSERT INTO Payment VALUES(7,44,'2019-7-12 12:23:34',1.0);
INSERT INTO Payment VALUES(8,45,'2019-11-12 12:23:34',20.0);
INSERT INTO Payment VALUES(9,46,'2019-1-12 12:23:34',12.999999999999999999);
INSERT INTO Payment VALUES(10,47,'2019-6-12 12:23:34',11.0);
INSERT INTO Payment VALUES(11,48,'2019-3-12 12:23:34',12.0);
INSERT INTO Payment VALUES(12,49,'2019-6-12 12:23:34',19.0);
INSERT INTO Payment VALUES(13,50,'2019-12-12 12:23:34',12.999999999999999999);
INSERT INTO Payment VALUES(14,51,'2019-8-12 12:23:34',7.0);
INSERT INTO Payment VALUES(15,52,'2019-2-12 12:23:34',3.0);
INSERT INTO Payment VALUES(16,53,'2019-7-12 12:23:34',2.0);
INSERT INTO Payment VALUES(17,54,'2019-5-12 12:23:34',1.0);
INSERT INTO Payment VALUES(18,55,'2019-3-12 12:23:34',3.0);
INSERT INTO Payment VALUES(19,56,'2019-10-12 12:23:34',4.0);
INSERT INTO Payment VALUES(20,57,'2019-4-12 12:23:34',0.0);
INSERT INTO Payment VALUES(21,58,'2019-3-12 12:23:34',19.0);
INSERT INTO Payment VALUES(22,59,'2019-6-12 12:23:34',4.0);
INSERT INTO Payment VALUES(23,60,'2019-5-12 12:23:34',11.0);
INSERT INTO Payment VALUES(24,61,'2019-4-12 12:23:34',10.0);
INSERT INTO Payment VALUES(25,62,'2019-3-12 12:23:34',12.999999999999999999);
INSERT INTO Payment VALUES(26,63,'2019-11-12 12:23:34',4.0);
INSERT INTO Payment VALUES(27,64,'2019-10-12 12:23:34',8.0);
INSERT INTO Payment VALUES(28,65,'2019-5-12 12:23:34',16.0);
INSERT INTO Payment VALUES(29,66,'2019-9-12 12:23:34',19.0);
INSERT INTO Payment VALUES(30,67,'2019-9-12 12:23:34',0.0);
INSERT INTO Payment VALUES(31,68,'2019-11-12 12:23:34',3.0);
INSERT INTO Payment VALUES(32,69,'2019-2-12 12:23:34',2.0);
INSERT INTO Payment VALUES(33,70,'2019-2-12 12:23:34',4.0);
INSERT INTO Payment VALUES(34,71,'2019-4-12 12:23:34',15.0);
INSERT INTO Payment VALUES(35,72,'2019-9-12 12:23:34',10.0);
INSERT INTO Payment VALUES(36,73,'2019-7-12 12:23:34',20.0);
CREATE TABLE IF NOT EXISTS "Saving_Account" 
(
   "Account_id"         integer               not null,
   "branch_name"        text                 not null,
   "balance"            real,
   "open_date"          text,
   "interest_rate"      real,
   "currency_type"      text,
   constraint PK_SAVING_Account primary key ("Account_id"),
   constraint FK_SAVING_A_Account_IN_Account foreign key ("Account_id")
      references "Account" ("Account_id"),
   constraint FK_Saving_ACCOUNT_OPEN_ACCO_BRANCH foreign key ("branch_name")
      references "Branch" ("branch_name")
);
INSERT INTO Saving_Account VALUES(0,'South_Bank',11.0,'2019-10-12 12:23:34',0.75988299999999997513,'USD');
INSERT INTO Saving_Account VALUES(1,'North_Bank',17.0,'2019-12-12 12:23:34',0.49387799999999998368,'USD');
INSERT INTO Saving_Account VALUES(2,'East_Bank',81.999999999999999998,'2019-4-12 12:23:34',0.69089100000000003287,'RMB');
INSERT INTO Saving_Account VALUES(3,'South_Bank',83.000000000000000001,'2019-4-12 12:23:34',0.13973599999999999909,'USD');
INSERT INTO Saving_Account VALUES(4,'West_Bank',9.0,'2019-10-12 12:23:34',0.85274399999999994648,'RMB');
INSERT INTO Saving_Account VALUES(5,'West_Bank',6.0,'2019-12-12 12:23:34',0.61031999999999997363,'USD');
INSERT INTO Saving_Account VALUES(6,'West_Bank',35.999999999999999999,'2019-2-12 12:23:34',0.32529000000000002357,'RMB');
INSERT INTO Saving_Account VALUES(7,'South_Bank',22.0,'2019-3-12 12:23:34',0.21890699999999999048,'RMB');
INSERT INTO Saving_Account VALUES(8,'South_Bank',32.0,'2019-1-12 12:23:34',0.032550000000000002542,'USD');
INSERT INTO Saving_Account VALUES(9,'North_Bank',37.999999999999999999,'2019-10-12 12:23:34',0.87841700000000000336,'USD');
INSERT INTO Saving_Account VALUES(10,'West_Bank',8.0,'2019-1-12 12:23:34',0.63936400000000004339,'RMB');
INSERT INTO Saving_Account VALUES(11,'North_Bank',39.0,'2019-9-12 12:23:34',0.81589299999999997936,'USD');
INSERT INTO Saving_Account VALUES(12,'South_Bank',25.999999999999999999,'2019-6-12 12:23:34',0.012937999999999999847,'RMB');
INSERT INTO Saving_Account VALUES(13,'South_Bank',86.999999999999999998,'2019-9-12 12:23:34',0.87231099999999994754,'USD');
INSERT INTO Saving_Account VALUES(14,'South_Bank',19.0,'2019-6-12 12:23:34',0.94400899999999998701,'USD');
INSERT INTO Saving_Account VALUES(15,'East_Bank',22.999999999999999999,'2019-10-12 12:23:34',0.49925700000000000633,'RMB');
INSERT INTO Saving_Account VALUES(16,'North_Bank',17.999999999999999999,'2019-5-12 12:23:34',0.85780500000000003968,'RMB');
INSERT INTO Saving_Account VALUES(17,'South_Bank',27.999999999999999999,'2019-10-12 12:23:34',0.9346999999999999753,'USD');
INSERT INTO Saving_Account VALUES(18,'North_Bank',5.0,'2019-5-12 12:23:34',0.57815499999999997449,'RMB');
INSERT INTO Saving_Account VALUES(19,'West_Bank',37.0,'2019-12-12 12:23:34',0.6128169999999999451,'USD');
INSERT INTO Saving_Account VALUES(20,'North_Bank',64.0,'2019-11-12 12:23:34',0.29204900000000000304,'RMB');
INSERT INTO Saving_Account VALUES(21,'East_Bank',27.0,'2019-4-12 12:23:34',0.94518199999999996663,'USD');
INSERT INTO Saving_Account VALUES(22,'South_Bank',17.999999999999999999,'2019-7-12 12:23:34',0.074730000000000004756,'USD');
INSERT INTO Saving_Account VALUES(23,'North_Bank',65.0,'2019-8-12 12:23:34',0.18745899999999998675,'RMB');
INSERT INTO Saving_Account VALUES(24,'North_Bank',83.999999999999999996,'2019-3-12 12:23:34',0.95394800000000001816,'USD');
INSERT INTO Saving_Account VALUES(25,'West_Bank',66.999999999999999998,'2019-4-12 12:23:34',0.20806299999999999794,'RMB');
INSERT INTO Saving_Account VALUES(26,'North_Bank',65.0,'2019-3-12 12:23:34',0.26229999999999997761,'USD');
INSERT INTO Saving_Account VALUES(27,'East_Bank',50.999999999999999999,'2019-12-12 12:23:34',0.45929799999999998405,'USD');
INSERT INTO Saving_Account VALUES(28,'West_Bank',58.000000000000000001,'2019-12-12 12:23:34',0.18605399999999999716,'USD');
INSERT INTO Saving_Account VALUES(29,'East_Bank',29.0,'2019-7-12 12:23:34',0.698631000000000002,'USD');
INSERT INTO Saving_Account VALUES(30,'East_Bank',64.0,'2019-9-12 12:23:34',0.28001199999999998313,'RMB');
INSERT INTO Saving_Account VALUES(31,'West_Bank',32.999999999999999999,'2019-3-12 12:23:34',0.9929480000000000528,'USD');
INSERT INTO Saving_Account VALUES(32,'South_Bank',11.0,'2019-9-12 12:23:34',0.80407600000000001294,'RMB');
INSERT INTO Saving_Account VALUES(33,'West_Bank',90.0,'2019-5-12 12:23:34',0.78643200000000001992,'RMB');
INSERT INTO Saving_Account VALUES(34,'North_Bank',96.999999999999999998,'2019-5-12 12:23:34',0.79219600000000001127,'USD');
INSERT INTO Saving_Account VALUES(35,'West_Bank',22.0,'2019-3-12 12:23:34',0.68572800000000000419,'RMB');
INSERT INTO Saving_Account VALUES(36,'South_Bank',48.000000000000000001,'2019-8-12 12:23:34',0.98122399999999998509,'RMB');
CREATE TABLE IF NOT EXISTS "borrow" 
(
   "customer_id"        integer               not null,
   "loan_id"            integer           not null,
   constraint PK_BORROW primary key ("customer_id", "loan_id"),
   constraint FK_BORROW_BORROW_CUSTOMER foreign key ("customer_id")
      references "Customer" ("customer_id"),
   constraint FK_BORROW_BORROW2_LOAN foreign key ("loan_id")
      references "Loan" ("loan_id")
);
INSERT INTO borrow VALUES(0,36);
INSERT INTO borrow VALUES(1,35);
INSERT INTO borrow VALUES(2,34);
INSERT INTO borrow VALUES(3,33);
INSERT INTO borrow VALUES(4,32);
INSERT INTO borrow VALUES(5,31);
INSERT INTO borrow VALUES(6,30);
INSERT INTO borrow VALUES(7,29);
INSERT INTO borrow VALUES(8,28);
INSERT INTO borrow VALUES(9,27);
INSERT INTO borrow VALUES(10,26);
INSERT INTO borrow VALUES(11,25);
INSERT INTO borrow VALUES(12,24);
INSERT INTO borrow VALUES(13,23);
INSERT INTO borrow VALUES(14,22);
INSERT INTO borrow VALUES(15,21);
INSERT INTO borrow VALUES(16,20);
INSERT INTO borrow VALUES(17,19);
INSERT INTO borrow VALUES(18,18);
INSERT INTO borrow VALUES(19,17);
INSERT INTO borrow VALUES(20,16);
INSERT INTO borrow VALUES(21,15);
INSERT INTO borrow VALUES(22,14);
INSERT INTO borrow VALUES(23,13);
INSERT INTO borrow VALUES(24,12);
INSERT INTO borrow VALUES(25,11);
INSERT INTO borrow VALUES(26,10);
INSERT INTO borrow VALUES(27,9);
INSERT INTO borrow VALUES(28,8);
INSERT INTO borrow VALUES(29,7);
INSERT INTO borrow VALUES(30,6);
INSERT INTO borrow VALUES(31,5);
INSERT INTO borrow VALUES(32,4);
INSERT INTO borrow VALUES(33,3);
INSERT INTO borrow VALUES(34,2);
INSERT INTO borrow VALUES(35,1);
INSERT INTO borrow VALUES(36,0);
CREATE TABLE IF NOT EXISTS "depositor" 
(
   "Account_id"         integer               not null,
   "customer_id"        integer               not null,
   "access_date"        text,
   constraint PK_DEPOSITOR primary key ("Account_id", "customer_id"),
   constraint FK_DEPOSITO_DEPOSITOR_Account foreign key ("Account_id")
      references "Account" ("Account_id"),
   constraint FK_DEPOSITO_DEPOSITOR_CUSTOMER foreign key ("customer_id")
      references "Customer" ("customer_id")
);
INSERT INTO depositor VALUES(0,36,'2019-3-12 12:23:34');
INSERT INTO depositor VALUES(1,35,'2019-7-12 12:23:34');
INSERT INTO depositor VALUES(2,34,'2019-4-12 12:23:34');
INSERT INTO depositor VALUES(3,33,'2019-3-12 12:23:34');
INSERT INTO depositor VALUES(4,32,'2019-3-12 12:23:34');
INSERT INTO depositor VALUES(5,31,'2019-4-12 12:23:34');
INSERT INTO depositor VALUES(6,30,'2019-1-12 12:23:34');
INSERT INTO depositor VALUES(7,29,'2019-8-12 12:23:34');
INSERT INTO depositor VALUES(8,28,'2019-5-12 12:23:34');
INSERT INTO depositor VALUES(9,27,'2019-1-12 12:23:34');
INSERT INTO depositor VALUES(10,26,'2019-3-12 12:23:34');
INSERT INTO depositor VALUES(11,25,'2019-1-12 12:23:34');
INSERT INTO depositor VALUES(12,24,'2019-8-12 12:23:34');
INSERT INTO depositor VALUES(13,23,'2019-2-12 12:23:34');
INSERT INTO depositor VALUES(14,22,'2019-4-12 12:23:34');
INSERT INTO depositor VALUES(15,21,'2019-5-12 12:23:34');
INSERT INTO depositor VALUES(16,20,'2019-7-12 12:23:34');
INSERT INTO depositor VALUES(17,19,'2019-5-12 12:23:34');
INSERT INTO depositor VALUES(18,18,'2019-11-12 12:23:34');
INSERT INTO depositor VALUES(19,17,'2019-3-12 12:23:34');
INSERT INTO depositor VALUES(20,16,'2019-7-12 12:23:34');
INSERT INTO depositor VALUES(21,15,'2019-8-12 12:23:34');
INSERT INTO depositor VALUES(22,14,'2019-8-12 12:23:34');
INSERT INTO depositor VALUES(23,13,'2019-6-12 12:23:34');
INSERT INTO depositor VALUES(24,12,'2019-10-12 12:23:34');
INSERT INTO depositor VALUES(25,11,'2019-4-12 12:23:34');
INSERT INTO depositor VALUES(26,10,'2019-6-12 12:23:34');
INSERT INTO depositor VALUES(27,9,'2019-2-12 12:23:34');
INSERT INTO depositor VALUES(28,8,'2019-3-12 12:23:34');
INSERT INTO depositor VALUES(29,7,'2019-10-12 12:23:34');
INSERT INTO depositor VALUES(30,6,'2019-6-12 12:23:34');
INSERT INTO depositor VALUES(31,5,'2019-7-12 12:23:34');
INSERT INTO depositor VALUES(32,4,'2019-2-12 12:23:34');
INSERT INTO depositor VALUES(33,3,'2019-2-12 12:23:34');
INSERT INTO depositor VALUES(34,2,'2019-3-12 12:23:34');
INSERT INTO depositor VALUES(35,1,'2019-2-12 12:23:34');
INSERT INTO depositor VALUES(36,0,'2019-2-12 12:23:34');
INSERT INTO depositor VALUES(37,36,'2019-10-12 12:23:34');
INSERT INTO depositor VALUES(38,35,'2019-2-12 12:23:34');
INSERT INTO depositor VALUES(39,34,'2019-5-12 12:23:34');
INSERT INTO depositor VALUES(40,33,'2019-10-12 12:23:34');
INSERT INTO depositor VALUES(41,32,'2019-5-12 12:23:34');
INSERT INTO depositor VALUES(42,31,'2019-11-12 12:23:34');
INSERT INTO depositor VALUES(43,30,'2019-3-12 12:23:34');
INSERT INTO depositor VALUES(44,29,'2019-3-12 12:23:34');
INSERT INTO depositor VALUES(45,28,'2019-11-12 12:23:34');
INSERT INTO depositor VALUES(46,27,'2019-6-12 12:23:34');
INSERT INTO depositor VALUES(47,26,'2019-5-12 12:23:34');
INSERT INTO depositor VALUES(48,25,'2019-6-12 12:23:34');
INSERT INTO depositor VALUES(49,24,'2019-12-12 12:23:34');
INSERT INTO depositor VALUES(50,23,'2019-2-12 12:23:34');
INSERT INTO depositor VALUES(51,22,'2019-2-12 12:23:34');
INSERT INTO depositor VALUES(52,21,'2019-2-12 12:23:34');
INSERT INTO depositor VALUES(53,20,'2019-4-12 12:23:34');
INSERT INTO depositor VALUES(54,19,'2019-6-12 12:23:34');
INSERT INTO depositor VALUES(55,18,'2019-12-12 12:23:34');
INSERT INTO depositor VALUES(56,17,'2019-3-12 12:23:34');
INSERT INTO depositor VALUES(57,16,'2019-4-12 12:23:34');
INSERT INTO depositor VALUES(58,15,'2019-8-12 12:23:34');
INSERT INTO depositor VALUES(59,14,'2019-10-12 12:23:34');
INSERT INTO depositor VALUES(60,13,'2019-2-12 12:23:34');
INSERT INTO depositor VALUES(61,12,'2019-6-12 12:23:34');
INSERT INTO depositor VALUES(62,11,'2019-7-12 12:23:34');
INSERT INTO depositor VALUES(63,10,'2019-9-12 12:23:34');
INSERT INTO depositor VALUES(64,9,'2019-6-12 12:23:34');
INSERT INTO depositor VALUES(65,8,'2019-2-12 12:23:34');
INSERT INTO depositor VALUES(66,7,'2019-4-12 12:23:34');
INSERT INTO depositor VALUES(67,6,'2019-10-12 12:23:34');
INSERT INTO depositor VALUES(68,5,'2019-9-12 12:23:34');
INSERT INTO depositor VALUES(69,4,'2019-3-12 12:23:34');
INSERT INTO depositor VALUES(70,3,'2019-9-12 12:23:34');
INSERT INTO depositor VALUES(71,2,'2019-10-12 12:23:34');
INSERT INTO depositor VALUES(72,1,'2019-10-12 12:23:34');
INSERT INTO depositor VALUES(73,0,'2019-1-12 12:23:34');
CREATE TABLE IF NOT EXISTS "responsible" 
(
   "employee_id"        integer               not null,
   "customer_id"        integer               not null,
   "type"               text,
   constraint PK_RESPONSIBLE primary key ("employee_id", "customer_id"),
   constraint FK_RESPONSI_RESPONSIB_EMPLOYEE foreign key ("employee_id")
      references "Employee" ("employee_id"),
   constraint FK_RESPONSI_RESPONSIB_CUSTOMER foreign key ("customer_id")
      references "Customer" ("customer_id")
   
);
INSERT INTO responsible VALUES(4,0,'B');
INSERT INTO responsible VALUES(1,1,'B');
INSERT INTO responsible VALUES(8,2,'B');
INSERT INTO responsible VALUES(8,3,'B');
INSERT INTO responsible VALUES(10,4,'B');
INSERT INTO responsible VALUES(0,5,'A');
INSERT INTO responsible VALUES(0,6,'A');
INSERT INTO responsible VALUES(9,7,'A');
INSERT INTO responsible VALUES(0,8,'B');
INSERT INTO responsible VALUES(11,9,'B');
INSERT INTO responsible VALUES(3,10,'B');
INSERT INTO responsible VALUES(8,11,'A');
INSERT INTO responsible VALUES(4,12,'B');
INSERT INTO responsible VALUES(7,13,'A');
INSERT INTO responsible VALUES(9,14,'B');
INSERT INTO responsible VALUES(7,15,'B');
INSERT INTO responsible VALUES(7,16,'B');
INSERT INTO responsible VALUES(1,17,'B');
INSERT INTO responsible VALUES(7,18,'B');
INSERT INTO responsible VALUES(3,19,'B');
INSERT INTO responsible VALUES(3,20,'B');
INSERT INTO responsible VALUES(12,21,'B');
INSERT INTO responsible VALUES(1,22,'B');
INSERT INTO responsible VALUES(11,23,'B');
INSERT INTO responsible VALUES(6,24,'B');
INSERT INTO responsible VALUES(0,25,'A');
INSERT INTO responsible VALUES(7,26,'B');
INSERT INTO responsible VALUES(2,27,'B');
INSERT INTO responsible VALUES(2,28,'A');
INSERT INTO responsible VALUES(7,29,'A');
INSERT INTO responsible VALUES(4,30,'A');
INSERT INTO responsible VALUES(11,31,'B');
INSERT INTO responsible VALUES(7,32,'A');
INSERT INTO responsible VALUES(4,33,'B');
INSERT INTO responsible VALUES(7,34,'B');
INSERT INTO responsible VALUES(0,35,'A');
INSERT INTO responsible VALUES(3,36,'A');
CREATE TRIGGER insert_saving after insert on saving_account
begin
insert into account("Account_id","branch_name")
	values (new.Account_id,new.branch_name);
end;
CREATE TRIGGER insert_check after insert on check_account
begin
insert into account("Account_id","branch_name")
	values (new.Account_id,new.branch_name);
end;
CREATE TRIGGER delete_saving after delete on saving_account
begin
delete from account 
	where Account_id=old.Account_id;
end;
CREATE TRIGGER delete_check after delete on check_account
begin
delete from account 
	where Account_id=old.Account_id;
end;
CREATE TRIGGER add_payment after insert on payment
begin
update loan 
   set total=total+new.amount
   where loan_id=new.loan_id;
end;
COMMIT;
