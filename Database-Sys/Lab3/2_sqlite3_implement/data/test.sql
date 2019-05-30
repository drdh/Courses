PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "Account" 
(
   "Account_id"         integer               not null,
   "branch_name"        text                  not null,
   "balance"            real,
   "open_date"          text,
   constraint PK_Account primary key ("Account_id"),
   constraint FK_ACCOUNT_OPEN_ACCO_BRANCH foreign key ("branch_name")
      references "Branch" ("branch_name")
);
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
--   "branch_name"        text,                 not null,
--   "balance"            real,
--   "open_date"          text,
   "overdraft"          real,
   constraint PK_CHECK_ACCOUNT primary key ("Account_id"),
   constraint FK_CHECK_AC_Account_IN_Account foreign key ("Account_id")
      references "Account" ("Account_id")
);
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
INSERT INTO Customer VALUES(1,'ab','123456','ustc','cd','234567','ab@mail.ustc.edu.cn','friends');
INSERT INTO Customer VALUES(2,'cd','234567','ustc','ef','234567','cd@mail.ustc.edu.cn','parents');
INSERT INTO Customer VALUES(3,'ef','345678','ustc','ab','123456','ef@mail.ustc.edu.cn','other');
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
INSERT INTO Employee VALUES(1,'North_Bank',2,'a','ustc','123','2019-04-25 11:25:13.333');
INSERT INTO Employee VALUES(2,'North_Bank',2,'b','ustc','234','2019-04-24 11:25:13.333');
CREATE TABLE IF NOT EXISTS "Loan" 
(
   "loan_id"            integer           not null,
   "branch_name"        text                 not null,
   "amount"             real,
   constraint PK_LOAN primary key ("loan_id"),
   constraint FK_LOAN_BRANCH_LO_BRANCH foreign key ("branch_name")
      references "Branch" ("branch_name")
);
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
CREATE TABLE IF NOT EXISTS "Saving_Account" 
(
   "Account_id"         integer               not null,
--   "branch_name"        text,                 not null,
--   "balance"            real,
--   "open_date"          text,
   "interest_rate"      real,
   "currency_type"      text,
   constraint PK_SAVING_Account primary key ("Account_id"),
   constraint FK_SAVING_A_Account_IN_Account foreign key ("Account_id")
      references "Account" ("Account_id")
);
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
COMMIT;
