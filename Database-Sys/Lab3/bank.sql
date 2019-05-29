/*==============================================================*/
/* DBMS name:      ORACLE Version 11g                           */
/* Created on:     2019/5/29 20:22:53                           */
/*==============================================================*/


alter table "Check_Account"
   drop constraint FK_CHECK_AC_ACOUNT_IN_ACOUNT;

alter table "Employee"
   drop constraint FK_EMPLOYEE_MANAGE_EMPLOYEE;

alter table "Employee"
   drop constraint FK_EMPLOYEE_WORK_FOR_BRANCH;

alter table "Loan"
   drop constraint FK_LOAN_BRANCH_LO_BRANCH;

alter table "Payment"
   drop constraint FK_PAYMENT_PAY_LOAN;

alter table "Saving_Accout"
   drop constraint FK_SAVING_A_ACOUNT_IN_ACOUNT;

alter table "borrow"
   drop constraint FK_BORROW_BORROW_CUSTOMER;

alter table "borrow"
   drop constraint FK_BORROW_BORROW2_LOAN;

alter table "depositor"
   drop constraint FK_DEPOSITO_DEPOSITOR_ACOUNT;

alter table "depositor"
   drop constraint FK_DEPOSITO_DEPOSITOR_CUSTOMER;

alter table "responsible"
   drop constraint FK_RESPONSI_RESPONSIB_EMPLOYEE;

alter table "responsible"
   drop constraint FK_RESPONSI_RESPONSIB_CUSTOMER;

drop table "Acount" cascade constraints;

drop table "Branch" cascade constraints;

drop table "Check_Account" cascade constraints;

drop table "Customer" cascade constraints;

drop index "manage_FK";

drop index "work_for_FK";

drop table "Employee" cascade constraints;

drop index "branch_loan_FK";

drop table "Loan" cascade constraints;

drop table "Payment" cascade constraints;

drop table "Saving_Accout" cascade constraints;

drop index "borrow2_FK";

drop index "borrow_FK";

drop table "borrow" cascade constraints;

drop index "depositor2_FK";

drop index "depositor_FK";

drop table "depositor" cascade constraints;

drop index "responsible2_FK";

drop index "responsible_FK";

drop table "responsible" cascade constraints;

/*==============================================================*/
/* Table: "Acount"                                              */
/*==============================================================*/
create table "Acount" 
(
   "acount_id"          NUMBER               not null,
   "balance"            NUMBER(8,2),
   "open_date"          DATE,
   constraint PK_ACOUNT primary key ("acount_id")
);

/*==============================================================*/
/* Table: "Branch"                                              */
/*==============================================================*/
create table "Branch" 
(
   "branch_name"        CLOB                 not null,
   "city"               CLOB,
   "asset"              NUMBER(8,2),
   constraint PK_BRANCH primary key ("branch_name")
);

/*==============================================================*/
/* Table: "Check_Account"                                       */
/*==============================================================*/
create table "Check_Account" 
(
   "acount_id"          NUMBER               not null,
   "balance"            NUMBER(8,2),
   "Aco_open_date"      DATE,
   "open_date"          DATE,
   "overdraft"          NUMBER(8,2),
   constraint PK_CHECK_ACCOUNT primary key ("acount_id")
);

/*==============================================================*/
/* Table: "Customer"                                            */
/*==============================================================*/
create table "Customer" 
(
   "customer_id"        NUMBER               not null,
   "name"               CLOB,
   "phone"              NUMBER,
   "address"            CLOB,
   "contact_name"       CLOB,
   "contact_phone"      NUMBER,
   "contact_email"      CLOB,
   "contact_relation"   CLOB,
   constraint PK_CUSTOMER primary key ("customer_id")
);

/*==============================================================*/
/* Table: "Employee"                                            */
/*==============================================================*/
create table "Employee" 
(
   "employee_id"        NUMBER               not null,
   "branch_name"        CLOB                 not null,
   "Emp_employee_id"    NUMBER,
   "name"               CLOB,
   "addree"             CLOB,
   "phone"              NUMBER,
   "start_date"         DATE,
   constraint PK_EMPLOYEE primary key ("employee_id")
);

/*==============================================================*/
/* Index: "work_for_FK"                                         */
/*==============================================================*/
create index "work_for_FK" on "Employee" (
   "branch_name" ASC
);

/*==============================================================*/
/* Index: "manage_FK"                                           */
/*==============================================================*/
create index "manage_FK" on "Employee" (
   "Emp_employee_id" ASC
);

/*==============================================================*/
/* Table: "Loan"                                                */
/*==============================================================*/
create table "Loan" 
(
   "load_id"            NUMBER(11)           not null,
   "branch_name"        CLOB                 not null,
   "amount"             NUMBER(8,2),
   constraint PK_LOAN primary key ("load_id")
);

/*==============================================================*/
/* Index: "branch_loan_FK"                                      */
/*==============================================================*/
create index "branch_loan_FK" on "Loan" (
   "branch_name" ASC
);

/*==============================================================*/
/* Table: "Payment"                                             */
/*==============================================================*/
create table "Payment" 
(
   "load_id"            NUMBER(11)           not null,
   "payment_date"       DATE,
   "amount"             NUMBER(8,2),
   constraint PK_PAYMENT primary key ("load_id")
);

/*==============================================================*/
/* Table: "Saving_Accout"                                       */
/*==============================================================*/
create table "Saving_Accout" 
(
   "acount_id"          NUMBER               not null,
   "balance"            NUMBER(8,2),
   "open_date"          DATE,
   "interest_rate"      FLOAT,
   "currency_type"      CLOB,
   constraint PK_SAVING_ACCOUT primary key ("acount_id")
);

/*==============================================================*/
/* Table: "borrow"                                              */
/*==============================================================*/
create table "borrow" 
(
   "customer_id"        NUMBER               not null,
   "load_id"            NUMBER(11)           not null,
   constraint PK_BORROW primary key ("customer_id", "load_id")
);

/*==============================================================*/
/* Index: "borrow_FK"                                           */
/*==============================================================*/
create index "borrow_FK" on "borrow" (
   "customer_id" ASC
);

/*==============================================================*/
/* Index: "borrow2_FK"                                          */
/*==============================================================*/
create index "borrow2_FK" on "borrow" (
   "load_id" ASC
);

/*==============================================================*/
/* Table: "depositor"                                           */
/*==============================================================*/
create table "depositor" 
(
   "acount_id"          NUMBER               not null,
   "customer_id"        NUMBER               not null,
   "access_date"        DATE,
   constraint PK_DEPOSITOR primary key ("acount_id", "customer_id")
);

/*==============================================================*/
/* Index: "depositor_FK"                                        */
/*==============================================================*/
create index "depositor_FK" on "depositor" (
   "acount_id" ASC
);

/*==============================================================*/
/* Index: "depositor2_FK"                                       */
/*==============================================================*/
create index "depositor2_FK" on "depositor" (
   "customer_id" ASC
);

/*==============================================================*/
/* Table: "responsible"                                         */
/*==============================================================*/
create table "responsible" 
(
   "employee_id"        NUMBER               not null,
   "customer_id"        NUMBER               not null,
   "type"               CLOB,
   constraint PK_RESPONSIBLE primary key ("employee_id", "customer_id")
);

/*==============================================================*/
/* Index: "responsible_FK"                                      */
/*==============================================================*/
create index "responsible_FK" on "responsible" (
   "employee_id" ASC
);

/*==============================================================*/
/* Index: "responsible2_FK"                                     */
/*==============================================================*/
create index "responsible2_FK" on "responsible" (
   "customer_id" ASC
);

alter table "Check_Account"
   add constraint FK_CHECK_AC_ACOUNT_IN_ACOUNT foreign key ("acount_id")
      references "Acount" ("acount_id");

alter table "Employee"
   add constraint FK_EMPLOYEE_MANAGE_EMPLOYEE foreign key ("Emp_employee_id")
      references "Employee" ("employee_id");

alter table "Employee"
   add constraint FK_EMPLOYEE_WORK_FOR_BRANCH foreign key ("branch_name")
      references "Branch" ("branch_name");

alter table "Loan"
   add constraint FK_LOAN_BRANCH_LO_BRANCH foreign key ("branch_name")
      references "Branch" ("branch_name");

alter table "Payment"
   add constraint FK_PAYMENT_PAY_LOAN foreign key ("load_id")
      references "Loan" ("load_id");

alter table "Saving_Accout"
   add constraint FK_SAVING_A_ACOUNT_IN_ACOUNT foreign key ("acount_id")
      references "Acount" ("acount_id");

alter table "borrow"
   add constraint FK_BORROW_BORROW_CUSTOMER foreign key ("customer_id")
      references "Customer" ("customer_id");

alter table "borrow"
   add constraint FK_BORROW_BORROW2_LOAN foreign key ("load_id")
      references "Loan" ("load_id");

alter table "depositor"
   add constraint FK_DEPOSITO_DEPOSITOR_ACOUNT foreign key ("acount_id")
      references "Acount" ("acount_id");

alter table "depositor"
   add constraint FK_DEPOSITO_DEPOSITOR_CUSTOMER foreign key ("customer_id")
      references "Customer" ("customer_id");

alter table "responsible"
   add constraint FK_RESPONSI_RESPONSIB_EMPLOYEE foreign key ("employee_id")
      references "Employee" ("employee_id");

alter table "responsible"
   add constraint FK_RESPONSI_RESPONSIB_CUSTOMER foreign key ("customer_id")
      references "Customer" ("customer_id");

