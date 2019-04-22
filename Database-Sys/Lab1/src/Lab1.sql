Drop Table Borrow;
Drop Table Book;
Drop Table Reader;
drop view Info;

Create Table Book(
    ID char(8) Constraint PK_BID Primary Key,
    name varchar2(10) Not NULL,
    author varchar2(10),
    price float,
    status int default 0
);

Create Table Reader(
    ID char(8) constraint PK_RID Primary Key,
    name varchar2(10),
    age int,
    address varchar2(20)
);

Create Table Borrow(
    book_ID char(8) Constraint FK_BID References Book(ID),
    reader_ID char(8) Constraint FK_RID References Reader(ID),
    borrow_Date date,
    return_Date date,
    Constraint PK_Bo Primary Key(book_ID,reader_ID)
);

--book
Insert Into Book Values('b1','Oracle1','Ullman',13.6,0);
Insert Into Book Values('b2','2Oracle','Ullman',34,0);
Insert Into Book Values('b3','database','drdh',234,0);
Insert Into Book Values('b4','system','drdh',22,0);
Insert Into Book Values('b5','webinfo','jpq',34,0);
Insert Into Book Values('b6','os','osh',33,0);

Insert Into Book Values('b10','os2','os2',33,0);

--reader
Insert Into Reader Values('r1','Rose',18,'Hefei');
Insert Into Reader Values('r2','jack',21,'Hefei');
Insert Into Reader Values('r3','李林',21,'Beijing');
Insert Into Reader Values('r4','张三',21,'Beijing');

--borrow
Insert Into Borrow 
       Values('b1','r1',to_date('01-03-2019','dd-mm-yy'),
                   to_date('03-03-2019','dd-mm-yy'));
                   
Insert Into Borrow(book_ID,reader_ID,borrow_Date) 
       Values('b2','r1',to_date('01-03-2019','dd-mm-yy'));
 
Insert Into Borrow(book_ID,reader_ID,borrow_Date) 
       Values('b3','r2',to_date('01-03-2019','dd-mm-yy'));
       
Insert Into Borrow(book_ID,reader_ID,borrow_Date) 
       Values('b3','r3',to_date('01-03-2019','dd-mm-yy'));
       
Insert Into Borrow 
       Values('b4','r3',to_date('01-03-2019','dd-mm-yy'),
                   to_date('03-03-2019','dd-mm-yy'));
        
Insert Into Borrow 
       Values('b5','r3',to_date('01-03-2019','dd-mm-yy'),
                   to_date('03-03-2019','dd-mm-yy'));
               
Insert Into Borrow 
       Values('b6','r3',to_date('01-03-2019','dd-mm-yy'),
                   to_date('03-03-2019','dd-mm-yy'));       

--(1)
Select ID,address
       from Reader
       where name='Rose';

--(2)
Select bk.name,br.borrow_Date
       from Book bk,Reader rd,Borrow br
       where br.book_ID=bk.ID
             and br.Reader_ID=rd.ID
             and rd.name='Rose';
--(3)             
Select rd.name
       from Reader rd
       where rd.ID not in (select distinct br.reader_ID from Borrow br); 

--(4)
select name,price
       from Book 
       where author='Ullman';
       
--(5)
select bk.ID,bk.name
       from Book bk,Reader rd,Borrow br
       where bk.ID=br.book_id
             and rd.ID=br.reader_id
             and rd.name='李林'
             and br.return_date is null;       

--(6)
select rd2.name 
       from (select br.reader_ID,count(br.book_ID) as count_bk 
                    from Borrow br
                    group by br.reader_ID)br2, Reader rd2
        where br2.reader_ID=rd2.ID
              and br2.count_bk>3; 

--(7)
select rd.name,rd.ID
       from Reader rd
       where not exists
             (select * from Borrow br
                     where br.book_ID in (
                           select distinct br2.book_ID
                                  from Borrow br2,Reader rd2
                                  where br2.reader_id=rd2.ID
                                        and rd2.name='李林'
                               
                     ) and br.reader_id=rd.ID
                     );


--(8)
select name,ID
       from Book
       where name like '%Oracle%';

--(9)
Create view Info(reader_ID,reader_name,book_ID,book_name,borrow_date)
       as Select rd.ID,rd.name,bk.ID,bk.name,br.borrow_date
                 from Book bk,Reader rd,Borrow br
                 where bk.ID=br.book_ID
                       and rd.ID=br.reader_id;


Select reader_ID,count(distinct book_ID)
       from Info
       where borrow_date>=add_months(sysdate,-12)
       group by reader_ID;


--违反完整性约束
/*
update Book
       set ID='b2'
       where ID='b1';
*/

/*
Select * from Book;
Select * from Reader;
Select * from Borrow;
*/

/*
Drop Table Borrow;
Drop Table Book;
Drop Table Reader;
*/
