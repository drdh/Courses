create or replace procedure ChangeBookID(
       old_id in char,new_id in char) as
begin
       execute immediate 'alter table Borrow disable constraint FK_BID';
       update Book
              set ID=new_id
              where ID=old_id;
              
       update Borrow
              set book_Id=new_id
              where book_Id=old_id;
       
       execute immediate 'alter table Borrow enable constraint FK_BID';
end ChangeBookID;
/
