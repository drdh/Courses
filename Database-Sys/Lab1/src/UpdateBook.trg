create or replace trigger UpdateBook
  after insert or update
  on borrow 
  for each row
declare
  -- local variables here
begin
     if :new.return_date is not null then
        update book set status=0 where id=:new.book_id;
     else 
        update book set status=1 where id=:new.book_id;
     end if;
end UpdateBook;
/
