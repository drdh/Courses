all: pre_make test_3

pre_make:
	cd hiredis && make clean && make && cd ..

test_3: 3_1.c 3_2.c 3_3.c 3_4.c 3_5.c
	gcc 3_1.c -o 3 -L./hiredis/ -lhiredis -g 
	gcc 3_2.c -o 3 -L./hiredis/ -lhiredis -g
	gcc 3_3.c -o 3 -L./hiredis/ -lhiredis -g
	gcc 3_4.c -o 3 -L./hiredis/ -lhiredis -g
	gcc 3_5.c -o 3 -L./hiredis/ -lhiredis -g

clean:
	rm 3 *.o
