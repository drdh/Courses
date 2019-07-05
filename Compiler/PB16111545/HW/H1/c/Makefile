test:test.o
	gcc -o test test.o
test.o:test.s
	gcc -c test.s
test.s:test.c
	gcc -S test.c
	gcc -S test.c -m32 -o test32.s
	gcc -E test.c > test.txt

clean:
	rm -rf test test.o test.s test.txt test32.s
