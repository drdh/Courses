CC=g++
CFLAGS1=-g -O3  -fpermissive -w
CFLAGS2+=-fprofile-arcs -ftest-coverage -w
TARGET1=main
TARGET2=WordListTest
SRCS=main.cpp WordChain.cpp 
OBJS=$(SRCS:.cpp=.o)
LIB=/usr/local/lib/libgtest.a		#google test的一个库，缺少可能无法运行

${TARGET2}: WordChain.h WordChain.cpp WordListTest.cpp ${TARGET1}
	${CC} ${CFLAGS2} WordChain.cpp WordListTest.cpp -lpthread ${LIB} -o ${TARGET2}

$(TARGET1):$(OBJS)
	$(CC) -o $@ $^

%.o:%.cpp
	$(CC) $(CFLAGS1) -std=c++11 -o $@ -c $<

.PHONY: html
html:
	gcov ${TARGET2}.cpp
	lcov -c -d . -o ${TARGET2}.info
	genhtml -o result ${TARGET2}.info

.PHONY: clean
clean:
	rm -rf $(TARGET1) $(OBJS) solution.txt *.gcov *.gcda *.gcno ${TARGET2} *.info ./result
