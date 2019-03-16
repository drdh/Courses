CC=g++
CFLAGS=-g -O3 
TARGET=main
SRCS=main.cpp LongestWordChain.cpp 
OBJS = $(SRCS:.cpp=.o)
$(TARGET):$(OBJS)
	$(CC) -o $@ $^

%.o:%.cpp
	$(CC) $(CFLAGS) -o $@ -c $<

clean:
	rm -rf $(TARGET) $(OBJS) solution.txt
