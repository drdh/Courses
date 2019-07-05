all: 
	@rm -rf fib_gen.ll fib_gen
	@c++ `llvm-config --cxxflags --ldflags` gen_fib.cpp `llvm-config --libs --system-libs` -o fib_gen
	@./fib_gen >fib_gen.ll
	@lli fib_gen.ll;echo $?

clean:
	rm fib_gen -rf
