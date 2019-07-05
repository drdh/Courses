#!/bin/sh

LLVMAlgebraic=../llvm-7.0.1.src/build/lib/LLVMAlgebraic.so

echo "********************"
echo "clang -O3"
clang -O3 test.c
time  --format "\nreal %e\nuser %U\nsys %S\n" ./a.out < data.txt

echo "********************"
echo "gcc -O3"
gcc -O3 test.c
time  --format "\nreal %e\nuser %U\nsys %S\n" ./a.out < data.txt

echo "********************"
echo "clang -O3 with our AlgOpt"
clang -O3 test.c -c -emit-llvm -o test.bc
opt -load $LLVMAlgebraic -algebraic < test.bc > test_optimized.bc
clang -O3 test_optimized.bc
time  --format "\nreal %e\nuser %U\nsys %S\n" ./a.out < data.txt

echo "********************"
