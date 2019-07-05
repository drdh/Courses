#!/bin/sh

LLVMAlgebraic=/media/dingfeng/NewDisk/llvm/build/lib/LLVMAlgebraic.so

clang -O2 test2.c -fno-inline -emit-llvm -c -o test2.bc
clang -S -emit-llvm -O2 test2.c -fno-inline -o test2.ll
opt -load $LLVMAlgebraic -algebraic < test2.bc > test_my2.ll
echo "This is clang -O2 . test.ll "
time  --format "\nreal %e\nuser %U\nsys %S\n" lli test2.ll < data.txt
echo "This is our AlgOpt . test_my.ll"
time  --format "\nreal %e\nuser %U\nsys %S\n" lli test_my2.ll < data.txt