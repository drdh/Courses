#!/bin/bash

lib="./LoadOpt.so -labopt"
if [ $1 = "-l" ]
then
    if [ $# -ne 5 ]
    then
        echo "./run.sh [-l <location_of_so_file> <structure>] <source_code>"
        exit
    fi
    lib=$2' '$3
    filename=$4
elif [ $1 = "-h" ]
then
    echo "./run.sh [-l <location_of_so_file> <structure>] <source_code>"
    exit
else
    filename=$1
fi

name=$(echo "$filename" | cut -f 1 -d '.')
ext=$(echo "$filename" | cut -f 2 -d '.')
if [ $ext = "c" -o $ext = "cpp" ]
then
    clang -S $filename -emit-llvm -c -o $name.bc
elif [ $ext = "ll" ]
then
    llvm-as $filename -o $name.bc
elif [ $ext = "bc" ]
then
    echo $filename
else
    echo "The format of input file must be '.c', '.cpp', '.ll' or '.bc'!"
fi
opt -load $lib -disable-output $name.bc 2> $name-opt.ll
