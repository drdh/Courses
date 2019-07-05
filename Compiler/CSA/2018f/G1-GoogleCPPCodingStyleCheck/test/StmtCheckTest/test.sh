#!/bin/bash

echo checktype is 6
checktype=6

if [ -z $1 ]; then
	all=`ls | grep -v 'test.sh'`
else
	all=$@
	for j in $all; do
		if [ ! -f $j ]; then
			echo "File not exist: $j" 
			echo "Usage: $0 [<test_file_1> ... ]"
			exit
		fi
	done
fi

cd ../../build/
make
cd ../test/StmtCheckTest
echo ""

for i in $all; do
	echo -e "\033[42;1m -------------------------- $i --------------------------------- \033[0m\n"
	../../build/src/google-cpp-coding-style-checker $i -c $checktype
	echo -e "\n\n"
done
