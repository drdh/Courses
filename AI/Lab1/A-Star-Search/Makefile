all:
	@rm A_Star IDA_Star output_A.txt output_IDA.txt -rf
	@g++ A_Star.cpp -o A_Star -O3 -g
	@g++ IDA_Star.cpp -o IDA_Star -O3 -g
	@#./A_Star 
	@#echo -e "A*:"
	@#cat output_A.txt 
	@#./IDA_Star
	@#echo -e "\nIDA*:"
	@#cat output_IDA.txt

clean: 
	@rm A_Star IDA_Star output_A.txt output_IDA.txt -rf
