all:
	@g++ BFS_small.cpp -o BFS_small
	@g++ BFS_Huge.cpp -o BFS_huge
	@g++ BFS_Random.cpp -o BFS_rand
	@g++ BFS_Huge_Test.cpp -o BFS_Huge_Test -O3
clean:
	@rm -rf BFS_small BFS_rand BFS_huge BFS_Huge_Test
test:
	@g++ test.cpp -o test
	@./test
	@rm -rf test
