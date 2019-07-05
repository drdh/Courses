all:
	@g++ RedBlackTree.cpp -g -o RedBlackTree 
	@g++ IntervalTree.cpp -g -o IntervalTree

test:
	@echo -e "RedBlackTree:\n"
	@./RedBlackTree 
	@dot -Tpng RBTree.dot -o RBTree.png
	@eog RBTree.png

	@echo -e "IntervalTree:\n"
	@./IntervalTree 
	@dot -Tpng IntervalTree.dot -o IntervalTree.png
	@eog IntervalTree.png

clean:
	@rm RedBlackTree IntervalTree RBTree.dot RBTree.png  IntervalTree.dot IntervalTree.png -rf
