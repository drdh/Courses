antlr4=java org.antlr.v4.Tool
grun=java org.antlr.v4.runtime.misc.TestRig
arg4lexer=C1Lexer tokens -tokens
arg4gparser=C1 compilationUnit -gui
arg4parser=C1 compilationUnit -tree

all:
	$(antlr4) *.g4
	javac *.java
lexer:
	$(grun) $(arg4lexer) $(f)
gparser:
	$(grun) $(arg4gparser) $(f)
parser:
	$(grun) $(arg4parser) $(f)

test:
	$(grun) $(arg4lexer)

example:
	@echo "\npt_comment.c1\n"
	@$(grun) $(arg4lexer) ../test/test_cases/lexer/pt_comment.c1

	@echo "\n\nft_comment.c1\n"
	@$(grun) $(arg4lexer) ../test/test_cases/lexer/ft_comment.c1

	@echo "\n\npt_number.c1\n"
	@$(grun) $(arg4lexer) ../test/test_cases/lexer/pt_number.c1

	@echo "\n\nft_number.c1\n"
	@$(grun) $(arg4lexer) ../test/test_cases/lexer/ft_number.c1

testParser:
	$(grun) $(arg4parser) 

exampleParser:
	@echo "\npt_function.c1\n"
	$(grun) $(arg4parser) ../test/test_cases/parser/pt_function.c1

	@echo "\nft_function.c1\n"
	$(grun) $(arg4parser) ../test/test_cases/parser/ft_function.c1

	@echo "\npt_cond.c1\n"
	$(grun) $(arg4parser) ../test/test_cases/parser/pt_cond.c1

	@echo "\nft_cond.c1\n"
	$(grun) $(arg4parser) ../test/test_cases/parser/ft_cond.c1

	@echo "\npt_exp.c1\n"
	$(grun) $(arg4parser) ../test/test_cases/parser/pt_exp.c1

	@echo "\nft_exp.c1\n"
	$(grun) $(arg4parser) ../test/test_cases/parser/ft_exp.c1

exampleGParser:
	@echo "\npt_function.c1\n"
	$(grun) $(arg4gparser) ../test/test_cases/parser/pt_function.c1

	@echo "\nft_function.c1\n"
	$(grun) $(arg4gparser) ../test/test_cases/parser/ft_function.c1

	@echo "\npt_cond.c1\n"
	$(grun) $(arg4gparser) ../test/test_cases/parser/pt_cond.c1

	@echo "\nft_cond.c1\n"
	$(grun) $(arg4gparser) ../test/test_cases/parser/ft_cond.c1

	@echo "\npt_exp.c1\n"
	$(grun) $(arg4gparser) ../test/test_cases/parser/pt_exp.c1

	@echo "\nft_exp.c1\n"
	$(grun) $(arg4gparser) ../test/test_cases/parser/ft_exp.c1
clean:
	rm -rf *.java *.tokens *.interp *.class .antlr
