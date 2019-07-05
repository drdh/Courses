cd ../../build
make
cd ../test/DeclCheckTest
../../build/src/google-cpp-coding-style-checker -no-class-check -no-stmt-check -no-pp-check
