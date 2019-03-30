#include "./include/riscv_test.h"
#include "./include/test_macros.h"

RVTEST_CODE_BEGIN
li x1,1
li x2,2
bne x1,x2,target
li x3,3
target:li x4,4
la x5,target



RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END