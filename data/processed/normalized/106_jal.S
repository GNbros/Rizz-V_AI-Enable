#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64U
RVTEST_CODE_BEGIN

test_2:
  li  TESTNUM, 2
  li  ra, 0

  jal tp, target_2
linkaddr_2:
  nop
  nop

  j fail

target_2:
  la  sp, linkaddr_2
  bne sp, tp, fail

  TEST_CASE( 3, ra, 3, \
  li  ra, 1; \
  jal zero, 1f; \
  addi ra, ra, 1; \
  addi ra, ra, 1; \
  addi ra, ra, 1; \
  addi ra, ra, 1; \
1:  addi ra, ra, 1; \
  addi ra, ra, 1; \
  )

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END
