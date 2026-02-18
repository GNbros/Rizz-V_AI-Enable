#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64M
RVTEST_CODE_BEGIN

#if __riscv_xlen == 64
  TEST_CASE(2, a0, 0x2, csrr a0, misa; srl a0, a0, 62)
#else
  TEST_CASE(2, a0, 0x1, csrr a0, misa; srl a0, a0, 30)
#endif

  TEST_CASE(3, a0, 0x0, csrr a0, mhartid)

  csrr a0, mimpid
  csrr a0, marchid
  csrr a0, mvendorid

  li t0, 0
  csrs mtvec, t0
  csrs mepc, t0

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END
