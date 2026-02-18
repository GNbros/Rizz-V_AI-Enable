#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64U
RVTEST_CODE_BEGIN

  TEST_RR_OP( 2, remuw,   2,  20,   6 );
  TEST_RR_OP( 3, remuw,   2, -20,   6 );
  TEST_RR_OP( 4, remuw,  20,  20,  -6 );
  TEST_RR_OP( 5, remuw, -20, -20,  -6 );

  TEST_RR_OP( 6, remuw,      0, -1<<31,  1 );
  TEST_RR_OP( 7, remuw, -1<<31, -1<<31, -1 );

  TEST_RR_OP( 8, remuw, -1<<31, -1<<31, 0 );
  TEST_RR_OP( 9, remuw,      1,      1, 0 );
  TEST_RR_OP(10, remuw,      0,      0, 0 );

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END

#include "riscv_test.h"
#undef RVTEST_RV64M
#define RVTEST_RV64M RVTEST_RV32M
#define __MACHINE_MODE

#include "../rv64mi/lh-misaligned.S"

#include "riscv_test.h"
#undef RVTEST_RV64S
#define RVTEST_RV64S RVTEST_RV32M
#define __MACHINE_MODE

#include "../rv64si/csr.S"

#include "riscv_test.h"
#undef RVTEST_RV64M
#define RVTEST_RV64M RVTEST_RV32M
#define __MACHINE_MODE

#include "../rv64mi/lw-misaligned.S"
