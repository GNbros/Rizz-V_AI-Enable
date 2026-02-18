#*****************************************************************************
# shamt.S
#-----------------------------------------------------------------------------
#
# Test illegal shamt[5] of shift instruction in 32-bit ISA.
#

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV32M
RVTEST_CODE_BEGIN

  # Make sure slli with shamt[4] set is legal.
  TEST_CASE( 2, a0, 65536, li a0, 1; slli a0, a0, 16);

  # Make sure slli with shamt[5] set is not legal.
  TEST_CASE( 3, x0, 1, .word 0x02051513); # slli a0, a0, 32

  TEST_PASSFAIL

.align 2
.global mtvec_handler
mtvec_handler:
  # Trapping on test 3 is good.
  li t0, 3
  bne TESTNUM, t0, fail

  # Make sure CAUSE indicates an illegal instructino.
  csrr t0, mcause
  li t1, CAUSE_ILLEGAL_INSTRUCTION
  bne t0, t1, fail
  j pass

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END
# See LICENSE for license details.

#include "riscv_test.h"
#undef RVTEST_RV64M
#define RVTEST_RV64M RVTEST_RV32M

#include "../rv64mi/illegal.S"
# See LICENSE for license details.

#include "riscv_test.h"
#undef RVTEST_RV64M
#define RVTEST_RV64M RVTEST_RV32M
#define __MACHINE_MODE

#include "../rv64mi/zicntr.S"
# See LICENSE for license details.

#include "riscv_test.h"
#undef RVTEST_RV64M
#define RVTEST_RV64M RVTEST_RV32M
#define __MACHINE_MODE

#include "../rv64mi/breakpoint.S"
# See LICENSE for license details.

#include "riscv_test.h"
#undef RVTEST_RV64S
#define RVTEST_RV64S RVTEST_RV32M
#define __MACHINE_MODE

#include "../rv64si/scall.S"
# See LICENSE for license details.

#include "riscv_test.h"
#undef RVTEST_RV64S
#define RVTEST_RV64S RVTEST_RV32M
#define __MACHINE_MODE

#include "../rv64si/sbreak.S"
# See LICENSE for license details.

#include "riscv_test.h"
#undef RVTEST_RV64M
#define RVTEST_RV64M RVTEST_RV32M
#define __MACHINE_MODE

#include "../rv64mi/sh-misaligned.S"
# See LICENSE for license details.

#include "riscv_test.h"
#undef RVTEST_RV64S
#define RVTEST_RV64S RVTEST_RV32M
#define __MACHINE_MODE

#include "../rv64si/ma_fetch.S"
# See LICENSE for license details.

#include "riscv_test.h"
#undef RVTEST_RV64M
#define RVTEST_RV64M RVTEST_RV32M
#define __MACHINE_MODE

#include "../rv64mi/sw-misaligned.S"
# See LICENSE for license details.

#include "riscv_test.h"
#undef RVTEST_RV64M
#define RVTEST_RV64M RVTEST_RV32M

#include "../rv64mi/ma_addr.S"
# See LICENSE for license details.

#include "riscv_test.h"
#undef RVTEST_RV64M
#define RVTEST_RV64M RVTEST_RV32M
#define __MACHINE_MODE

#include "../rv64mi/instret_overflow.S"
# See LICENSE for license details.

#include "riscv_test.h"
#undef RVTEST_RV64M
#define RVTEST_RV64M RVTEST_RV32M

#include "../rv64mi/mcsr.S"
# See LICENSE for license details.

