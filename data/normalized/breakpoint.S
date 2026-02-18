#*****************************************************************************
# breakpoint.S
#-----------------------------------------------------------------------------
#
# Test breakpoints, if they are implemented.
#

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64M
RVTEST_CODE_BEGIN

  # Set up breakpoint to trap on M-mode fetches.
  li TESTNUM, 2

  # Set tcontrol.mte, otherwise breakpoints are disabled. This may trap,
  # because tcontrol is an optional register.
  la a0, 1f
  csrrw a0, mtvec, a0
  li a1, 0x8
  csrs tcontrol, a1
.p2align 2
1:
  csrw mtvec, a0

  # Enable interrupts; see https://github.com/riscv/riscv-debug-spec/blob/f510a7dd33317d0eee0f26b4fa082cd43a5ac7ea/Sdtrig.tex#L213-L214
  csrsi mstatus, MSTATUS_MIE

  # Skip tselect if hard-wired.
  csrw tselect, x0
  csrr a1, tselect
  bne x0, a1, pass

  la a2, 1f
  csrw tdata2, a2
  li a0, (2 << (__riscv_xlen - 4)) | MCONTROL_M | MCONTROL_EXECUTE
  csrw tdata1, a0
  # Skip if breakpoint type is unsupported.
  csrr a1, tdata1
  bne a0, a1, 2f
  .align 2
1:
  # Trap handler should skip this instruction.
  beqz x0, fail

  # Make sure reads don't trap.
  li TESTNUM, 3
  lw a0, (a2)

2:
  # Set up breakpoint to trap on M-mode reads.
  li TESTNUM, 4
  li a0, (2 << (__riscv_xlen - 4)) | MCONTROL_M | MCONTROL_LOAD
  csrw tdata1, a0
  # Skip if breakpoint type is unsupported.
  csrr a1, tdata1
  bne a0, a1, 2f
  la a2, data1
  csrw tdata2, a2

  # Trap handler should skip this instruction.
  lw a2, (a2)
  beqz a2, fail

  # Make sure writes don't trap.
  li TESTNUM, 5
  sw x0, (a2)

2:
  # Set up breakpoint to trap on M-mode stores.
  li TESTNUM, 6
  li a0, (2 << (__riscv_xlen - 4)) |  MCONTROL_M | MCONTROL_STORE
  csrw tdata1, a0
  # Skip if breakpoint type is unsupported.
  csrr a1, tdata1
  bne a0, a1, 2f

  # Trap handler should skip this instruction.
  sw a2, (a2)

  # Make sure store didn't succeed.
  li TESTNUM, 7
  lw a2, (a2)
  bnez a2, fail

  # Try to set up a second breakpoint.
  li a0, 1
  csrw tselect, a0
  csrr a1, tselect
  bne a0, a1, pass

  li a0, (2 << (__riscv_xlen - 4)) | MCONTROL_M | MCONTROL_LOAD
  csrw tdata1, a0
  la a3, data2
  csrw tdata2, a3

  # Make sure the second breakpoint triggers.
  li TESTNUM, 8
  lw a3, (a3)
  beqz a3, fail

  # Make sure the first breakpoint still triggers.
  li TESTNUM, 10
  la a2, data1
  sw a2, (a2)
  li TESTNUM, 11
  lw a2, (a2)
  bnez a2, fail

2:
  TEST_PASSFAIL

  .align 2
  .global mtvec_handler
mtvec_handler:
  # Only even-numbered tests should trap.
  andi t0, TESTNUM, 1
  bnez t0, fail

  li t0, CAUSE_BREAKPOINT
  csrr t1, mcause
  bne t0, t1, fail

  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

data1: .word 0
data2: .word 0

RVTEST_DATA_END
# See LICENSE for license details.

#include "riscv_test.h"
#undef RVTEST_RV64S
#define RVTEST_RV64S RVTEST_RV64M
#define __MACHINE_MODE

#include "../rv64si/scall.S"
# See LICENSE for license details.

#*****************************************************************************
# lw-unaligned.S
#-----------------------------------------------------------------------------
#
# Test that misaligned loads work or raise the correct exception
# This test assumes the target is little-endian
#

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64M
RVTEST_CODE_BEGIN

  TEST_LD_OP( 2, ld, 0x0807060504030201, 0, tdat );
  TEST_LD_OP( 3, ld, 0x0908070605040302, 1, tdat );
  TEST_LD_OP( 4, ld, 0x0a09080706050403, 2, tdat );
  TEST_LD_OP( 5, ld, 0x0b0a090807060504, 3, tdat );
  TEST_LD_OP( 6, ld, 0x0c0b0a0908070605, 4, tdat );
  TEST_LD_OP( 7, ld, 0x0d0c0b0a09080706, 5, tdat );
  TEST_LD_OP( 8, ld, 0x0e0d0c0b0a090807, 6, tdat );
  TEST_LD_OP( 9, ld, 0x0f0e0d0c0b0a0908, 7, tdat );

2:
  TEST_PASSFAIL

  .align 2
  .global mtvec_handler
mtvec_handler:
  MISALIGNED_LOAD_HANDLER

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

tdat:
    .byte 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08
    .byte 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10

RVTEST_DATA_END
# See LICENSE for license details.

#include "riscv_test.h"
#undef RVTEST_RV64S
#define RVTEST_RV64S RVTEST_RV64M
#define __MACHINE_MODE

#include "../rv64si/sbreak.S"
# See LICENSE for license details.

#*****************************************************************************
# sh-unaligned.S
#-----------------------------------------------------------------------------
#
# Test that misaligned stores work or raise the correct exception
# This test assumes the target is little-endian
#

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64M
RVTEST_CODE_BEGIN

  TEST_ST_OP( 2, lh, sh, 0x1234, 0, tdat );
  TEST_ST_OP( 3, lh, sh, 0x5678, 1, tdat );

2:
  TEST_PASSFAIL

  .align 2
  .global mtvec_handler
mtvec_handler:
  MISALIGNED_STORE_HANDLER

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

tdat:
    .zero 4

RVTEST_DATA_END
# See LICENSE for license details.

#include "riscv_test.h"
#undef RVTEST_RV64S
#define RVTEST_RV64S RVTEST_RV64M
#define __MACHINE_MODE

#include "../rv64si/ma_fetch.S"
# See LICENSE for license details.

#*****************************************************************************
# sw-unaligned.S
#-----------------------------------------------------------------------------
#
# Test that misaligned stores work or raise the correct exception
# This test assumes the target is little-endian
#

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64M
RVTEST_CODE_BEGIN

  TEST_ST_OP( 2, lw, sw, 0x12345678, 0, tdat );
  TEST_ST_OP( 3, lw, sw, 0xffffffff9abcdef0, 1, tdat );
  TEST_ST_OP( 4, lw, sw, 0xffffffffdeadbeef, 2, tdat );
  TEST_ST_OP( 5, lw, sw, 0xfffffffffeed0011, 3, tdat );

2:
  TEST_PASSFAIL

  .align 2
  .global mtvec_handler
mtvec_handler:
  MISALIGNED_STORE_HANDLER

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

tdat:
    .zero 8

RVTEST_DATA_END
# See LICENSE for license details.

