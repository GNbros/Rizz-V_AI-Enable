#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64M
RVTEST_CODE_BEGIN

  li TESTNUM, 1

  csrw pmpcfg0, zero
  li t0, -1
  csrw pmpaddr0, t0
  csrr t0, pmpaddr0

  neg t1, t0
  and a7, t0, t1

  beqz a7, fail
  srl a7, a7, 1
  beqz a7, pass

#define PMPADDR_Gm1_MASK a7
#define PMPCFG_A_MASK (0x3 << 3)

.macro set_pmpaddr_bit value
.if \value
  csrs pmpaddr0, PMPADDR_Gm1_MASK
.else
  csrc pmpaddr0, PMPADDR_Gm1_MASK
.endif
.endm

.macro set_mode_off
  csrc pmpcfg0, PMPCFG_A_MASK
.endm

.macro set_mode_napot
  csrs pmpcfg0, PMPCFG_A_MASK
.endm

.macro check_pmpaddr_bit expected_value
  li TESTNUM, (2 + \@)
  csrr t6, pmpaddr0
  and t6, t6, PMPADDR_Gm1_MASK
.if \expected_value
  beqz t6, fail
.else
  bnez t6, fail
.endif
.endm

.macro check_pmpaddr_bit_clear
  csrr t6, pmpaddr0
  and t6, t6, PMPADDR_Gm1_MASK
  bnez t6, fail
.endm

  set_mode_napot
  set_pmpaddr_bit 0
  check_pmpaddr_bit 0
  set_pmpaddr_bit 1
  check_pmpaddr_bit 1
  set_mode_off
  check_pmpaddr_bit 0
  set_mode_napot
  check_pmpaddr_bit 1

  set_pmpaddr_bit 0
  set_mode_off
  set_pmpaddr_bit 1
  set_mode_napot
  check_pmpaddr_bit 1

  set_mode_off
  csrc pmpaddr0, zero
  csrs pmpaddr0, zero
  set_mode_napot
  check_pmpaddr_bit 1

  set_mode_off
  not t0, PMPADDR_Gm1_MASK
  csrs pmpaddr0, t0
  set_mode_napot
  check_pmpaddr_bit 0

  j pass

  TEST_PASSFAIL

  .align 2
  .global mtvec_handler
mtvec_handler:
  j fail

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END
