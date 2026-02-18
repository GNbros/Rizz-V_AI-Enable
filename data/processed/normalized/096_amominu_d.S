#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64U
RVTEST_CODE_BEGIN

  TEST_CASE(2, a4, 0xffffffff80000000, \
  li a0, 0xffffffff80000000; \
  li a1, 0xfffffffffffff800; \
  la a3, amo_operand; \
  sw a0, 0(a3); \
  amominu.w	a4, a1, 0(a3); \
  )

  TEST_CASE(3, a5, 0xffffffff80000000, lw a5, 0(a3))

  TEST_CASE(4, a4, 0, \
  li a1, 0xffffffffffffffff; \
  sw zero, 0(a3); \
  amominu.w	a4, a1, 0(a3); \
  )

  TEST_CASE(5, a5, 0, lw a5, 0(a3))

  #if __riscv_xlen == 64
  TEST_CASE(6, a4, 1, \
  li a0, 0x0000000000000001; \
  li a1, 0x8000000000000000; \
  la a3, amo_operand; \
  sw a0, 0(a3); \
  amominu.w	a4, a1, 0(a3); \
  )

  TEST_CASE(7, a5, 0, lw a5, 0(a3))
  #endif

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END

  .bss
  .align 3
amo_operand:
  .dword 0

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64U
RVTEST_CODE_BEGIN

  TEST_CASE(2, a4, 0xffffffff80000000, \
  li a0, 0xffffffff80000000; \
  li a1, 0xfffffffffffff800; \
  la a3, amo_operand; \
  sd a0, 0(a3); \
  amoswap.d	a4, a1, 0(a3); \
  )

  TEST_CASE(3, a5, 0xfffffffffffff800, ld a5, 0(a3))

  TEST_CASE(4, a4, 0xfffffffffffff800, \
  li  a1, 0x0000000080000000; \
  amoswap.d a4, a1, 0(a3); \
  )

  TEST_CASE(5, a5, 0x0000000080000000, ld a5, 0(a3))

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END

  .bss
  .align 3
amo_operand:
  .dword 0
