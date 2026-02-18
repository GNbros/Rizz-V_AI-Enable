#*****************************************************************************
# fcvt_w.S
#-----------------------------------------------------------------------------
#
# Test fcvt{wu|w|lu|l}.s instructions.
#

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64UF
RVTEST_CODE_BEGIN

  #-------------------------------------------------------------
  # Arithmetic tests
  #-------------------------------------------------------------

  TEST_FP_INT_OP_S( 2,  fcvt.w.s, 0x01,         -1, -1.1, rtz);
  TEST_FP_INT_OP_S( 3,  fcvt.w.s, 0x00,         -1, -1.0, rtz);
  TEST_FP_INT_OP_S( 4,  fcvt.w.s, 0x01,          0, -0.9, rtz);
  TEST_FP_INT_OP_S( 5,  fcvt.w.s, 0x01,          0,  0.9, rtz);
  TEST_FP_INT_OP_S( 6,  fcvt.w.s, 0x00,          1,  1.0, rtz);
  TEST_FP_INT_OP_S( 7,  fcvt.w.s, 0x01,          1,  1.1, rtz);
  TEST_FP_INT_OP_S( 8,  fcvt.w.s, 0x10,     -1<<31, -3e9, rtz);
  TEST_FP_INT_OP_S( 9,  fcvt.w.s, 0x10,  (1<<31)-1,  3e9, rtz);

  TEST_FP_INT_OP_S(12, fcvt.wu.s, 0x10,          0, -3.0, rtz);
  TEST_FP_INT_OP_S(13, fcvt.wu.s, 0x10,          0, -1.0, rtz);
  TEST_FP_INT_OP_S(14, fcvt.wu.s, 0x01,          0, -0.9, rtz);
  TEST_FP_INT_OP_S(15, fcvt.wu.s, 0x01,          0,  0.9, rtz);
  TEST_FP_INT_OP_S(16, fcvt.wu.s, 0x00,          1,  1.0, rtz);
  TEST_FP_INT_OP_S(17, fcvt.wu.s, 0x01,          1,  1.1, rtz);
  TEST_FP_INT_OP_S(18, fcvt.wu.s, 0x10,          0, -3e9, rtz);
  TEST_FP_INT_OP_S(19, fcvt.wu.s, 0x00, 3000000000,  3e9, rtz);

#if __riscv_xlen >= 64
  TEST_FP_INT_OP_S(22,  fcvt.l.s, 0x01,         -1, -1.1, rtz);
  TEST_FP_INT_OP_S(23,  fcvt.l.s, 0x00,         -1, -1.0, rtz);
  TEST_FP_INT_OP_S(24,  fcvt.l.s, 0x01,          0, -0.9, rtz);
  TEST_FP_INT_OP_S(25,  fcvt.l.s, 0x01,          0,  0.9, rtz);
  TEST_FP_INT_OP_S(26,  fcvt.l.s, 0x00,          1,  1.0, rtz);
  TEST_FP_INT_OP_S(27,  fcvt.l.s, 0x01,          1,  1.1, rtz);

  TEST_FP_INT_OP_S(32, fcvt.lu.s, 0x10,          0, -3.0, rtz);
  TEST_FP_INT_OP_S(33, fcvt.lu.s, 0x10,          0, -1.0, rtz);
  TEST_FP_INT_OP_S(34, fcvt.lu.s, 0x01,          0, -0.9, rtz);
  TEST_FP_INT_OP_S(35, fcvt.lu.s, 0x01,          0,  0.9, rtz);
  TEST_FP_INT_OP_S(36, fcvt.lu.s, 0x00,          1,  1.0, rtz);
  TEST_FP_INT_OP_S(37, fcvt.lu.s, 0x01,          1,  1.1, rtz);
  TEST_FP_INT_OP_S(38, fcvt.lu.s, 0x10,          0, -3e9, rtz);
#endif

  # test negative NaN, negative infinity conversion
  TEST_CASE( 42, x1, 0x000000007fffffff, la x1, tdat  ; flw f1,  0(x1); fcvt.w.s x1, f1)
  TEST_CASE( 44, x1, 0xffffffff80000000, la x1, tdat  ; flw f1,  8(x1); fcvt.w.s x1, f1)
#if __riscv_xlen >= 64
  TEST_CASE( 43, x1, 0x7fffffffffffffff, la x1, tdat  ; flw f1,  0(x1); fcvt.l.s x1, f1)
  TEST_CASE( 45, x1, 0x8000000000000000, la x1, tdat  ; flw f1,  8(x1); fcvt.l.s x1, f1)
#endif

  # test positive NaN, positive infinity conversion
  TEST_CASE( 52, x1, 0x000000007fffffff, la x1, tdat  ; flw f1,  4(x1); fcvt.w.s x1, f1)
  TEST_CASE( 54, x1, 0x000000007fffffff, la x1, tdat  ; flw f1, 12(x1); fcvt.w.s x1, f1)
#if __riscv_xlen >= 64
  TEST_CASE( 53, x1, 0x7fffffffffffffff, la x1, tdat  ; flw f1,  4(x1); fcvt.l.s x1, f1)
  TEST_CASE( 55, x1, 0x7fffffffffffffff, la x1, tdat  ; flw f1, 12(x1); fcvt.l.s x1, f1)
#endif

  # test NaN, infinity conversions to unsigned integer
  TEST_CASE( 62, x1, 0xffffffffffffffff, la x1, tdat  ; flw f1,  0(x1); fcvt.wu.s x1, f1)
  TEST_CASE( 63, x1, 0xffffffffffffffff, la x1, tdat  ; flw f1,  4(x1); fcvt.wu.s x1, f1)
  TEST_CASE( 64, x1,                  0, la x1, tdat  ; flw f1,  8(x1); fcvt.wu.s x1, f1)
  TEST_CASE( 65, x1, 0xffffffffffffffff, la x1, tdat  ; flw f1, 12(x1); fcvt.wu.s x1, f1)
#if __riscv_xlen >= 64
  TEST_CASE( 66, x1, 0xffffffffffffffff, la x1, tdat  ; flw f1,  0(x1); fcvt.lu.s x1, f1)
  TEST_CASE( 67, x1, 0xffffffffffffffff, la x1, tdat  ; flw f1,  4(x1); fcvt.lu.s x1, f1)
  TEST_CASE( 68, x1,                  0, la x1, tdat  ; flw f1,  8(x1); fcvt.lu.s x1, f1)
  TEST_CASE( 69, x1, 0xffffffffffffffff, la x1, tdat  ; flw f1, 12(x1); fcvt.lu.s x1, f1)
#endif

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

# -NaN, NaN, -inf, +inf
tdat:
.word 0xffffffff
.word 0x7fffffff
.word 0xff800000
.word 0x7f800000

tdat_d:
.dword 0xffffffffffffffff
.dword 0x7fffffffffffffff
.dword 0xfff0000000000000
.dword 0x7ff0000000000000

RVTEST_DATA_END
# See LICENSE for license details.

#*****************************************************************************
# amoor.w.S
#-----------------------------------------------------------------------------
#
# Test amoor.w instruction.
#

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64U
RVTEST_CODE_BEGIN

  TEST_CASE(2, a4, 0xffffffff80000000, \
    li a0, 0xffffffff80000000; \
    li a1, 0xfffffffffffff800; \
    la a3, amo_operand; \
    sw a0, 0(a3); \
    amoor.w	a4, a1, 0(a3); \
  )

  TEST_CASE(3, a5, 0xfffffffffffff800, lw a5, 0(a3))

  # try again after a cache miss
  TEST_CASE(4, a4, 0xfffffffffffff800, \
    li  a1, 1; \
    amoor.w a4, a1, 0(a3); \
  )

  TEST_CASE(5, a5, 0xfffffffffffff801, lw a5, 0(a3))

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
# See LICENSE for license details.

