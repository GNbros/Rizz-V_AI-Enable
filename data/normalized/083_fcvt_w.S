#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64UF
RVTEST_CODE_BEGIN

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

    TEST_CASE( 42, ra, 0x000000007fffffff, la ra, tdat  ; flw f1,  0(ra); fcvt.w.s ra, f1)
    TEST_CASE( 44, ra, 0xffffffff80000000, la ra, tdat  ; flw f1,  8(ra); fcvt.w.s ra, f1)
#if __riscv_xlen >= 64
    TEST_CASE( 43, ra, 0x7fffffffffffffff, la ra, tdat  ; flw f1,  0(ra); fcvt.l.s ra, f1)
    TEST_CASE( 45, ra, 0x8000000000000000, la ra, tdat  ; flw f1,  8(ra); fcvt.l.s ra, f1)
#endif

    TEST_CASE( 52, ra, 0x000000007fffffff, la ra, tdat  ; flw f1,  4(ra); fcvt.w.s ra, f1)
    TEST_CASE( 54, ra, 0x000000007fffffff, la ra, tdat  ; flw f1, 12(ra); fcvt.w.s ra, f1)
#if __riscv_xlen >= 64
    TEST_CASE( 53, ra, 0x7fffffffffffffff, la ra, tdat  ; flw f1,  4(ra); fcvt.l.s ra, f1)
    TEST_CASE( 55, ra, 0x7fffffffffffffff, la ra, tdat  ; flw f1, 12(ra); fcvt.l.s ra, f1)
#endif

    TEST_CASE( 62, ra, 0xffffffffffffffff, la ra, tdat  ; flw f1,  0(ra); fcvt.wu.s ra, f1)
    TEST_CASE( 63, ra, 0xffffffffffffffff, la ra, tdat  ; flw f1,  4(ra); fcvt.wu.s ra, f1)
    TEST_CASE( 64, ra,                  0, la ra, tdat  ; flw f1,  8(ra); fcvt.wu.s ra, f1)
    TEST_CASE( 65, ra, 0xffffffffffffffff, la ra, tdat  ; flw f1, 12(ra); fcvt.wu.s ra, f1)
#if __riscv_xlen >= 64
    TEST_CASE( 66, ra, 0xffffffffffffffff, la ra, tdat  ; flw f1,  0(ra); fcvt.lu.s ra, f1)
    TEST_CASE( 67, ra, 0xffffffffffffffff, la ra, tdat  ; flw f1,  4(ra); fcvt.lu.s ra, f1)
    TEST_CASE( 68, ra,                  0, la ra, tdat  ; flw f1,  8(ra); fcvt.lu.s ra, f1)
    TEST_CASE( 69, ra, 0xffffffffffffffff, la ra, tdat  ; flw f1, 12(ra); fcvt.lu.s ra, f1)
#endif

    TEST_PASSFAIL

RVTEST_CODE_END

    .data
RVTEST_DATA_BEGIN

    TEST_DATA

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
