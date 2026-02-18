#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64UF
RVTEST_CODE_BEGIN

    TEST_FP_INT_OP_H( 2,  fcvt.w.h, 0x01,         -1,    -1.1, rtz);
    TEST_FP_INT_OP_H( 3,  fcvt.w.h, 0x00,         -1,    -1.0, rtz);
    TEST_FP_INT_OP_H( 4,  fcvt.w.h, 0x01,          0,    -0.9, rtz);
    TEST_FP_INT_OP_H( 5,  fcvt.w.h, 0x01,          0,     0.9, rtz);
    TEST_FP_INT_OP_H( 6,  fcvt.w.h, 0x00,          1,     1.0, rtz);
    TEST_FP_INT_OP_H( 7,  fcvt.w.h, 0x01,          1,     1.1, rtz);
    TEST_FP_INT_OP_H( 8,  fcvt.w.h, 0x00,      -2054, 0h:e803, rtz);
    TEST_FP_INT_OP_H( 9,  fcvt.w.h, 0x00,       2054, 0h:6803, rtz);

    TEST_FP_INT_OP_H(12, fcvt.wu.h, 0x10,          0,    -3.0, rtz);
    TEST_FP_INT_OP_H(13, fcvt.wu.h, 0x10,          0,    -1.0, rtz);
    TEST_FP_INT_OP_H(14, fcvt.wu.h, 0x01,          0,    -0.9, rtz);
    TEST_FP_INT_OP_H(15, fcvt.wu.h, 0x01,          0,     0.9, rtz);
    TEST_FP_INT_OP_H(16, fcvt.wu.h, 0x00,          1,     1.0, rtz);
    TEST_FP_INT_OP_H(17, fcvt.wu.h, 0x01,          1,     1.1, rtz);
    TEST_FP_INT_OP_H(18, fcvt.wu.h, 0x10,          0, 0h:e803, rtz);
    TEST_FP_INT_OP_H(19, fcvt.wu.h, 0x00,       2054, 0h:6803, rtz);

#if __riscv_xlen >= 64
    TEST_FP_INT_OP_H(22,  fcvt.l.h, 0x01,         -1,    -1.1, rtz);
    TEST_FP_INT_OP_H(23,  fcvt.l.h, 0x00,         -1,    -1.0, rtz);
    TEST_FP_INT_OP_H(24,  fcvt.l.h, 0x01,          0,    -0.9, rtz);
    TEST_FP_INT_OP_H(25,  fcvt.l.h, 0x01,          0,     0.9, rtz);
    TEST_FP_INT_OP_H(26,  fcvt.l.h, 0x00,          1,     1.0, rtz);
    TEST_FP_INT_OP_H(27,  fcvt.l.h, 0x01,          1,     1.1, rtz);

    TEST_FP_INT_OP_H(32, fcvt.lu.h, 0x10,          0,    -3.0, rtz);
    TEST_FP_INT_OP_H(33, fcvt.lu.h, 0x10,          0,    -1.0, rtz);
    TEST_FP_INT_OP_H(34, fcvt.lu.h, 0x01,          0,    -0.9, rtz);
    TEST_FP_INT_OP_H(35, fcvt.lu.h, 0x01,          0,     0.9, rtz);
    TEST_FP_INT_OP_H(36, fcvt.lu.h, 0x00,          1,     1.0, rtz);
    TEST_FP_INT_OP_H(37, fcvt.lu.h, 0x01,          1,     1.1, rtz);
    TEST_FP_INT_OP_H(38, fcvt.lu.h, 0x10,          0, 0h:e483, rtz);
#endif

    TEST_CASE( 42, ra, 0x000000007fffffff, la ra, tdat  ; flw f1,  0(ra); fcvt.w.h ra, f1)
    TEST_CASE( 43, ra, 0xffffffff80000000, la ra, tdat  ; flw f1,  8(ra); fcvt.w.h ra, f1)
#if __riscv_xlen >= 64
    TEST_CASE( 44, ra, 0x7fffffffffffffff, la ra, tdat  ; flw f1,  0(ra); fcvt.l.h ra, f1)
    TEST_CASE( 45, ra, 0x8000000000000000, la ra, tdat  ; flw f1,  8(ra); fcvt.l.h ra, f1)
#endif

    TEST_CASE( 52, ra, 0x000000007fffffff, la ra, tdat  ; flw f1,  4(ra); fcvt.w.h ra, f1)
    TEST_CASE( 53, ra, 0x000000007fffffff, la ra, tdat  ; flw f1, 12(ra); fcvt.w.h ra, f1)
#if __riscv_xlen >= 64
    TEST_CASE( 54, ra, 0x7fffffffffffffff, la ra, tdat  ; flw f1,  4(ra); fcvt.l.h ra, f1)
    TEST_CASE( 55, ra, 0x7fffffffffffffff, la ra, tdat  ; flw f1, 12(ra); fcvt.l.h ra, f1)
#endif

    TEST_CASE( 62, ra, 0xffffffffffffffff, la ra, tdat  ; flw f1,  0(ra); fcvt.wu.h ra, f1)
    TEST_CASE( 63, ra, 0xffffffffffffffff, la ra, tdat  ; flw f1,  4(ra); fcvt.wu.h ra, f1)
    TEST_CASE( 64, ra,                  0, la ra, tdat  ; flw f1,  8(ra); fcvt.wu.h ra, f1)
    TEST_CASE( 65, ra, 0xffffffffffffffff, la ra, tdat  ; flw f1, 12(ra); fcvt.wu.h ra, f1)
#if __riscv_xlen >= 64
    TEST_CASE( 66, ra, 0xffffffffffffffff, la ra, tdat  ; flw f1,  0(ra); fcvt.lu.h ra, f1)
    TEST_CASE( 67, ra, 0xffffffffffffffff, la ra, tdat  ; flw f1,  4(ra); fcvt.lu.h ra, f1)
    TEST_CASE( 68, ra,                  0, la ra, tdat  ; flw f1,  8(ra); fcvt.lu.h ra, f1)
    TEST_CASE( 69, ra, 0xffffffffffffffff, la ra, tdat  ; flw f1, 12(ra); fcvt.lu.h ra, f1)
#endif

    TEST_PASSFAIL

RVTEST_CODE_END

    .data
RVTEST_DATA_BEGIN

    TEST_DATA

tdat:
.word 0xffffffff
.word 0xffff7fff
.word 0xfffffc00
.word 0xffff7c00

RVTEST_DATA_END

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64M
RVTEST_CODE_BEGIN

    la  ra, tdat
    .word 0x0040A00F
    TEST_LD_OP( 1, ld, 0, 0,  tdat )
    TEST_LD_OP( 2, ld, 0, 8,  tdat )
    TEST_LD_OP( 3, ld, 0, 16, tdat )
    TEST_LD_OP( 4, ld, 0, 24, tdat )
    TEST_LD_OP( 5, ld, 0, 32, tdat )
    TEST_LD_OP( 6, ld, 0, 40, tdat )
    TEST_LD_OP( 7, ld, 0, 48, tdat )
    TEST_LD_OP( 8, ld, 0, 56, tdat )

    j pass

    TEST_PASSFAIL

RVTEST_CODE_END

    .data
RVTEST_DATA_BEGIN

    TEST_DATA

tdat:  .dword 0xdeadbeefdeadbeef

RVTEST_DATA_END
