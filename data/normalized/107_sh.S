#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64U
RVTEST_CODE_BEGIN

    TEST_ST_OP( 2, lh, sh, 0x00000000000000aa, 0, tdat );
    TEST_ST_OP( 3, lh, sh, 0xffffffffffffaa00, 2, tdat );
    TEST_ST_OP( 4, lw, sh, 0xffffffffbeef0aa0, 4, tdat );
    TEST_ST_OP( 5, lh, sh, 0xffffffffffffa00a, 6, tdat );

    TEST_ST_OP( 6, lh, sh, 0x00000000000000aa, -6, tdat8 );
    TEST_ST_OP( 7, lh, sh, 0xffffffffffffaa00, -4, tdat8 );
    TEST_ST_OP( 8, lh, sh, 0x0000000000000aa0, -2, tdat8 );
    TEST_ST_OP( 9, lh, sh, 0xffffffffffffa00a, 0,  tdat8 );

    TEST_CASE( 10, t0, 0x5678, \
    la  ra, tdat9; \
    li  sp, 0x12345678; \
    addi tp, ra, -32; \
    sh sp, 32(tp); \
    lh t0, 0(ra); \
    )

    TEST_CASE( 11, t0, 0x3098, \
    la  ra, tdat9; \
    li  sp, 0x00003098; \
    addi ra, ra, -5; \
    sh sp, 7(ra); \
    la  tp, tdat10; \
    lh t0, 0(tp); \
    )

    TEST_ST_SRC12_BYPASS( 12, 0, 0, lh, sh, 0xffffffffffffccdd, 0,  tdat );
    TEST_ST_SRC12_BYPASS( 13, 0, 1, lh, sh, 0xffffffffffffbccd, 2,  tdat );
    TEST_ST_SRC12_BYPASS( 14, 0, 2, lh, sh, 0xffffffffffffbbcc, 4,  tdat );
    TEST_ST_SRC12_BYPASS( 15, 1, 0, lh, sh, 0xffffffffffffabbc, 6, tdat );
    TEST_ST_SRC12_BYPASS( 16, 1, 1, lh, sh, 0xffffffffffffaabb, 8, tdat );
    TEST_ST_SRC12_BYPASS( 17, 2, 0, lh, sh, 0xffffffffffffdaab, 10, tdat );

    TEST_ST_SRC21_BYPASS( 18, 0, 0, lh, sh, 0x2233, 0,  tdat );
    TEST_ST_SRC21_BYPASS( 19, 0, 1, lh, sh, 0x1223, 2,  tdat );
    TEST_ST_SRC21_BYPASS( 20, 0, 2, lh, sh, 0x1122, 4,  tdat );
    TEST_ST_SRC21_BYPASS( 21, 1, 0, lh, sh, 0x0112, 6, tdat );
    TEST_ST_SRC21_BYPASS( 22, 1, 1, lh, sh, 0x0011, 8, tdat );
    TEST_ST_SRC21_BYPASS( 23, 2, 0, lh, sh, 0x3001, 10, tdat );

    li a0, 0xbeef
    la a1, tdat
    sh a0, 6(a1)

    TEST_PASSFAIL

RVTEST_CODE_END

    .data
RVTEST_DATA_BEGIN

    TEST_DATA

tdat:
tdat1:  .half 0xbeef
tdat2:  .half 0xbeef
tdat3:  .half 0xbeef
tdat4:  .half 0xbeef
tdat5:  .half 0xbeef
tdat6:  .half 0xbeef
tdat7:  .half 0xbeef
tdat8:  .half 0xbeef
tdat9:  .half 0xbeef
tdat10: .half 0xbeef

RVTEST_DATA_END
