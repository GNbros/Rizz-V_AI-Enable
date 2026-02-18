#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64U
RVTEST_CODE_BEGIN

    TEST_LD_OP( 2, lw, 0x0000000000ff00ff, 0,  tdat );
    TEST_LD_OP( 3, lw, 0xffffffffff00ff00, 4,  tdat );
    TEST_LD_OP( 4, lw, 0x000000000ff00ff0, 8,  tdat );
    TEST_LD_OP( 5, lw, 0xfffffffff00ff00f, 12, tdat );

    TEST_LD_OP( 6, lw, 0x0000000000ff00ff, -12, tdat4 );
    TEST_LD_OP( 7, lw, 0xffffffffff00ff00, -8,  tdat4 );
    TEST_LD_OP( 8, lw, 0x000000000ff00ff0, -4,  tdat4 );
    TEST_LD_OP( 9, lw, 0xfffffffff00ff00f, 0,   tdat4 );

    TEST_CASE( 10, t0, 0x0000000000ff00ff, \
    la  ra, tdat; \
    addi ra, ra, -32; \
    lw t0, 32(ra); \
    )

    TEST_CASE( 11, t0, 0xffffffffff00ff00, \
    la  ra, tdat; \
    addi ra, ra, -3; \
    lw t0, 7(ra); \
    )

    TEST_LD_DEST_BYPASS( 12, 0, lw, 0x000000000ff00ff0, 4, tdat2 );
    TEST_LD_DEST_BYPASS( 13, 1, lw, 0xfffffffff00ff00f, 4, tdat3 );
    TEST_LD_DEST_BYPASS( 14, 2, lw, 0xffffffffff00ff00, 4, tdat1 );

    TEST_LD_SRC1_BYPASS( 15, 0, lw, 0x000000000ff00ff0, 4, tdat2 );
    TEST_LD_SRC1_BYPASS( 16, 1, lw, 0xfffffffff00ff00f, 4, tdat3 );
    TEST_LD_SRC1_BYPASS( 17, 2, lw, 0xffffffffff00ff00, 4, tdat1 );

    TEST_CASE( 18, sp, 2, \
    la  t0, tdat; \
    lw  sp, 0(t0); \
    li  sp, 2; \
    )

    TEST_CASE( 19, sp, 2, \
    la  t0, tdat; \
    lw  sp, 0(t0); \
    nop; \
    li  sp, 2; \
    )

    TEST_PASSFAIL

RVTEST_CODE_END

    .data
RVTEST_DATA_BEGIN

    TEST_DATA

tdat:
tdat1:  .word 0x00ff00ff
tdat2:  .word 0xff00ff00
tdat3:  .word 0x0ff00ff0
tdat4:  .word 0xf00ff00f

RVTEST_DATA_END
