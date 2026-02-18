#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64U
RVTEST_CODE_BEGIN

    TEST_LD_OP( 2, lwu, 0x0000000000ff00ff, 0,  tdat );
    TEST_LD_OP( 3, lwu, 0x00000000ff00ff00, 4,  tdat );
    TEST_LD_OP( 4, lwu, 0x000000000ff00ff0, 8,  tdat );
    TEST_LD_OP( 5, lwu, 0x00000000f00ff00f, 12, tdat );

    TEST_LD_OP( 6, lwu, 0x0000000000ff00ff, -12, tdat4 );
    TEST_LD_OP( 7, lwu, 0x00000000ff00ff00, -8,  tdat4 );
    TEST_LD_OP( 8, lwu, 0x000000000ff00ff0, -4,  tdat4 );
    TEST_LD_OP( 9, lwu, 0x00000000f00ff00f, 0,   tdat4 );

    TEST_CASE( 10, t0, 0x0000000000ff00ff, \
    la  ra, tdat; \
    addi ra, ra, -32; \
    lwu t0, 32(ra); \
    )

    TEST_CASE( 11, t0, 0x00000000ff00ff00, \
    la  ra, tdat; \
    addi ra, ra, -3; \
    lwu t0, 7(ra); \
    )

    TEST_LD_DEST_BYPASS( 12, 0, lwu, 0x000000000ff00ff0, 4, tdat2 );
    TEST_LD_DEST_BYPASS( 13, 1, lwu, 0x00000000f00ff00f, 4, tdat3 );
    TEST_LD_DEST_BYPASS( 14, 2, lwu, 0x00000000ff00ff00, 4, tdat1 );

    TEST_LD_SRC1_BYPASS( 15, 0, lwu, 0x000000000ff00ff0, 4, tdat2 );
    TEST_LD_SRC1_BYPASS( 16, 1, lwu, 0x00000000f00ff00f, 4, tdat3 );
    TEST_LD_SRC1_BYPASS( 17, 2, lwu, 0x00000000ff00ff00, 4, tdat1 );

    TEST_CASE( 18, sp, 2, \
    la  t0, tdat; \
    lwu sp, 0(t0); \
    li  sp, 2; \
    )

    TEST_CASE( 19, sp, 2, \
    la  t0, tdat; \
    lwu sp, 0(t0); \
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
