#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64U
RVTEST_CODE_BEGIN

    TEST_LD_OP( 2, ld, 0x00ff00ff00ff00ff, 0,  tdat );
    TEST_LD_OP( 3, ld, 0xff00ff00ff00ff00, 8,  tdat );
    TEST_LD_OP( 4, ld, 0x0ff00ff00ff00ff0, 16, tdat );
    TEST_LD_OP( 5, ld, 0xf00ff00ff00ff00f, 24, tdat );

    TEST_LD_OP( 6, ld, 0x00ff00ff00ff00ff, -24, tdat4 );
    TEST_LD_OP( 7, ld, 0xff00ff00ff00ff00, -16, tdat4 );
    TEST_LD_OP( 8, ld, 0x0ff00ff00ff00ff0, -8,  tdat4 );
    TEST_LD_OP( 9, ld, 0xf00ff00ff00ff00f, 0,   tdat4 );

    TEST_CASE( 10, t0, 0x00ff00ff00ff00ff, \
    la  ra, tdat; \
    addi ra, ra, -32; \
    ld t0, 32(ra); \
    )

    TEST_CASE( 11, t0, 0xff00ff00ff00ff00, \
    la  ra, tdat; \
    addi ra, ra, -3; \
    ld t0, 11(ra); \
    )

    TEST_LD_DEST_BYPASS( 12, 0, ld, 0x0ff00ff00ff00ff0, 8, tdat2 );
    TEST_LD_DEST_BYPASS( 13, 1, ld, 0xf00ff00ff00ff00f, 8, tdat3 );
    TEST_LD_DEST_BYPASS( 14, 2, ld, 0xff00ff00ff00ff00, 8, tdat1 );

    TEST_LD_SRC1_BYPASS( 15, 0, ld, 0x0ff00ff00ff00ff0, 8, tdat2 );
    TEST_LD_SRC1_BYPASS( 16, 1, ld, 0xf00ff00ff00ff00f, 8, tdat3 );
    TEST_LD_SRC1_BYPASS( 17, 2, ld, 0xff00ff00ff00ff00, 8, tdat1 );

    TEST_CASE( 18, sp, 2, \
    la  t0, tdat; \
    ld  sp, 0(t0); \
    li  sp, 2; \
    )

    TEST_CASE( 19, sp, 2, \
    la  t0, tdat; \
    ld  sp, 0(t0); \
    nop; \
    li  sp, 2; \
    )

    TEST_PASSFAIL

RVTEST_CODE_END

    .data
RVTEST_DATA_BEGIN

    TEST_DATA

tdat:
tdat1:  .dword 0x00ff00ff00ff00ff
tdat2:  .dword 0xff00ff00ff00ff00
tdat3:  .dword 0x0ff00ff00ff00ff0
tdat4:  .dword 0xf00ff00ff00ff00f

RVTEST_DATA_END
