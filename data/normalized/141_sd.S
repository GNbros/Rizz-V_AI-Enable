#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64U
RVTEST_CODE_BEGIN

    TEST_ST_OP( 2, ld, sd, 0x00aa00aa00aa00aa, 0,  tdat );
    TEST_ST_OP( 3, ld, sd, 0xaa00aa00aa00aa00, 8,  tdat );
    TEST_ST_OP( 4, ld, sd, 0x0aa00aa00aa00aa0, 16,  tdat );
    TEST_ST_OP( 5, ld, sd, 0xa00aa00aa00aa00a, 24, tdat );

    TEST_ST_OP( 6, ld, sd, 0x00aa00aa00aa00aa, -24, tdat8 );
    TEST_ST_OP( 7, ld, sd, 0xaa00aa00aa00aa00, -16, tdat8 );
    TEST_ST_OP( 8, ld, sd, 0x0aa00aa00aa00aa0, -8,  tdat8 );
    TEST_ST_OP( 9, ld, sd, 0xa00aa00aa00aa00a, 0,   tdat8 );

    TEST_CASE( 10, t0, 0x1234567812345678, \
    la  ra, tdat9; \
    li  sp, 0x1234567812345678; \
    addi tp, ra, -32; \
    sd sp, 32(tp); \
    ld t0, 0(ra); \
    )

    TEST_CASE( 11, t0, 0x5821309858213098, \
    la  ra, tdat9; \
    li  sp, 0x5821309858213098; \
    addi ra, ra, -3; \
    sd sp, 11(ra); \
    la  tp, tdat10; \
    ld t0, 0(tp); \
    )

    TEST_ST_SRC12_BYPASS( 12, 0, 0, ld, sd, 0xabbccdd, 0,  tdat );
    TEST_ST_SRC12_BYPASS( 13, 0, 1, ld, sd, 0xaabbccd, 8,  tdat );
    TEST_ST_SRC12_BYPASS( 14, 0, 2, ld, sd, 0xdaabbcc, 16, tdat );
    TEST_ST_SRC12_BYPASS( 15, 1, 0, ld, sd, 0xddaabbc, 24, tdat );
    TEST_ST_SRC12_BYPASS( 16, 1, 1, ld, sd, 0xcddaabb, 32, tdat );
    TEST_ST_SRC12_BYPASS( 17, 2, 0, ld, sd, 0xccddaab, 40, tdat );

    TEST_ST_SRC21_BYPASS( 18, 0, 0, ld, sd, 0x00112233, 0,  tdat );
    TEST_ST_SRC21_BYPASS( 19, 0, 1, ld, sd, 0x30011223, 8,  tdat );
    TEST_ST_SRC21_BYPASS( 20, 0, 2, ld, sd, 0x33001122, 16, tdat );
    TEST_ST_SRC21_BYPASS( 21, 1, 0, ld, sd, 0x23300112, 24, tdat );
    TEST_ST_SRC21_BYPASS( 22, 1, 1, ld, sd, 0x22330011, 32, tdat );
    TEST_ST_SRC21_BYPASS( 23, 2, 0, ld, sd, 0x12233001, 40, tdat );

    TEST_PASSFAIL

RVTEST_CODE_END

    .data
RVTEST_DATA_BEGIN

    TEST_DATA

tdat:
tdat1:  .dword 0xdeadbeefdeadbeef
tdat2:  .dword 0xdeadbeefdeadbeef
tdat3:  .dword 0xdeadbeefdeadbeef
tdat4:  .dword 0xdeadbeefdeadbeef
tdat5:  .dword 0xdeadbeefdeadbeef
tdat6:  .dword 0xdeadbeefdeadbeef
tdat7:  .dword 0xdeadbeefdeadbeef
tdat8:  .dword 0xdeadbeefdeadbeef
tdat9:  .dword 0xdeadbeefdeadbeef
tdat10: .dword 0xdeadbeefdeadbeef

RVTEST_DATA_END
