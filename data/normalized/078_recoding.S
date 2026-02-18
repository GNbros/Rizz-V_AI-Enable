#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64UF
RVTEST_CODE_BEGIN

    flw f0, minf, a0
    flw f1, three, a0
    fmul.s f1, f1, f0
    TEST_CASE( 2, a0, 1, feq.s a0, f0, f1)
    TEST_CASE( 3, a0, 1, fle.s a0, f0, f1)
    TEST_CASE( 4, a0, 0, flt.s a0, f0, f1)

    fcvt.s.w f0, zero
    li a0, 1
    fcvt.s.w f1, a0
    fmul.s f1, f1, f0
    TEST_CASE(5, a0, 1, feq.s a0, f0, f1)
    TEST_CASE(6, a0, 1, fle.s a0, f0, f1)
    TEST_CASE(7, a0, 0, flt.s a0, f0, f1)

    TEST_PASSFAIL

RVTEST_CODE_END

    .data
RVTEST_DATA_BEGIN

minf: .float -Inf
three: .float 3.0

RVTEST_DATA_END
