#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64UF
RVTEST_CODE_BEGIN

  fld f0, minf, a0
  fld f1, three, a0
  fmul.d f1, f1, f0
  TEST_CASE( 2, a0, 1, feq.d a0, f0, f1)
  TEST_CASE( 3, a0, 1, fle.d a0, f0, f1)
  TEST_CASE( 4, a0, 0, flt.d a0, f0, f1)

  fcvt.d.w f0, zero
  li a0, 1
  fcvt.d.w f1, a0
  fmul.d f1, f1, f0
  TEST_CASE(5, a0, 1, feq.d a0, f0, f1)
  TEST_CASE(6, a0, 1, fle.d a0, f0, f1)
  TEST_CASE(7, a0, 0, flt.d a0, f0, f1)

  flw f0, big, a0
  fld f1, tiny, a0
  fcvt.s.d f1, f1
  fmul.s f0, f0, f1
  fmv.x.s a0, f0
  lw a1, small
  TEST_CASE(10, a0, 0, sub a0, a0, a1)

  flw f0, three, a0
  fadd.s f1, f0, f0
  fadd.s f0, f0, f0
  fsd f0, tiny, a0
  fld f0, tiny, a0
  TEST_CASE(20, a0, 1, feq.s a0, f0, f1)

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

minf: .double -Inf
three: .double 3.0
big: .float 1221
small: .float 2.9133121e-37
tiny: .double 2.3860049081905093e-40

RVTEST_DATA_END
