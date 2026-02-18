#*****************************************************************************
# clmul.S
#-----------------------------------------------------------------------------
#
# Test clmul instruction.
#

#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64U
RVTEST_CODE_BEGIN

  #-------------------------------------------------------------
  # Arithmetic tests
  #-------------------------------------------------------------

  TEST_RR_OP(32,  clmul, 0x0000000000005a00, 0x0000000000007e00, 0x6db6db6db6db6db7 );
  TEST_RR_OP(33,  clmul, 0x0000000000005b40, 0x0000000000007fc0, 0x6db6db6db6db6db7 );

  TEST_RR_OP( 2,  clmul, 0x00000000, 0x00000000, 0x00000000 );
  TEST_RR_OP( 3,  clmul, 0x00000001, 0x00000001, 0x00000001 );
  TEST_RR_OP( 4,  clmul, 0x00000009, 0x00000003, 0x00000007 );

  TEST_RR_OP( 5,  clmul, 0x0000000000000000, 0x0000000000000000, 0xffffffffffff8000 );
  TEST_RR_OP( 6,  clmul, 0x0000000000000000, 0xffffffff80000000, 0x00000000 );
  TEST_RR_OP( 7,  clmul, 0x5555400000000000, 0xffffffff80000000, 0xffffffffffff8000 );

  TEST_RR_OP(30,  clmul, 0xfffffffffffc324f, 0xaaaaaaaaaaaaaaab, 0x000000000002fe7d );
  TEST_RR_OP(31,  clmul, 0xfffffffffffc324f, 0x000000000002fe7d, 0xaaaaaaaaaaaaaaab );

  #-------------------------------------------------------------
  # Source/Destination tests
  #-------------------------------------------------------------

  TEST_RR_SRC1_EQ_DEST( 8, clmul, 0x7f, 13, 11 );
  TEST_RR_SRC2_EQ_DEST( 9, clmul, 0x62, 14, 11 );
  TEST_RR_SRC12_EQ_DEST( 10, clmul, 0x51, 13 );

  #-------------------------------------------------------------
  # Bypassing tests
  #-------------------------------------------------------------

  TEST_RR_DEST_BYPASS( 11, 0, clmul, 0x7f, 13, 11 );
  TEST_RR_DEST_BYPASS( 12, 1, clmul, 0x62, 14, 11 );
  TEST_RR_DEST_BYPASS( 13, 2, clmul, 0x69, 15, 11 );

  TEST_RR_SRC12_BYPASS( 14, 0, 0, clmul, 0x7f, 13, 11 );
  TEST_RR_SRC12_BYPASS( 15, 0, 1, clmul, 0x62, 14, 11 );
  TEST_RR_SRC12_BYPASS( 16, 0, 2, clmul, 0x69, 15, 11 );
  TEST_RR_SRC12_BYPASS( 17, 1, 0, clmul, 0x7f, 13, 11 );
  TEST_RR_SRC12_BYPASS( 18, 1, 1, clmul, 0x62, 14, 11 );
  TEST_RR_SRC12_BYPASS( 19, 2, 0, clmul, 0x69, 15, 11 );

  TEST_RR_SRC21_BYPASS( 20, 0, 0, clmul, 0x7f, 13, 11 );
  TEST_RR_SRC21_BYPASS( 21, 0, 1, clmul, 0x62, 14, 11 );
  TEST_RR_SRC21_BYPASS( 22, 0, 2, clmul, 0x69, 15, 11 );
  TEST_RR_SRC21_BYPASS( 23, 1, 0, clmul, 0x7f, 13, 11 );
  TEST_RR_SRC21_BYPASS( 24, 1, 1, clmul, 0x62, 14, 11 );
  TEST_RR_SRC21_BYPASS( 25, 2, 0, clmul, 0x69, 15, 11 );

  TEST_RR_ZEROSRC1( 26, clmul, 0, 31 );
  TEST_RR_ZEROSRC2( 27, clmul, 0, 32 );
  TEST_RR_ZEROSRC12( 28, clmul, 0 );
  TEST_RR_ZERODEST( 29, clmul, 33, 34 );

  TEST_PASSFAIL

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN

  TEST_DATA

RVTEST_DATA_END
# See LICENSE for license details.

#include "riscv_test.h"
#undef RVTEST_RV64S
#define RVTEST_RV64S RVTEST_RV32S

#include "../rv64si/csr.S"
# See LICENSE for license details.

#include "riscv_test.h"
#undef RVTEST_RV64S
#define RVTEST_RV64S RVTEST_RV32S

#include "../rv64si/scall.S"
# See LICENSE for license details.

#include "riscv_test.h"
#undef RVTEST_RV64S
#define RVTEST_RV64S RVTEST_RV32S

#include "../rv64si/sbreak.S"
# See LICENSE for license details.

#include "riscv_test.h"
#undef RVTEST_RV64S
#define RVTEST_RV64S RVTEST_RV32S

#include "../rv64si/ma_fetch.S"
# See LICENSE for license details.

#include "riscv_test.h"
#undef RVTEST_RV64M
#define RVTEST_RV64M RVTEST_RV32M

#undef SATP_MODE_SV39
#define SATP_MODE_SV39 SATP_MODE_SV32

#include "../rv64si/dirty.S"
# See LICENSE for license details.

#include "riscv_test.h"
#undef RVTEST_RV64S
#define RVTEST_RV64S RVTEST_RV32S

#include "../rv64si/wfi.S"
    .text
    .balign 4
    .global vec_sgemm_nn
# RV64IDV system
#
# void
# vec_sgemm_nn(size_t n,
#          size_t m,
#          size_t k,
#          const float*a,   // m * k matrix
#          size_t lda,
#          const float*b,   // k * n matrix
#          size_t ldb,
#          float*c,         // m * n matrix
#          size_t ldc)
#
#  c += a*b (alpha=1, no transpose on input matrices)
#  matrices stored in C row-major order

#define n a0
#define m a1
#define k a2
#define ap a3
#define astride a4
#define bp a5
#define bstride a6
#define cp a7
#define cstride t0
#define kt t1
#define nt t2
#define bnp t3
#define cnp t4
#define akp t5
#define bkp s0
#define nvl s1
#define ccp s2
#define amp s3

# Use args as additional temporaries
#define ft12 fa0
#define ft13 fa1
#define ft14 fa2
#define ft15 fa3

#define FRAMESIZE 32

# This version holds a 16*VLMAX block of C matrix in vector registers
# in inner loop, but otherwise does not cache or TLB tiling.

vec_sgemm_nn:
    ld cstride, 0(sp)   # Get arg from stack frame	
    addi sp, sp, -FRAMESIZE
    sd s0, 0(sp)
    sd s1, 8(sp)
    sd s2, 16(sp)

    # Check for zero size matrices        
    beqz n, exit
    beqz m, exit
    beqz k, exit

    # Convert elements strides to byte strides.
    slli astride, astride, 2
    slli bstride, bstride, 2
    slli cstride, cstride, 2

    slti t6, m, 16
    bnez t6, end_rows

c_row_loop: # Loop across rows of C blocks

    mv nt, n  # Initialize n counter for next row of C blocks

    mv bnp, bp # Initialize B n-loop pointer to start
    mv cnp, cp # Initialize C n-loop pointer

c_col_loop: # Loop across one row of C blocks
    vsetvli nvl, nt, e32, ta, ma  # 32-bit vectors, LMUL=1

    mv akp, ap   # reset pointer into A to beginning
    mv bkp, bnp # step to next column in B matrix

    # Initalize current C submatrix block from memory.
    vle32.v  v0, (cnp); add ccp, cnp, cstride;
    vle32.v  v1, (ccp); add ccp, ccp, cstride;
    vle32.v  v2, (ccp); add ccp, ccp, cstride;
    vle32.v  v3, (ccp); add ccp, ccp, cstride;
    vle32.v  v4, (ccp); add ccp, ccp, cstride;
    vle32.v  v5, (ccp); add ccp, ccp, cstride;
    vle32.v  v6, (ccp); add ccp, ccp, cstride;
    vle32.v  v7, (ccp); add ccp, ccp, cstride;
    vle32.v  v8, (ccp); add ccp, ccp, cstride;
    vle32.v  v9, (ccp); add ccp, ccp, cstride;
    vle32.v v10, (ccp); add ccp, ccp, cstride;
    vle32.v v11, (ccp); add ccp, ccp, cstride;
    vle32.v v12, (ccp); add ccp, ccp, cstride;
    vle32.v v13, (ccp); add ccp, ccp, cstride;
    vle32.v v14, (ccp); add ccp, ccp, cstride;
    vle32.v v15, (ccp)


    mv kt, k # Initialize inner loop counter

    # Inner loop scheduled assuming 4-clock occupancy of vfmacc instruction and single-issue pipeline
    # Software pipeline loads
    flw ft0, (akp); add amp, akp, astride;
    flw ft1, (amp); add amp, amp, astride;
    flw ft2, (amp); add amp, amp, astride;
    flw ft3, (amp); add amp, amp, astride;
    # Get vector from B matrix
    vle32.v v16, (bkp)

    # Loop on inner dimension for current C block
 k_loop:
    vfmacc.vf v0, ft0, v16
    add bkp, bkp, bstride
    flw ft4, (amp)
    add amp, amp, astride
    vfmacc.vf v1, ft1, v16
    addi kt, kt, -1    # Decrement k counter
    flw ft5, (amp)
    add amp, amp, astride
    vfmacc.vf v2, ft2, v16
    flw ft6, (amp)
    add amp, amp, astride
    flw ft7, (amp)
    vfmacc.vf v3, ft3, v16
    add amp, amp, astride
    flw ft8, (amp)
    add amp, amp, astride
    vfmacc.vf v4, ft4, v16
    flw ft9, (amp)
    add amp, amp, astride
    vfmacc.vf v5, ft5, v16
    flw ft10, (amp)
    add amp, amp, astride
    vfmacc.vf v6, ft6, v16
    flw ft11, (amp)
    add amp, amp, astride
    vfmacc.vf v7, ft7, v16
    flw ft12, (amp)
    add amp, amp, astride
    vfmacc.vf v8, ft8, v16
    flw ft13, (amp)
    add amp, amp, astride
    vfmacc.vf v9, ft9, v16
    flw ft14, (amp)
    add amp, amp, astride
    vfmacc.vf v10, ft10, v16
    flw ft15, (amp)
    add amp, amp, astride
    addi akp, akp, 4            # Move to next column of a
    vfmacc.vf v11, ft11, v16
    beqz kt, 1f                 # Don't load past end of matrix
    flw ft0, (akp)
    add amp, akp, astride
1:  vfmacc.vf v12, ft12, v16
    beqz kt, 1f
    flw ft1, (amp)
    add amp, amp, astride
1:  vfmacc.vf v13, ft13, v16
    beqz kt, 1f
    flw ft2, (amp)
    add amp, amp, astride
1:  vfmacc.vf v14, ft14, v16
    beqz kt, 1f                 # Exit out of loop
    flw ft3, (amp)
    add amp, amp, astride
    vfmacc.vf v15, ft15, v16
    vle32.v v16, (bkp)            # Get next vector from B matrix, overlap loads with jump stalls
    j k_loop

1:  vfmacc.vf v15, ft15, v16
    
    # Save C matrix block back to memory
    vse32.v  v0, (cnp); add ccp, cnp, cstride;
    vse32.v  v1, (ccp); add ccp, ccp, cstride;
    vse32.v  v2, (ccp); add ccp, ccp, cstride;
    vse32.v  v3, (ccp); add ccp, ccp, cstride;
    vse32.v  v4, (ccp); add ccp, ccp, cstride;
    vse32.v  v5, (ccp); add ccp, ccp, cstride;
    vse32.v  v6, (ccp); add ccp, ccp, cstride;
    vse32.v  v7, (ccp); add ccp, ccp, cstride;
    vse32.v  v8, (ccp); add ccp, ccp, cstride;
    vse32.v  v9, (ccp); add ccp, ccp, cstride;
    vse32.v v10, (ccp); add ccp, ccp, cstride;
    vse32.v v11, (ccp); add ccp, ccp, cstride;
    vse32.v v12, (ccp); add ccp, ccp, cstride;
    vse32.v v13, (ccp); add ccp, ccp, cstride;
    vse32.v v14, (ccp); add ccp, ccp, cstride;
    vse32.v v15, (ccp)

    # Following tail instructions should be scheduled earlier in free slots during C block save.
    # Leaving here for clarity.

    # Bump pointers for loop across blocks in one row
    slli t6, nvl, 2
    add cnp, cnp, t6                         # Move C block pointer over
    add bnp, bnp, t6                         # Move B block pointer over
    sub nt, nt, nvl                          # Decrement element count in n dimension
    bnez nt, c_col_loop                      # Any more to do?

    # Move to next set of rows
    addi m, m, -16  # Did 16 rows above
    slli t6, astride, 4  # Multiply astride by 16
    add ap, ap, t6         # Move A matrix pointer down 16 rows
    slli t6, cstride, 4  # Multiply cstride by 16
    add cp, cp, t6         # Move C matrix pointer down 16 rows
    
    slti t6, m, 16
    beqz t6, c_row_loop

    # Handle end of matrix with fewer than 16 rows.
    # Can use smaller versions of above decreasing in powers-of-2 depending on code-size concerns.
end_rows:
    # Not done.

exit:
    ld s0, 0(sp)
    ld s1, 8(sp)
    ld s2, 16(sp)
    addi sp, sp, FRAMESIZE
    ret
# See LICENSE for license details.

#include "encoding.h"

#if __riscv_xlen == 64
# define LREG ld
# define SREG sd
# define REGBYTES 8
#else
# define LREG lw
# define SREG sw
# define REGBYTES 4
#endif

  .section ".text.init"
  .globl _start
_start:
  li  x1, 0
  li  x2, 0
  li  x3, 0
  li  x4, 0
  li  x5, 0
  li  x6, 0
  li  x7, 0
  li  x8, 0
  li  x9, 0
  li  x10,0
  li  x11,0
  li  x12,0
  li  x13,0
  li  x14,0
  li  x15,0
  li  x16,0
  li  x17,0
  li  x18,0
  li  x19,0
  li  x20,0
  li  x21,0
  li  x22,0
  li  x23,0
  li  x24,0
  li  x25,0
  li  x26,0
  li  x27,0
  li  x28,0
  li  x29,0
  li  x30,0
  li  x31,0

  # enable FPU, vector, and accelerator if present
  li t0, MSTATUS_FS | MSTATUS_XS | MSTATUS_VS
  csrs mstatus, t0

  # make sure XLEN agrees with compilation choice
  li t0, 1
  slli t0, t0, 31
#if __riscv_xlen == 64
  bgez t0, 1f
#else
  bltz t0, 1f
#endif
2:
  li a0, 1
  sw a0, tohost, t0
  j 2b
1:

#ifdef __riscv_flen
  # initialize FPU if we have one
  la t0, 1f
  csrw mtvec, t0

  fssr    x0
  fmv.s.x f0, x0
  fmv.s.x f1, x0
  fmv.s.x f2, x0
  fmv.s.x f3, x0
  fmv.s.x f4, x0
  fmv.s.x f5, x0
  fmv.s.x f6, x0
  fmv.s.x f7, x0
  fmv.s.x f8, x0
  fmv.s.x f9, x0
  fmv.s.x f10,x0
  fmv.s.x f11,x0
  fmv.s.x f12,x0
  fmv.s.x f13,x0
  fmv.s.x f14,x0
  fmv.s.x f15,x0
  fmv.s.x f16,x0
  fmv.s.x f17,x0
  fmv.s.x f18,x0
  fmv.s.x f19,x0
  fmv.s.x f20,x0
  fmv.s.x f21,x0
  fmv.s.x f22,x0
  fmv.s.x f23,x0
  fmv.s.x f24,x0
  fmv.s.x f25,x0
  fmv.s.x f26,x0
  fmv.s.x f27,x0
  fmv.s.x f28,x0
  fmv.s.x f29,x0
  fmv.s.x f30,x0
  fmv.s.x f31,x0
1:
#endif

  # initialize trap vector
  la t0, trap_entry
  csrw mtvec, t0

  # initialize global pointer
.option push
.option norelax
  la gp, __global_pointer$
.option pop

  la  tp, _end + 63
  and tp, tp, -64

  # get core id
  csrr a0, mhartid
  # for now, assume only 1 core
  li a1, 1
1:bgeu a0, a1, 1b

  # give each core 128KB of stack + TLS
#define STKSHIFT 17
  add sp, a0, 1
  sll sp, sp, STKSHIFT
  add sp, sp, tp
  sll a2, a0, STKSHIFT
  add tp, tp, a2

  j _init

  .align 2
trap_entry:
  addi sp, sp, -272

  SREG x1, 1*REGBYTES(sp)
  SREG x2, 2*REGBYTES(sp)
  SREG x3, 3*REGBYTES(sp)
  SREG x4, 4*REGBYTES(sp)
  SREG x5, 5*REGBYTES(sp)
  SREG x6, 6*REGBYTES(sp)
  SREG x7, 7*REGBYTES(sp)
  SREG x8, 8*REGBYTES(sp)
  SREG x9, 9*REGBYTES(sp)
  SREG x10, 10*REGBYTES(sp)
  SREG x11, 11*REGBYTES(sp)
  SREG x12, 12*REGBYTES(sp)
  SREG x13, 13*REGBYTES(sp)
  SREG x14, 14*REGBYTES(sp)
  SREG x15, 15*REGBYTES(sp)
  SREG x16, 16*REGBYTES(sp)
  SREG x17, 17*REGBYTES(sp)
  SREG x18, 18*REGBYTES(sp)
  SREG x19, 19*REGBYTES(sp)
  SREG x20, 20*REGBYTES(sp)
  SREG x21, 21*REGBYTES(sp)
  SREG x22, 22*REGBYTES(sp)
  SREG x23, 23*REGBYTES(sp)
  SREG x24, 24*REGBYTES(sp)
  SREG x25, 25*REGBYTES(sp)
  SREG x26, 26*REGBYTES(sp)
  SREG x27, 27*REGBYTES(sp)
  SREG x28, 28*REGBYTES(sp)
  SREG x29, 29*REGBYTES(sp)
  SREG x30, 30*REGBYTES(sp)
  SREG x31, 31*REGBYTES(sp)

  csrr a0, mcause
  csrr a1, mepc
  mv a2, sp
  jal handle_trap
  csrw mepc, a0

  # Remain in M-mode after eret
  li t0, MSTATUS_MPP
  csrs mstatus, t0

  LREG x1, 1*REGBYTES(sp)
  LREG x2, 2*REGBYTES(sp)
  LREG x3, 3*REGBYTES(sp)
  LREG x4, 4*REGBYTES(sp)
  LREG x5, 5*REGBYTES(sp)
  LREG x6, 6*REGBYTES(sp)
  LREG x7, 7*REGBYTES(sp)
  LREG x8, 8*REGBYTES(sp)
  LREG x9, 9*REGBYTES(sp)
  LREG x10, 10*REGBYTES(sp)
  LREG x11, 11*REGBYTES(sp)
  LREG x12, 12*REGBYTES(sp)
  LREG x13, 13*REGBYTES(sp)
  LREG x14, 14*REGBYTES(sp)
  LREG x15, 15*REGBYTES(sp)
  LREG x16, 16*REGBYTES(sp)
  LREG x17, 17*REGBYTES(sp)
  LREG x18, 18*REGBYTES(sp)
  LREG x19, 19*REGBYTES(sp)
  LREG x20, 20*REGBYTES(sp)
  LREG x21, 21*REGBYTES(sp)
  LREG x22, 22*REGBYTES(sp)
  LREG x23, 23*REGBYTES(sp)
  LREG x24, 24*REGBYTES(sp)
  LREG x25, 25*REGBYTES(sp)
  LREG x26, 26*REGBYTES(sp)
  LREG x27, 27*REGBYTES(sp)
  LREG x28, 28*REGBYTES(sp)
  LREG x29, 29*REGBYTES(sp)
  LREG x30, 30*REGBYTES(sp)
  LREG x31, 31*REGBYTES(sp)

  addi sp, sp, 272
  mret

.section ".tohost","aw",@progbits
.align 6
.globl tohost
tohost: .dword 0
.align 6
.globl fromhost
fromhost: .dword 0
    .text
    .balign 4
    .global vec_daxpy
# void
# vec_daxpy(size_t n, const double a, const double *x, const double *y, double* z)
# {
#   size_t i;
#   for (i=0; i<n; i++)
#     z[i] = a * x[i] + y[i];
# }
#
# register arguments:
#     a0      n
#     fa0     a
#     a1      x
#     a2      y
#     a3      z
vec_daxpy:
    vsetvli a4, a0, e64, m8, ta, ma
    vle64.v v0, (a1)
    sub a0, a0, a4
    slli a4, a4, 3
    add a1, a1, a4
    vle64.v v8, (a2)
    vfmacc.vf v8, fa0, v0
    vse64.v v8, (a3)
    add a2, a2, a4
    add a3, a3, a4
    bnez a0, vec_daxpy
    ret
   .text
    .balign 4
    .global vec_memcpy
    # void *vec_memcpy(void* dest, const void* src, size_t n)
    # a0=dest, a1=src, a2=n
    #
  vec_memcpy:
      mv a3, a0 # Copy destination
  loop:
    vsetvli t0, a2, e8, m8, ta, ma   # Vectors of 8b
    vle8.v v0, (a1)               # Load bytes
      add a1, a1, t0              # Bump pointer
      sub a2, a2, t0              # Decrement count
    vse8.v v0, (a3)               # Store bytes
      add a3, a3, t0              # Bump pointer
      bnez a2, loop               # Any more?
      ret                         # Return
    .text
    .balign 4
    .global vec_strcmp
  # int vec_strcmp(const char *src1, const char* src2)
vec_strcmp:
    ##  Using LMUL=2, but same register names work for larger LMULs
    li t1, 0                # Initial pointer bump
loop:
    vsetvli t0, x0, e8, m2, ta, ma  # Max length vectors of bytes
    add a0, a0, t1          # Bump src1 pointer
    vle8ff.v v8, (a0)       # Get src1 bytes
    add a1, a1, t1          # Bump src2 pointer
    vle8ff.v v16, (a1)      # Get src2 bytes

    vmseq.vi v0, v8, 0      # Flag zero bytes in src1
    vmsne.vv v1, v8, v16    # Flag if src1 != src2
    vmor.mm v0, v0, v1      # Combine exit conditions
    
    vfirst.m a2, v0         # ==0 or != ?
    csrr t1, vl             # Get number of bytes fetched
    
    bltz a2, loop           # Loop if all same and no zero byte

    add a0, a0, a2          # Get src1 element address
    lbu a3, (a0)            # Get src1 byte from memory

    add a1, a1, a2          # Get src2 element address
    lbu a4, (a1)            # Get src2 byte from memory

    sub a0, a3, a4          # Return value.

    ret
#if XLEN == 64
# define LREG ld
# define SREG sd
# define REGBYTES 8
#else
# define LREG lw
# define SREG sw
# define REGBYTES 4
#endif

#ifdef CLINT
#define MTIME_ADDR      CLINT + 0xbff8
#endif

        .global main
main:
        li      s0, 0
        li      s1, MTIME_ADDR
loop:
        addi    s0, s0, 1
        LREG    s2, 0(s1)
        j       loop
#include "encoding.h"

        .global main
        .global main_end
        .global main_post_csrr

        // Load constants into all registers so we can test no register are
        // clobbered after attaching.
main:
        csrr    x1, CSR_MHARTID
        slli    x1, x1, 8
main_post_csrr:
        addi    x2, x1, 1
        addi    x3, x2, 1
        addi    x4, x3, 1
        addi    x5, x4, 1
        addi    x6, x5, 1
        addi    x7, x6, 1
        addi    x8, x7, 1
        addi    x9, x8, 1
        addi    x10, x9, 1
        addi    x11, x10, 1
        addi    x12, x11, 1
        addi    x13, x12, 1
        addi    x14, x13, 1
        addi    x15, x14, 1
        addi    x16, x15, 1
        addi    x17, x16, 1
        addi    x18, x17, 1
        addi    x19, x18, 1
        addi    x20, x19, 1
        addi    x21, x20, 1
        addi    x22, x21, 1
        addi    x23, x22, 1
        addi    x24, x23, 1
        addi    x25, x24, 1
        addi    x26, x25, 1
        addi    x27, x26, 1
        addi    x28, x27, 1
        addi    x29, x28, 1
        addi    x30, x29, 1
        addi    x31, x30, 1
main_end:
        j       main_end
#include "encoding.h"

#if XLEN == 64
# define LREG ld
# define SREG sd
# define REGBYTES 8
#else
# define LREG lw
# define SREG sw
# define REGBYTES 4
#endif

        .global         main

        .section        .text
main:

        la      a0, data
        li      t0, 0
just_before_read_loop:
        li      t2, 16
read_loop:
        lw      t1, 0(a0)
        addi    t1, t1, 1
        addi    t0, t0, 1
read_again:
        lw      t1, 0(a0)
        addi    a0, a0, 4
        blt     t0, t2, read_loop

        la      a0, data
just_before_write_loop:
        li      t0, 1
write_loop:
        sw      t0, 0(a0)
        addi    t0, t0, 1
        addi    a0, a0, 4
        blt     t0, t2, write_loop

        j       main_exit

write_store_trigger:
        /* 2<<60 is for RV64. 2<<28 is for RV32. That's safe because on RV64 bits 28 and 29 are 0. */
        li      a0, (2<<60) | (2<<28) | (1<<6) | (1<<1)
        li      a1, 0xdeadbee0
        jal     write_triggers
        la      a0, data
        jal     read_triggers

write_load_trigger:
        li      a0, (2<<60) | (2<<28) | (1<<6) | (1<<0)
        li      a1, 0xfeedac00
        jal     write_triggers
        la      a0, data
        jal     read_triggers

// Clear triggers so the next test can use them.
clear_triggers:
        li      a0, 0
        jal     write_triggers

main_exit:
        li      a0, 10
        j       _exit

write_triggers:
        // a0: value to write to each tdata1
        // a1: value to write to each tdata2
        li      t0, 0
2:
        csrw    CSR_TSELECT, t0
        csrr    t1, CSR_TSELECT
        bne     t0, t1, 1f
        addi    t0, t0, 1
        csrw    CSR_TDATA2, a1
        csrw    CSR_TDATA1, a0
        j       2b
1:      ret

read_triggers:
        // a0: address where data should be written
        li      t0, 0
2:
        csrw    CSR_TSELECT, t0
        csrr    t1, CSR_TSELECT
        bne     t0, t1, 1f
        addi    t0, t0, 1
        csrr    t1, CSR_TDATA1
        SREG    t1, 0(a0)
        csrr    t1, CSR_TDATA2
        SREG    t1, REGBYTES(a0)
        addi    a0, a0, 2*REGBYTES
        j       2b
1:      SREG    zero, 0(a0)
        ret

        .section .data
        .align  3
data:   .word   0x40
        .word   0x41
        .word   0x42
        .word   0x43
        .word   0x44
        .word   0x45
        .word   0x46
        .word   0x47
        .word   0x48
        .word   0x49
        .word   0x4a
        .word   0x4b
        .word   0x4c
        .word   0x4d
        .word   0x4e
        .word   0x4f
// Test stepping over a variety of instructions.

        .global main

main:
        la      t0, trap_entry          // 0, 4
        csrw    mtvec, t0               // 0x8

        li      t0, 5                   // 0xc
        beq     zero, zero, one         // 0x10
        nop                             // 0x14
one:
        beq     zero, t0, one           // 0x18
        // Use t0 instead of ra to force a 32-bit opcode in C mode. Otherwise
        // 32-bit and 64-bit binaries end up with different instructions (I
        // didn't pursue this).
        jal     t0, two                 // 0x1c

three:
        .word   0                       // 0x20
        nop                             // 0x24

two:
        jr      t0                      // 0x28

        .align  2
trap_entry:
        j       trap_entry              // 0x2c
#include "encoding.h"

#if XLEN == 64
# define LREG ld
# define SREG sd
# define REGBYTES 8
#else
# define LREG lw
# define SREG sw
# define REGBYTES 4
#endif

#define get_field(reg, mask) (((reg) & (mask)) / ((mask) & ~((mask) << 1)))
#define set_field(reg, mask, val) (((reg) & ~(mask)) | (((val) * ((mask) & ~((mask) << 1))) & (mask)))

        .global main
        .global main_end
        .global main_post_csrr

        // Load constants into all registers so we can test no register are
        // clobbered after attaching.
main:
        SREG    ra, 0(sp)
        addi    sp, sp, REGBYTES

        // Set VS=1
        csrr    t0, CSR_MSTATUS
        li      t1, set_field(0, MSTATUS_VS, 1)
        or      t0, t0, t1
        csrw    CSR_MSTATUS, t0

        // copy a to b
        la      a0, a
        jal     vector_load_v4
        la      a0, b
        jal     shift_store_v4

        // assert a == b
        la      a0, a
        la      a1, b
        jal     check_equal
test0:
        bne     a0, zero, return_from_main

        // copy b to c
        la      a0, b
        jal     shift_load_v4
        la      a0, c
        jal     vector_store_v4

        // assert b == c
        la      a0, b
        la      a1, c
        jal     check_equal
test1:
        bne     a0, zero, return_from_main

return_from_main:
        addi    sp, sp, -REGBYTES
        LREG    ra, 0(sp)
        ret

vector_load_v4:
        // a0: point to memory to load from
        csrr    s0, vlenb
        vsetvli zero, s0, e8, m1  # Vectors of 8b
        vle8.v v4, 0(a0)          # Load bytes
        ret

vector_store_v4:
        // a0: point to memory to store to
        csrr    s0, vlenb
        vsetvli zero, s0, e8, m1  # Vectors of 8b
        vse8.v v4, 0(a0)          # Load bytes
        ret

shift_load_v4:
        // a0: pointer to memory to load from

        // Configure all elements in the chain
        csrr    s0, vlenb
#if XLEN == 32
        vsetvli zero, s0, e32
#else
        vsetvli zero, s0, e64
#endif

        // Figure out how long the chain is.
        csrr    s0, vlenb
        li      s1, XLEN/8
        divu    s0, s0, s1

1:
        LREG    s2, 0(a0)
        vslide1down.vx  v4, v4, s2
        addi    a0, a0, REGBYTES
        addi    s0, s0, -1
        bne     s0, zero, 1b

        ret

shift_store_v4:
        // a0: pointer to memory to store to

        // Configure all elements in the chain
        csrr    s0, vlenb
#if XLEN == 32
        vsetvli zero, s0, e32
#else
        vsetvli zero, s0, e64
#endif

        // Figure out how long the chain is.
        csrr    s0, vlenb
        li      s1, XLEN/8
        divu    s0, s0, s1

1:
        vmv.x.s s2, v4
        SREG    s2, 0(a0)
        vslide1down.vx  v4, v4, s2
        addi    a0, a0, REGBYTES
        addi    s0, s0, -1
        bne     s0, zero, 1b

        ret

check_equal:
        csrr    s0, vlenb
1:
        lb      s1, 0(a0)
        lb      s2, 0(a1)
        bne     s1, s2, 2f
        addi    a0, a0, 1
        addi    a1, a1, 1
        addi    s0, s0, -1
        bne     s0, zero, 1b
        li      a0, 0   // equal
        ret
2:      // unequal
        li      a0, 1
        ret

        .data
        .align  6
a:      .word   0xaa00, 0xaa01, 0xaa02, 0xaa03, 0xaa04, 0xaa05, 0xaa06, 0xaa07
        .word   0xaa08, 0xaa09, 0xaa0a, 0xaa0b, 0xaa0c, 0xaa0d, 0xaa0e, 0xaa0f
        .word   0xaa10, 0xaa11, 0xaa12, 0xaa13, 0xaa14, 0xaa15, 0xaa16, 0xaa17
        .word   0xaa18, 0xaa19, 0xaa1a, 0xaa1b, 0xaa1c, 0xaa1d, 0xaa1e, 0xaa1f

b:      .word   0xbb00, 0xbb01, 0xbb02, 0xbb03, 0xbb04, 0xbb05, 0xbb06, 0xbb07
        .word   0xbb08, 0xbb09, 0xbb0b, 0xbb0b, 0xbb0c, 0xbb0d, 0xbb0e, 0xbb0f
        .word   0xbb10, 0xbb11, 0xbb13, 0xbb13, 0xbb14, 0xbb15, 0xbb16, 0xbb17
        .word   0xbb18, 0xbb19, 0xbb1b, 0xbb1b, 0xbb1c, 0xbb1d, 0xbb1e, 0xbb1f

c:      .word   0xcc00, 0xcc01, 0xcc02, 0xcc03, 0xcc04, 0xcc05, 0xcc06, 0xcc07
        .word   0xcc08, 0xcc09, 0xcc0c, 0xcc0c, 0xcc0c, 0xcc0d, 0xcc0e, 0xcc0f
        .word   0xcc10, 0xcc11, 0xcc13, 0xcc13, 0xcc14, 0xcc15, 0xcc16, 0xcc17
        .word   0xcc18, 0xcc19, 0xcc1c, 0xcc1c, 0xcc1c, 0xcc1d, 0xcc1e, 0xcc1f#include "encoding.h"

#define STACK_SIZE (90 * XLEN / 8)

#if XLEN == 64
# define LREG ld
# define SREG sd
# define REGBYTES 8
#else
# define LREG lw
# define SREG sw
# define REGBYTES 4
#endif

  .section      .text.entry
  .globl _start
_start:
  j handle_reset

nmi_vector:
  j nmi_vector

trap_vector:
  j trap_entry

handle_reset:
  // If misa doesn't exist (or is following an old spec where it has a
  // different number), skip the next block.
  la t0, 3f
  csrw mtvec, t0
  csrwi mstatus, 0

  // make sure these registers exist by seeing if either S or U bits
  // are set before attempting to zero them out.
  csrr   t1, misa
  addi   t2, x0, 1 	
  slli   t2, t2, 20 // U_EXTENSION
  and    t2, t1, t2
  bne    x0, t2, 1f
  addi   t2, x0, 1
  slli   t2, t2, 18 // S_EXTENSION	
  and    t2, t1, t2
  bne    x0, t2, 1f
  j 2f
1:
  csrwi mideleg, 0
  csrwi medeleg, 0
2:
  csrwi mie, 0
3:
  la t0, trap_entry
  csrw mtvec, t0
  csrwi mstatus, 0

  # initialize global pointer
.option push
.option norelax
  la gp, __global_pointer$
.option pop

  # Initialize stack pointer.
  la    sp, stack_bottom
  # Give each hart STACK_SIZE of stack.
  # Assume hart IDs are contiguous and start at 0.
  li    t1, STACK_SIZE
  csrr  t0, CSR_MHARTID
  # Don't use mul instruction because not all harts support it.
  addi  t0, t0, 1
1:
  add   sp, sp, t1
  addi  t0, t0, -1
  bnez  t0, 1b

  # Catch trap in case trigger module is not implemented
  la t2, 2f
  csrrw t2, mtvec, t2
  # Clear all hardware triggers
  li    t0, ~0
1:
  addi  t0, t0, 1
  csrw  CSR_TSELECT, t0
  csrw  CSR_TDATA1, zero
  csrr  t1, CSR_TSELECT
  beq   t0, t1, 1b
.p2align 2
2:
  # Restore mtvec and mstatus
  csrw mtvec, t2
  csrwi mstatus, 0

#ifdef MULTICORE
  csrr  t0, CSR_MHARTID
  bnez  t0, wait_until_initialized
#endif

  la    t0, __bss_start
  la    t1, __bss_end
1:
  bge   t0, t1, 2f
  sb    zero, 0(t0)
  addi  t0, t0, 1
  j     1b
2:
#ifdef MULTICORE
  la    t0, initialized
  li    t1, 1
  sw    t1, 0(t0)

wait_until_initialized:      # Wait for hart 0 to perform initialization.
  la    t0, initialized
1:
  lw    t1, 0(t0)
  beqz  t1, 1b
#endif

  # perform the rest of initialization in C
  j     _init


.align 2
trap_entry:
  addi sp, sp, -32*REGBYTES

  SREG x1, 1*REGBYTES(sp)
  SREG x2, 2*REGBYTES(sp)
  SREG x3, 3*REGBYTES(sp)
  SREG x4, 4*REGBYTES(sp)
  SREG x5, 5*REGBYTES(sp)
  SREG x6, 6*REGBYTES(sp)
  SREG x7, 7*REGBYTES(sp)
  SREG x8, 8*REGBYTES(sp)
  SREG x9, 9*REGBYTES(sp)
  SREG x10, 10*REGBYTES(sp)
  SREG x11, 11*REGBYTES(sp)
  SREG x12, 12*REGBYTES(sp)
  SREG x13, 13*REGBYTES(sp)
  SREG x14, 14*REGBYTES(sp)
  SREG x15, 15*REGBYTES(sp)
#ifndef RV32E
  SREG x16, 16*REGBYTES(sp)
  SREG x17, 17*REGBYTES(sp)
  SREG x18, 18*REGBYTES(sp)
  SREG x19, 19*REGBYTES(sp)
  SREG x20, 20*REGBYTES(sp)
  SREG x21, 21*REGBYTES(sp)
  SREG x22, 22*REGBYTES(sp)
  SREG x23, 23*REGBYTES(sp)
  SREG x24, 24*REGBYTES(sp)
  SREG x25, 25*REGBYTES(sp)
  SREG x26, 26*REGBYTES(sp)
  SREG x27, 27*REGBYTES(sp)
  SREG x28, 28*REGBYTES(sp)
  SREG x29, 29*REGBYTES(sp)
  SREG x30, 30*REGBYTES(sp)
  SREG x31, 31*REGBYTES(sp)
#endif

  csrr a0, mcause
  csrr a1, mepc
  mv a2, sp
  jal handle_trap
  csrw mepc, a0

  # Remain in M-mode after mret
  li t0, MSTATUS_MPP
  csrs mstatus, t0

  LREG x1, 1*REGBYTES(sp)
  LREG x2, 2*REGBYTES(sp)
  LREG x3, 3*REGBYTES(sp)
  LREG x4, 4*REGBYTES(sp)
  LREG x5, 5*REGBYTES(sp)
  LREG x6, 6*REGBYTES(sp)
  LREG x7, 7*REGBYTES(sp)
  LREG x8, 8*REGBYTES(sp)
  LREG x9, 9*REGBYTES(sp)
  LREG x10, 10*REGBYTES(sp)
  LREG x11, 11*REGBYTES(sp)
  LREG x12, 12*REGBYTES(sp)
  LREG x13, 13*REGBYTES(sp)
  LREG x14, 14*REGBYTES(sp)
  LREG x15, 15*REGBYTES(sp)
#ifndef RV32E
  LREG x16, 16*REGBYTES(sp)
  LREG x17, 17*REGBYTES(sp)
  LREG x18, 18*REGBYTES(sp)
  LREG x19, 19*REGBYTES(sp)
  LREG x20, 20*REGBYTES(sp)
  LREG x21, 21*REGBYTES(sp)
  LREG x22, 22*REGBYTES(sp)
  LREG x23, 23*REGBYTES(sp)
  LREG x24, 24*REGBYTES(sp)
  LREG x25, 25*REGBYTES(sp)
  LREG x26, 26*REGBYTES(sp)
  LREG x27, 27*REGBYTES(sp)
  LREG x28, 28*REGBYTES(sp)
  LREG x29, 29*REGBYTES(sp)
  LREG x30, 30*REGBYTES(sp)
  LREG x31, 31*REGBYTES(sp)
#endif

  addi sp, sp, 32*REGBYTES
  mret

loop_forever:
  j loop_forever

  .align 2
precease:
    // Loop a while so that OpenOCD might have confirmed the resume before the
    // hart becomes unavailable.
    li t1, 100000
1:
    addi t1, t1, -1
    bnez t1, 1b
cease:
  .word 0x30500073  // cease
  j loop_forever

  // Fill the stack with data so we can see if it was overrun.
  .section .data
  .align 4
stack_bottom:
  .fill NHARTS * STACK_SIZE/4, 4, 0x22446688
stack_top:
initialized:
  .word 0
#include "encoding.h"

        .global         main

        .section        .text
main:
        # MISA is only readable from machine mode
        csrr    t0, CSR_MISA
        csrr    t0, CSR_MISA
        csrr    t0, CSR_MISA
        csrr    t0, CSR_MISA
#if XLEN == 64
# define LREG ld
# define SREG sd
# define REGBYTES 8
#else
# define LREG lw
# define SREG sw
# define REGBYTES 4
#endif

#include "encoding.h"

        .global main
main:
        nop
        j       main

write_regs:
        SREG    x2, 0(x1)
        SREG    x3, 8(x1)
        SREG    x4, 16(x1)
        SREG    x5, 24(x1)
        SREG    x6, 32(x1)
        SREG    x7, 40(x1)
        SREG    x8, 48(x1)
        SREG    x9, 56(x1)
        SREG    x10, 64(x1)
        SREG    x11, 72(x1)
        SREG    x12, 80(x1)
        SREG    x13, 88(x1)
        SREG    x14, 96(x1)
        SREG    x15, 104(x1)
#ifndef RV32E
        SREG    x16, 112(x1)
        SREG    x17, 120(x1)
        SREG    x18, 128(x1)
        SREG    x19, 136(x1)
        SREG    x20, 144(x1)
        SREG    x21, 152(x1)
        SREG    x22, 160(x1)
        SREG    x23, 168(x1)
        SREG    x24, 176(x1)
        SREG    x25, 184(x1)
        SREG    x26, 192(x1)
        SREG    x27, 200(x1)
        SREG    x28, 208(x1)
        SREG    x29, 216(x1)
        SREG    x30, 224(x1)
        SREG    x31, 232(x1)
#endif

        csrr    x1, CSR_MSCRATCH

all_done:
        j       all_done

        .section .bss
        .balign  16
data:
        .fill   64, 8, 0
#include "encoding.h"
#define PGSHIFT         12

        .global         main

        .section        .text
main:
        # Set up a page table entry that maps 0x0... to 0x8...
        la      t0, page_table
        srli    t0, t0, PGSHIFT
        csrw    CSR_SATP, t0

        # update mstatus
        csrr    t1, CSR_MSTATUS
#if XLEN == 32
        li      t0, (MSTATUS_MPRV | (SATP_MODE_SV32 << 24))
#else
        li      t0, (MSTATUS_MPRV | (SATP_MODE_SV39 << 24))
#endif
        #li      t0, ((VM_SV39 << 24))
        or      t1, t0, t1
        csrw    CSR_MSTATUS, t1

        la      t0, (loop - 0x80000000)
        csrw    CSR_MEPC, t0

        # Exit supervisor mode, entering user mode at loop.
        mret

loop:
        la      t0, data
        lw      t1, 0(t0)
        j       loop

        .section        .data
data:
        .word   0xbead

        .balign 0x1000
page_table:
#if XLEN == 32
        .word   ((0x80000000 >> 2) | PTE_V | PTE_R | PTE_W | PTE_X | PTE_G | PTE_U)
#else
        .word   ((0x80000000 >> 2) | PTE_V | PTE_R | PTE_W | PTE_X | PTE_G | PTE_U)
        .word   0
#endif
