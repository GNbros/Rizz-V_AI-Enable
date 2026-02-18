#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64U
RVTEST_CODE_BEGIN

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

    TEST_RR_SRC1_EQ_DEST( 8, clmul, 0x7f, 13, 11 );
    TEST_RR_SRC2_EQ_DEST( 9, clmul, 0x62, 14, 11 );
    TEST_RR_SRC12_EQ_DEST( 10, clmul, 0x51, 13 );

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

#include "riscv_test.h"
#undef RVTEST_RV64S
#define RVTEST_RV64S RVTEST_RV32S

#include "../rv64si/csr.S"

#include "riscv_test.h"
#undef RVTEST_RV64S
#define RVTEST_RV64S RVTEST_RV32S

#include "../rv64si/scall.S"

#include "riscv_test.h"
#undef RVTEST_RV64S
#define RVTEST_RV64S RVTEST_RV32S

#include "../rv64si/sbreak.S"

#include "riscv_test.h"
#undef RVTEST_RV64S
#define RVTEST_RV64S RVTEST_RV32S

#include "../rv64si/ma_fetch.S"

#include "riscv_test.h"
#undef RVTEST_RV64M
#define RVTEST_RV64M RVTEST_RV32M

#undef SATP_MODE_SV39
#define SATP_MODE_SV39 SATP_MODE_SV32

#include "../rv64si/dirty.S"

#include "riscv_test.h"
#undef RVTEST_RV64S
#define RVTEST_RV64S RVTEST_RV32S

#include "../rv64si/wfi.S"
    .text
    .balign 4
    .global vec_sgemm_nn

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

#define ft12 fa0
#define ft13 fa1
#define ft14 fa2
#define ft15 fa3

#define FRAMESIZE 32

vec_sgemm_nn:
    ld cstride, 0(sp)
    addi sp, sp, -FRAMESIZE
    sd s0, 0(sp)
    sd s1, 8(sp)
    sd s2, 16(sp)

    beqz n, exit
    beqz m, exit
    beqz k, exit

    slli astride, astride, 2
    slli bstride, bstride, 2
    slli cstride, cstride, 2

    slti t6, m, 16
    bnez t6, end_rows

c_row_loop:

    mv nt, n

    mv bnp, bp
    mv cnp, cp

c_col_loop:
    vsetvli nvl, nt, e32, ta, ma

    mv akp, ap
    mv bkp, bnp

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

    mv kt, k

    flw ft0, (akp); add amp, akp, astride;
    flw ft1, (amp); add amp, amp, astride;
    flw ft2, (amp); add amp, amp, astride;
    flw ft3, (amp); add amp, amp, astride;
    vle32.v v16, (bkp)

    k_loop:
    vfmacc.vf v0, ft0, v16
    add bkp, bkp, bstride
    flw ft4, (amp)
    add amp, amp, astride
    vfmacc.vf v1, ft1, v16
    addi kt, kt, -1
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
    addi akp, akp, 4
    vfmacc.vf v11, ft11, v16
    beqz kt, 1f
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
    beqz kt, 1f
    flw ft3, (amp)
    add amp, amp, astride
    vfmacc.vf v15, ft15, v16
    vle32.v v16, (bkp)
    j k_loop

1:  vfmacc.vf v15, ft15, v16

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

    slli t6, nvl, 2
    add cnp, cnp, t6
    add bnp, bnp, t6
    sub nt, nt, nvl
    bnez nt, c_col_loop

    addi m, m, -16
    slli t6, astride, 4
    add ap, ap, t6
    slli t6, cstride, 4
    add cp, cp, t6

    slti t6, m, 16
    beqz t6, c_row_loop

end_rows:

exit:
    ld s0, 0(sp)
    ld s1, 8(sp)
    ld s2, 16(sp)
    addi sp, sp, FRAMESIZE
    ret

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
    li  ra, 0
    li  sp, 0
    li  gp, 0
    li  tp, 0
    li  t0, 0
    li  t1, 0
    li  t2, 0
    li  s0, 0
    li  s1, 0
    li  a0,0
    li  a1,0
    li  a2,0
    li  a3,0
    li  a4,0
    li  a5,0
    li  a6,0
    li  a7,0
    li  s2,0
    li  s3,0
    li  s4,0
    li  s5,0
    li  s6,0
    li  s7,0
    li  s8,0
    li  s9,0
    li  s10,0
    li  s11,0
    li  t3,0
    li  t4,0
    li  t5,0
    li  t6,0

    li t0, MSTATUS_FS | MSTATUS_XS | MSTATUS_VS
    csrs mstatus, t0

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
    la t0, 1f
    csrw mtvec, t0

    fssr    zero
    fmv.s.x f0, zero
    fmv.s.x f1, zero
    fmv.s.x f2, zero
    fmv.s.x f3, zero
    fmv.s.x f4, zero
    fmv.s.x f5, zero
    fmv.s.x f6, zero
    fmv.s.x f7, zero
    fmv.s.x f8, zero
    fmv.s.x f9, zero
    fmv.s.x f10,zero
    fmv.s.x f11,zero
    fmv.s.x f12,zero
    fmv.s.x f13,zero
    fmv.s.x f14,zero
    fmv.s.x f15,zero
    fmv.s.x f16,zero
    fmv.s.x f17,zero
    fmv.s.x f18,zero
    fmv.s.x f19,zero
    fmv.s.x f20,zero
    fmv.s.x f21,zero
    fmv.s.x f22,zero
    fmv.s.x f23,zero
    fmv.s.x f24,zero
    fmv.s.x f25,zero
    fmv.s.x f26,zero
    fmv.s.x f27,zero
    fmv.s.x f28,zero
    fmv.s.x f29,zero
    fmv.s.x f30,zero
    fmv.s.x f31,zero
1:
#endif

    la t0, trap_entry
    csrw mtvec, t0

.option push
.option norelax
    la gp, __global_pointer$
.option pop

    la  tp, _end + 63
    and tp, tp, -64

    csrr a0, mhartid
    li a1, 1
1:bgeu a0, a1, 1b

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

    SREG ra, 1*REGBYTES(sp)
    SREG sp, 2*REGBYTES(sp)
    SREG gp, 3*REGBYTES(sp)
    SREG tp, 4*REGBYTES(sp)
    SREG t0, 5*REGBYTES(sp)
    SREG t1, 6*REGBYTES(sp)
    SREG t2, 7*REGBYTES(sp)
    SREG s0, 8*REGBYTES(sp)
    SREG s1, 9*REGBYTES(sp)
    SREG a0, 10*REGBYTES(sp)
    SREG a1, 11*REGBYTES(sp)
    SREG a2, 12*REGBYTES(sp)
    SREG a3, 13*REGBYTES(sp)
    SREG a4, 14*REGBYTES(sp)
    SREG a5, 15*REGBYTES(sp)
    SREG a6, 16*REGBYTES(sp)
    SREG a7, 17*REGBYTES(sp)
    SREG s2, 18*REGBYTES(sp)
    SREG s3, 19*REGBYTES(sp)
    SREG s4, 20*REGBYTES(sp)
    SREG s5, 21*REGBYTES(sp)
    SREG s6, 22*REGBYTES(sp)
    SREG s7, 23*REGBYTES(sp)
    SREG s8, 24*REGBYTES(sp)
    SREG s9, 25*REGBYTES(sp)
    SREG s10, 26*REGBYTES(sp)
    SREG s11, 27*REGBYTES(sp)
    SREG t3, 28*REGBYTES(sp)
    SREG t4, 29*REGBYTES(sp)
    SREG t5, 30*REGBYTES(sp)
    SREG t6, 31*REGBYTES(sp)

    csrr a0, mcause
    csrr a1, mepc
    mv a2, sp
    jal handle_trap
    csrw mepc, a0

    li t0, MSTATUS_MPP
    csrs mstatus, t0

    LREG ra, 1*REGBYTES(sp)
    LREG sp, 2*REGBYTES(sp)
    LREG gp, 3*REGBYTES(sp)
    LREG tp, 4*REGBYTES(sp)
    LREG t0, 5*REGBYTES(sp)
    LREG t1, 6*REGBYTES(sp)
    LREG t2, 7*REGBYTES(sp)
    LREG s0, 8*REGBYTES(sp)
    LREG s1, 9*REGBYTES(sp)
    LREG a0, 10*REGBYTES(sp)
    LREG a1, 11*REGBYTES(sp)
    LREG a2, 12*REGBYTES(sp)
    LREG a3, 13*REGBYTES(sp)
    LREG a4, 14*REGBYTES(sp)
    LREG a5, 15*REGBYTES(sp)
    LREG a6, 16*REGBYTES(sp)
    LREG a7, 17*REGBYTES(sp)
    LREG s2, 18*REGBYTES(sp)
    LREG s3, 19*REGBYTES(sp)
    LREG s4, 20*REGBYTES(sp)
    LREG s5, 21*REGBYTES(sp)
    LREG s6, 22*REGBYTES(sp)
    LREG s7, 23*REGBYTES(sp)
    LREG s8, 24*REGBYTES(sp)
    LREG s9, 25*REGBYTES(sp)
    LREG s10, 26*REGBYTES(sp)
    LREG s11, 27*REGBYTES(sp)
    LREG t3, 28*REGBYTES(sp)
    LREG t4, 29*REGBYTES(sp)
    LREG t5, 30*REGBYTES(sp)
    LREG t6, 31*REGBYTES(sp)

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
    vec_memcpy:
    mv a3, a0
    loop:
    vsetvli t0, a2, e8, m8, ta, ma
    vle8.v v0, (a1)
    add a1, a1, t0
    sub a2, a2, t0
    vse8.v v0, (a3)
    add a3, a3, t0
    bnez a2, loop
    ret
    .text
    .balign 4
    .global vec_strcmp
vec_strcmp:
    li t1, 0
loop:
    vsetvli t0, zero, e8, m2, ta, ma
    add a0, a0, t1
    vle8ff.v v8, (a0)
    add a1, a1, t1
    vle8ff.v v16, (a1)

    vmseq.vi v0, v8, 0
    vmsne.vv v1, v8, v16
    vmor.mm v0, v0, v1

    vfirst.m a2, v0
    csrr t1, vl

    bltz a2, loop

    add a0, a0, a2
    lbu a3, (a0)

    add a1, a1, a2
    lbu a4, (a1)

    sub a0, a3, a4

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
    csrr    ra, CSR_MHARTID
    slli    ra, ra, 8
main_post_csrr:
    addi    sp, ra, 1
    addi    gp, sp, 1
    addi    tp, gp, 1
    addi    t0, tp, 1
    addi    t1, t0, 1
    addi    t2, t1, 1
    addi    s0, t2, 1
    addi    s1, s0, 1
    addi    a0, s1, 1
    addi    a1, a0, 1
    addi    a2, a1, 1
    addi    a3, a2, 1
    addi    a4, a3, 1
    addi    a5, a4, 1
    addi    a6, a5, 1
    addi    a7, a6, 1
    addi    s2, a7, 1
    addi    s3, s2, 1
    addi    s4, s3, 1
    addi    s5, s4, 1
    addi    s6, s5, 1
    addi    s7, s6, 1
    addi    s8, s7, 1
    addi    s9, s8, 1
    addi    s10, s9, 1
    addi    s11, s10, 1
    addi    t3, s11, 1
    addi    t4, t3, 1
    addi    t5, t4, 1
    addi    t6, t5, 1
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
    vsetvli zero, s0, e8, m1
    vle8.v v4, 0(a0)
    ret

vector_store_v4:
    // a0: point to memory to store to
    csrr    s0, vlenb
    vsetvli zero, s0, e8, m1
    vse8.v v4, 0(a0)
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
    addi   t2, zero, 1 	
    slli   t2, t2, 20 // U_EXTENSION
    and    t2, t1, t2
    bne    zero, t2, 1f
    addi   t2, zero, 1
    slli   t2, t2, 18 // S_EXTENSION	
    and    t2, t1, t2
    bne    zero, t2, 1f
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

.option push
.option norelax
    la gp, __global_pointer$
.option pop

    la    sp, stack_bottom
    li    t1, STACK_SIZE
    csrr  t0, CSR_MHARTID
    addi  t0, t0, 1
1:
    add   sp, sp, t1
    addi  t0, t0, -1
    bnez  t0, 1b

    la t2, 2f
    csrrw t2, mtvec, t2
    li    t0, ~0
1:
    addi  t0, t0, 1
    csrw  CSR_TSELECT, t0
    csrw  CSR_TDATA1, zero
    csrr  t1, CSR_TSELECT
    beq   t0, t1, 1b
.p2align 2
2:
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

wait_until_initialized:
    la    t0, initialized
1:
    lw    t1, 0(t0)
    beqz  t1, 1b
#endif

    j     _init

.align 2
trap_entry:
    addi sp, sp, -32*REGBYTES

    SREG ra, 1*REGBYTES(sp)
    SREG sp, 2*REGBYTES(sp)
    SREG gp, 3*REGBYTES(sp)
    SREG tp, 4*REGBYTES(sp)
    SREG t0, 5*REGBYTES(sp)
    SREG t1, 6*REGBYTES(sp)
    SREG t2, 7*REGBYTES(sp)
    SREG s0, 8*REGBYTES(sp)
    SREG s1, 9*REGBYTES(sp)
    SREG a0, 10*REGBYTES(sp)
    SREG a1, 11*REGBYTES(sp)
    SREG a2, 12*REGBYTES(sp)
    SREG a3, 13*REGBYTES(sp)
    SREG a4, 14*REGBYTES(sp)
    SREG a5, 15*REGBYTES(sp)
#ifndef RV32E
    SREG a6, 16*REGBYTES(sp)
    SREG a7, 17*REGBYTES(sp)
    SREG s2, 18*REGBYTES(sp)
    SREG s3, 19*REGBYTES(sp)
    SREG s4, 20*REGBYTES(sp)
    SREG s5, 21*REGBYTES(sp)
    SREG s6, 22*REGBYTES(sp)
    SREG s7, 23*REGBYTES(sp)
    SREG s8, 24*REGBYTES(sp)
    SREG s9, 25*REGBYTES(sp)
    SREG s10, 26*REGBYTES(sp)
    SREG s11, 27*REGBYTES(sp)
    SREG t3, 28*REGBYTES(sp)
    SREG t4, 29*REGBYTES(sp)
    SREG t5, 30*REGBYTES(sp)
    SREG t6, 31*REGBYTES(sp)
#endif

    csrr a0, mcause
    csrr a1, mepc
    mv a2, sp
    jal handle_trap
    csrw mepc, a0

    li t0, MSTATUS_MPP
    csrs mstatus, t0

    LREG ra, 1*REGBYTES(sp)
    LREG sp, 2*REGBYTES(sp)
    LREG gp, 3*REGBYTES(sp)
    LREG tp, 4*REGBYTES(sp)
    LREG t0, 5*REGBYTES(sp)
    LREG t1, 6*REGBYTES(sp)
    LREG t2, 7*REGBYTES(sp)
    LREG s0, 8*REGBYTES(sp)
    LREG s1, 9*REGBYTES(sp)
    LREG a0, 10*REGBYTES(sp)
    LREG a1, 11*REGBYTES(sp)
    LREG a2, 12*REGBYTES(sp)
    LREG a3, 13*REGBYTES(sp)
    LREG a4, 14*REGBYTES(sp)
    LREG a5, 15*REGBYTES(sp)
#ifndef RV32E
    LREG a6, 16*REGBYTES(sp)
    LREG a7, 17*REGBYTES(sp)
    LREG s2, 18*REGBYTES(sp)
    LREG s3, 19*REGBYTES(sp)
    LREG s4, 20*REGBYTES(sp)
    LREG s5, 21*REGBYTES(sp)
    LREG s6, 22*REGBYTES(sp)
    LREG s7, 23*REGBYTES(sp)
    LREG s8, 24*REGBYTES(sp)
    LREG s9, 25*REGBYTES(sp)
    LREG s10, 26*REGBYTES(sp)
    LREG s11, 27*REGBYTES(sp)
    LREG t3, 28*REGBYTES(sp)
    LREG t4, 29*REGBYTES(sp)
    LREG t5, 30*REGBYTES(sp)
    LREG t6, 31*REGBYTES(sp)
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
    SREG    sp, 0(ra)
    SREG    gp, 8(ra)
    SREG    tp, 16(ra)
    SREG    t0, 24(ra)
    SREG    t1, 32(ra)
    SREG    t2, 40(ra)
    SREG    s0, 48(ra)
    SREG    s1, 56(ra)
    SREG    a0, 64(ra)
    SREG    a1, 72(ra)
    SREG    a2, 80(ra)
    SREG    a3, 88(ra)
    SREG    a4, 96(ra)
    SREG    a5, 104(ra)
#ifndef RV32E
    SREG    a6, 112(ra)
    SREG    a7, 120(ra)
    SREG    s2, 128(ra)
    SREG    s3, 136(ra)
    SREG    s4, 144(ra)
    SREG    s5, 152(ra)
    SREG    s6, 160(ra)
    SREG    s7, 168(ra)
    SREG    s8, 176(ra)
    SREG    s9, 184(ra)
    SREG    s10, 192(ra)
    SREG    s11, 200(ra)
    SREG    t3, 208(ra)
    SREG    t4, 216(ra)
    SREG    t5, 224(ra)
    SREG    t6, 232(ra)
#endif

    csrr    ra, CSR_MSCRATCH

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
    la      t0, page_table
    srli    t0, t0, PGSHIFT
    csrw    CSR_SATP, t0

    csrr    t1, CSR_MSTATUS
#if XLEN == 32
    li      t0, (MSTATUS_MPRV | (SATP_MODE_SV32 << 24))
#else
    li      t0, (MSTATUS_MPRV | (SATP_MODE_SV39 << 24))
#endif
    or      t1, t0, t1
    csrw    CSR_MSTATUS, t1

    la      t0, (loop - 0x80000000)
    csrw    CSR_MEPC, t0

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
