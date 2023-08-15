/// Test that --print-supported-extensions lists supported extensions.

// REQUIRES: riscv-registered-target

// RUN: %clang -Werror --target=riscv64 --print-supported-extensions 2>&1 | \
// RUN:   FileCheck --strict-whitespace --match-full-lines %s

//       CHECK:Target: riscv64
//       CHECK:All available -march extensions for RISC-V
//       CHECK:	Name                Version
//  CHECK-NEXT:	i                   2.1
//  CHECK-NEXT:	e                   2.0
//  CHECK-NEXT:	m                   2.0
//  CHECK-NEXT:	a                   2.1
//  CHECK-NEXT:	f                   2.2
//  CHECK-NEXT:	d                   2.2
//  CHECK-NEXT:	c                   2.0
//  CHECK-NEXT:	v                   1.0
//  CHECK-NEXT:	h                   1.0
//  CHECK-NEXT:	zicbom              1.0
//  CHECK-NEXT:	zicbop              1.0
//  CHECK-NEXT:	zicboz              1.0
//  CHECK-NEXT:	zicntr              1.0
//  CHECK-NEXT:	zicsr               2.0
//  CHECK-NEXT:	zifencei            2.0
//  CHECK-NEXT:	zihintpause         2.0
//  CHECK-NEXT:	zihpm               1.0
//  CHECK-NEXT:	zmmul               1.0
//  CHECK-NEXT:	zawrs               1.0
//  CHECK-NEXT:	zfh                 1.0
//  CHECK-NEXT:	zfhmin              1.0
//  CHECK-NEXT:	zfinx               1.0
//  CHECK-NEXT:	zdinx               1.0
//  CHECK-NEXT:	zca                 1.0
//  CHECK-NEXT:	zcb                 1.0
//  CHECK-NEXT:	zcd                 1.0
//  CHECK-NEXT:	zce                 1.0
//  CHECK-NEXT:	zcf                 1.0
//  CHECK-NEXT:	zcmp                1.0
//  CHECK-NEXT:	zcmt                1.0
//  CHECK-NEXT:	zba                 1.0
//  CHECK-NEXT:	zbb                 1.0
//  CHECK-NEXT:	zbc                 1.0
//  CHECK-NEXT:	zbkb                1.0
//  CHECK-NEXT:	zbkc                1.0
//  CHECK-NEXT:	zbkx                1.0
//  CHECK-NEXT:	zbs                 1.0
//  CHECK-NEXT:	zk                  1.0
//  CHECK-NEXT:	zkn                 1.0
//  CHECK-NEXT:	zknd                1.0
//  CHECK-NEXT:	zkne                1.0
//  CHECK-NEXT:	zknh                1.0
//  CHECK-NEXT:	zkr                 1.0
//  CHECK-NEXT:	zks                 1.0
//  CHECK-NEXT:	zksed               1.0
//  CHECK-NEXT:	zksh                1.0
//  CHECK-NEXT:	zkt                 1.0
//  CHECK-NEXT:	zve32f              1.0
//  CHECK-NEXT:	zve32x              1.0
//  CHECK-NEXT:	zve64d              1.0
//  CHECK-NEXT:	zve64f              1.0
//  CHECK-NEXT:	zve64x              1.0
//  CHECK-NEXT:	zvfh                1.0
//  CHECK-NEXT:	zvl1024b            1.0
//  CHECK-NEXT:	zvl128b             1.0
//  CHECK-NEXT:	zvl16384b           1.0
//  CHECK-NEXT:	zvl2048b            1.0
//  CHECK-NEXT:	zvl256b             1.0
//  CHECK-NEXT:	zvl32768b           1.0
//  CHECK-NEXT:	zvl32b              1.0
//  CHECK-NEXT:	zvl4096b            1.0
//  CHECK-NEXT:	zvl512b             1.0
//  CHECK-NEXT:	zvl64b              1.0
//  CHECK-NEXT:	zvl65536b           1.0
//  CHECK-NEXT:	zvl8192b            1.0
//  CHECK-NEXT:	zhinx               1.0
//  CHECK-NEXT:	zhinxmin            1.0
//  CHECK-NEXT:	svinval             1.0
//  CHECK-NEXT:	svnapot             1.0
//  CHECK-NEXT:	svpbmt              1.0
//  CHECK-NEXT:	xcvalu              1.0
//  CHECK-NEXT:	xcvbi               1.0
//  CHECK-NEXT:	xcvbitmanip         1.0
//  CHECK-NEXT:	xcvmac              1.0
//  CHECK-NEXT:	xcvsimd             1.0
//  CHECK-NEXT:	xsfcie              1.0
//  CHECK-NEXT:	xsfvcp              1.0
//  CHECK-NEXT:	xtheadba            1.0
//  CHECK-NEXT:	xtheadbb            1.0
//  CHECK-NEXT:	xtheadbs            1.0
//  CHECK-NEXT:	xtheadcmo           1.0
//  CHECK-NEXT:	xtheadcondmov       1.0
//  CHECK-NEXT:	xtheadfmemidx       1.0
//  CHECK-NEXT:	xtheadmac           1.0
//  CHECK-NEXT:	xtheadmemidx        1.0
//  CHECK-NEXT:	xtheadmempair       1.0
//  CHECK-NEXT:	xtheadsync          1.0
//  CHECK-NEXT:	xtheadvdot          1.0
//  CHECK-NEXT:	xventanacondops     1.0
// CHECK-EMPTY:
//       CHECK:Experimental extensions
//  CHECK-NEXT:	zicond              1.0
//  CHECK-NEXT:	zihintntl           0.2
//  CHECK-NEXT:	zacas               1.0
//  CHECK-NEXT:	zfa                 0.2
//  CHECK-NEXT:	zfbfmin             0.8
//  CHECK-NEXT:	ztso                0.1
//  CHECK-NEXT:	zvbb                1.0
//  CHECK-NEXT:	zvbc                1.0
//  CHECK-NEXT:	zvfbfmin            0.8
//  CHECK-NEXT:	zvfbfwma            0.8
//  CHECK-NEXT:	zvkg                1.0
//  CHECK-NEXT:	zvkn                1.0
//  CHECK-NEXT:	zvknc               1.0
//  CHECK-NEXT:	zvkned              1.0
//  CHECK-NEXT:	zvkng               1.0
//  CHECK-NEXT:	zvknha              1.0
//  CHECK-NEXT:	zvknhb              1.0
//  CHECK-NEXT:	zvks                1.0
//  CHECK-NEXT:	zvksc               1.0
//  CHECK-NEXT:	zvksed              1.0
//  CHECK-NEXT:	zvksg               1.0
//  CHECK-NEXT:	zvksh               1.0
//  CHECK-NEXT:	zvkt                1.0
//  CHECK-NEXT:	smaia               1.0
//  CHECK-NEXT:	ssaia               1.0
// CHECK-EMPTY:
//       CHECK:Use -march to specify the target's extension.
//  CHECK-NEXT:For example, clang -march=rv32i_v1p0
