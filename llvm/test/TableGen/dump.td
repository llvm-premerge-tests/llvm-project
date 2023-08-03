// RUN: llvm-tblgen %s -o %t 2>&1 | FileCheck %s

defvar a = 1;
defvar b = !add(!dump("a = ", a), 1);
dump "b = ", b;
// CHECK-LABEL: a = 1
// CHECK-LABEL: b = 2

class C<int i> {
  int I = i;
}
def c: C<3>;
dump "c = ", c;
// CHECK-LABEL: c = c {	// C
// CHECK-NEXT:     int I = 3;
// CHECK-NEXT:  }

dump "anon = ", C<5>;
// CHECK-LABEL: anon = anonymous_{{.*}} {	// C
// CHECK-NEXT:    int I = 5;
// CHECK-NEXT:  }

defvar lst = [C<1>, C<2>];
dump "lst = ", lst;
// CHECK-LABEL: lst = [anonymous_{{.*}}, anonymous_{{.*}}]

defvar p5 = !foreach(i, lst, !dump("elt = ", i));
// CHECK-LABEL: elt = anonymous_{{.*}} {	// C
// CHECK-NEXT:    int I = 1;
// CHECK-NEXT:  }
// CHECK-LABEL: elt = anonymous_{{.*}} {	// C
// CHECK-NEXT:    int I = 2;
// CHECK-NEXT:  }

foreach e = lst in {
  dump "e = ", e;
}
// CHECK-LABEL: e = anonymous_{{.*}} {	// C
// CHECK-NEXT:    int I = 1;
// CHECK-NEXT:  }
// CHECK-LABEL: e = anonymous_{{.*}} {	// C
// CHECK-NEXT:    int I = 2;
// CHECK-NEXT:  }

class CC<int i> {
  dump "CC:i = ", i;
  int I = !add(i, 1);
}
def cc : CC<1>;
// CHECK-LABEL: CC:i = 1

class MCC<int i> {
  int I = !add(i, 1);
}
multiclass MC<int i> {
  dump "MC:i = ", i;
  def : MCC<i>;
}
defm : MC<2>;
// CHECK-LABEL: MC:i = 2
