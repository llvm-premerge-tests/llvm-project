// RUN: llvm-tblgen %s -o %t 2>&1 | FileCheck %s

defvar a = 1111;
defvar b = !add(!dump( a), 1);
dump  b;
// CHECK-LABEL: 1111
// CHECK-LABEL: 1112

class C<int i> {
  int I = i;
}
def c: C<3>;
dump c;
// CHECK-LABEL: c { // C
// CHECK-NEXT:     int I = 3;
// CHECK-NEXT:  }

dump C<5>;
// CHECK-LABEL: anonymous_{{.*}} { // C
// CHECK-NEXT:    int I = 5;
// CHECK-NEXT:  }

defvar lst = [C<1>, C<2>];
dump lst;
// CHECK-LABEL: [anonymous_{{.*}}, anonymous_{{.*}}]

defvar p5 = !foreach(i, lst, !dump(i));
// CHECK-LABEL: anonymous_{{.*}} { // C
// CHECK-NEXT:    int I = 1;
// CHECK-NEXT:  }
// CHECK-LABEL: anonymous_{{.*}} { // C
// CHECK-NEXT:    int I = 2;
// CHECK-NEXT:  }

foreach e = lst in {
  dump  e;
}
// CHECK-LABEL: anonymous_{{.*}} { // C
// CHECK-NEXT:    int I = 1;
// CHECK-NEXT:  }
// CHECK-LABEL: anonymous_{{.*}} { // C
// CHECK-NEXT:    int I = 2;
// CHECK-NEXT:  }

class CC<int i> {
  dump "i in CC<i> =";
  dump  i;
  int I = !add(i, 1);
}
def cc : CC<33>;
// CHECK-LABEL: i in CC<i> =
// CHECK-NEXT: 33

class MCC<int i> {
  int I = !add(i, 1);
}
multiclass MC<int i> {
  dump "i in MCC<i> = ";
  dump  i;
  def : MCC<i>;
}
defm : MC<27>;
// CHECK-LABEL: i in MCC<i> =
// CHECK-NEXT: 27
