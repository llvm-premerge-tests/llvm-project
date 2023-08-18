// RUN: %clang_cc1 -std=c++2b -fno-rtti -emit-llvm -triple=x86_64-pc-win32 -o - %s  2>/dev/null | FileCheck %s
struct S {
friend void test();
public:
    void a(this auto){}
    void b(this auto&){}
    void c(this S){}
    void c(this S, int){}
private:
    void d(this auto){}
    void e(this auto&){}
    void f(this S){}
    void f(this S, int){}
protected:
    void g(this auto){}
    void h(this auto&){}
    void i(this S){}
    void i(this S, int){}
};

void test() {
    S s;
    s.a();
    // CHECK: call void @"??$a@US@@@S@@SAX_VU0@@Z"
    s.b();
    // CHECK: call void @"??$b@US@@@S@@SAX_VAEAU0@@Z"
    s.c();
    // CHECK: call void @"?c@S@@SAX_VU1@@Z"
    s.c(0);
    // CHECK: call void @"?c@S@@SAX_VU1@H@Z"
    s.d();
    // CHECK: call void @"??$d@US@@@S@@CAX_VU0@@Z"
    s.e();
    // CHECK: call void @"??$e@US@@@S@@CAX_VAEAU0@@Z"
    s.f();
    // CHECK: call void @"?f@S@@CAX_VU1@@Z"
    s.f(0);
    // CHECK: call void @"?f@S@@CAX_VU1@H@Z"
    s.g();
    // CHECK: call void @"??$g@US@@@S@@KAX_VU0@@Z"
    s.h();
    // CHECK: call void @"??$h@US@@@S@@KAX_VAEAU0@@Z"
    s.i();
    // CHECK: call void @"?i@S@@KAX_VU1@@Z"
    s.i(0);
    // CHECK: call void @"?i@S@@KAX_VU1@H@Z"
}
