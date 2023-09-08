//===-- DeclarePureVirtualsTests.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ParsedAST.h"
#include "TestTU.h"
#include "TweakTesting.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/DeclCXX.h"
#include "llvm/ADT/ArrayRef.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TWEAK_TEST(DeclarePureVirtuals);

// TODO: remove and undo changes
#if 0

void printer(llvm::ArrayRef<const Decl*> topLevelDecls) {
  for (const auto *delc : topLevelDecls) {
    if (const CXXRecordDecl *rc = dyn_cast<CXXRecordDecl>(delc)) {
      if (rc->getName() == "C") {
        CXXFinalOverriderMap FinalOverriders;
        rc->getFinalOverriders(FinalOverriders);

        for (const auto &bar : FinalOverriders) { // TODO: see line 44
          // const CXXMethodDecl *Method = bar.first;
          llvm::outs() << bar.first->getQualifiedNameAsString() << ", " << bar.second.size() << "\n";
          const OverridingMethods &Overrides = bar.second;

          for (const std::pair<unsigned int,
                               llvm::SmallVector<UniqueVirtualMethod, 4>>
                   &meow : // TODO
               Overrides) {
            llvm::outs() << "    " << meow.first << ", " << meow.second.size() << "\n";
            const auto &IdontKnowWhatThisIs = meow.second; // TODO
            for (const UniqueVirtualMethod &Override : IdontKnowWhatThisIs) {
              llvm::outs() << "        "
                           << Override.Method->getQualifiedNameAsString()
                           << ", " << Override.Subobject << ", " << Override.InVirtualSubobject
                           << "\n";
            }
          }
        }
      }
    }
  }
}

TEST_F(DeclarePureVirtualsTest, Error) {
  auto foo = [this](const char* txt) {
    const auto p = buildErr(txt);
    llvm::outs() << txt << "\n";
    printer(p.first.getLocalTopLevelDecls());
    llvm::outs() << "\n======================\n";
  };

  foo(R"cpp(
class MyBase { virtual void foo() = 0; };
class C : MyBase {};
  )cpp");

  foo(R"cpp(
class MyBase { virtual void foo() = 0; };
class C : MyBase { void foo() override; };
  )cpp");

  foo(R"cpp(
class MyBase { virtual void foo() = 0; };
class B : MyBase { void foo() override; };
class C : MyBase {};
  )cpp");

  foo(R"cpp(
class MyBase { virtual void foo() = 0; };
class B : MyBase {};
class C : MyBase { void foo() override; };
  )cpp");

  foo(R"cpp(
class MyBase { virtual void foo() = 0; };
class B : MyBase { void foo() override; };
class C : MyBase { void foo() override; };
  )cpp");

  foo(R"cpp(
class MyBase { virtual void foo() = 0; };
class A : MyBase { void foo() override; };
class B : MyBase { void foo() override; };
class C : A, B {};
  )cpp");

  foo(R"cpp(
class A : { virtual void foo(); };
class B : { virtual void bar(); };
class C : A, B {};
  )cpp");

  foo(R"cpp(
class A : { virtual void foo(); };
class C : virtual A {};
  )cpp");

  foo(R"cpp(
class A : { virtual void foo(); };
class B : { virtual void bar(); };
class C : virtual A, virtual B {};
  )cpp");

  foo(R"cpp(
class A : { virtual void foo(); };
class B : { virtual void bar(); };
class C : A, B { void foo() override; void bar() override; };
  )cpp");

  foo(R"cpp(
    class MyBase {
      virtual void myFunction() = 0;
    };

    class A : virtual MyBase {
      void myFunction() override;
    };

    class B : virtual MyBase {
      void myFunction() override;
    };

    class C : A, B {};
  )cpp");

  foo(R"cpp(
class A : { virtual void foo() = 0; };
class B : A { virtual void foo() = 0; };
class C : B { virtual void foo() = 0; };
  )cpp");

  foo(R"cpp(
class MyBase {
  virtual void myFunction() = 0;
};

class C : MyBase {
  void myFunction() override;
};
  )cpp"
);
}

#else

TEST_F(DeclarePureVirtualsTest, AvailabilityTriggerOnBaseSpecifier) {
  EXPECT_AVAILABLE(R"cpp(
    class MyBase {
      virtual void myFunction() = 0;
    };

    class MyDerived : ^p^u^b^l^i^c^ ^M^y^B^a^s^e {
    };

    class MyDerived2 : ^M^y^B^a^s^e {
    };
  )cpp");
}

TEST_F(DeclarePureVirtualsTest, AvailabilityTriggerOnClass) {
  EXPECT_AVAILABLE(R"cpp(
    class MyBase {
      virtual void myFunction() = 0;
    };

    class ^M^y^D^e^r^i^v^e^d: public MyBase {^
    // but not here, see AvailabilityTriggerInsideClass
    ^};
  )cpp");
}

TEST_F(DeclarePureVirtualsTest, AvailabilityTriggerInsideClass) {
  // TODO: this should actually be available but I don't know how to implement
  // it: the common node of the selection returns the TU, so I get no
  // information about which class we're in.
  EXPECT_UNAVAILABLE(R"cpp(
    class MyBase {
      virtual void myFunction() = 0;
    };

    class MyDerived : public MyBase {
    ^
    };
  )cpp");
}

TEST_F(DeclarePureVirtualsTest, UnavailabilityNoBases) {
  EXPECT_UNAVAILABLE(R"cpp(
    class ^N^o^D^e^r^i^v^e^d^ ^{^
    ^};
  )cpp");
}

// TODO: should the tweak available if there are no pure virtual functions and
// do nothing? or should it be unavailable?
TEST_F(DeclarePureVirtualsTest, UnavailabilityNoVirtuals) {
  EXPECT_UNAVAILABLE(R"cpp(
    class MyBase {
      void myFunction();
    };

    class MyDerived : public MyBase {
    ^
    };
  )cpp");
}

TEST_F(DeclarePureVirtualsTest, UnavailabilityNoPureVirtuals) {
  EXPECT_UNAVAILABLE(R"cpp(
    class MyBase {
      virtual void myFunction();
    };

    class MyDerived : public MyBase {
    ^
    };
  )cpp");
}

TEST_F(DeclarePureVirtualsTest, UnavailabilityNoVirtualsInSelectedBase) {
  EXPECT_UNAVAILABLE(R"cpp(
    class MyBase {
      void myFunction();
    };

    class MyOtherBase {
      virtual void otherFunction() = 0;
    };

    class MyDerived : ^p^u^b^l^i^c^ ^M^y^B^a^s^e, MyOtherBase {
    };
  )cpp");
}

TEST_F(DeclarePureVirtualsTest, UnavailabilityOnForwardDecls) {
  EXPECT_UNAVAILABLE(R"cpp(
  class ^M^y^C^l^a^s^s^;
  )cpp");
}

TEST_F(DeclarePureVirtualsTest, SinglePureVirtualFunction) {
  const char *Test = R"cpp(
class MyBase {
  virtual void myFunction() = 0;
};

class MyDerived : pub^lic MyBase {
};
  )cpp";

  const char *Expected = R"cpp(
class MyBase {
  virtual void myFunction() = 0;
};

class MyDerived : public MyBase {
void myFunction() override;
};
  )cpp";

  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DeclarePureVirtualsTest, MultipleInheritanceFirstClass) {
  const char *Test = R"cpp(
class MyBase1 {
  virtual void myFunction1() = 0;
};

class MyBase2 {
  virtual void myFunction2() = 0;
};

class MyDerived : pub^lic MyBase1, public MyBase2 {
};
  )cpp";

  // TODO: use qualified name to refer to function to show base class; maybe use
  // original declaration?
  const char *ExpectedTitle = R"cpp(Override pure virtual function:
void myFunction1() override;
)cpp";

  EXPECT_EQ(title(Test), ExpectedTitle);

  const char *Expected = R"cpp(
class MyBase1 {
  virtual void myFunction1() = 0;
};

class MyBase2 {
  virtual void myFunction2() = 0;
};

class MyDerived : public MyBase1, public MyBase2 {
void myFunction1() override;
};
  )cpp";

  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DeclarePureVirtualsTest, MultipleInheritanceSecondClass) {
  const char *Test = R"cpp(
class MyBase1 {
  virtual void myFunction1() = 0;
};

class MyBase2 {
  virtual void myFunction2() = 0;
};

class MyDerived : public MyBase1, pub^lic MyBase2 {
};
  )cpp";

  const char *Expected = R"cpp(
class MyBase1 {
  virtual void myFunction1() = 0;
};

class MyBase2 {
  virtual void myFunction2() = 0;
};

class MyDerived : public MyBase1, public MyBase2 {
void myFunction2() override;
};
  )cpp";

  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DeclarePureVirtualsTest, SingleInheritanceMultiplePureVirtualFunctions) {
  const char *Test = R"cpp(
class MyBase {
  virtual void myFunction1() = 0;
  virtual void myFunction2() = 0;
};

class MyDerived : pub^lic MyBase {
};
  )cpp";

  const char *Expected = R"cpp(
class MyBase {
  virtual void myFunction1() = 0;
  virtual void myFunction2() = 0;
};

class MyDerived : public MyBase {
void myFunction1() override;
void myFunction2() override;
};
  )cpp";

  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DeclarePureVirtualsTest, SingleInheritanceMixedVirtualFunctions) {
  const char *Test = R"cpp(
class MyBase {
  virtual void myFunction1() = 0;
  virtual void myFunction2();
};

class MyDerived : pub^lic MyBase {
};
  )cpp";

  const char *Expected = R"cpp(
class MyBase {
  virtual void myFunction1() = 0;
  virtual void myFunction2();
};

class MyDerived : public MyBase {
void myFunction1() override;
};
  )cpp";

  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DeclarePureVirtualsTest,
       TwoLevelsInheritanceOnePureVirtualFunctionInTopBase) {
  const char *Test = R"cpp(
class MyBase {
  virtual void myFunction1() = 0;
};

class MyIntermediate : public MyBase {
};

class MyDerived : pub^lic MyIntermediate {
};
  )cpp";

  const char *Expected = R"cpp(
class MyBase {
  virtual void myFunction1() = 0;
};

class MyIntermediate : public MyBase {
};

class MyDerived : public MyIntermediate {
void myFunction1() override;
};
  )cpp";

  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DeclarePureVirtualsTest,
       TwoLevelsInheritancePureVirtualFunctionsInBothBases) {
  const char *Test = R"cpp(
class MyBase {
  virtual void myFunction1() = 0;
};

class MyIntermediate : public MyBase {
  virtual void myFunction2() = 0;
};

class MyDerived : pub^lic MyIntermediate {
};
  )cpp";

  const char *Expected = R"cpp(
class MyBase {
  virtual void myFunction1() = 0;
};

class MyIntermediate : public MyBase {
  virtual void myFunction2() = 0;
};

class MyDerived : public MyIntermediate {
void myFunction1() override;
void myFunction2() override;
};
  )cpp";

  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DeclarePureVirtualsTest,
       TwoLevelInheritanceFunctionNoLongerPureVirtual) {
  const char *Test = R"cpp(
class MyBase {
  virtual void myFunction1() = 0;
};

class MyIntermediate : public MyBase {
  void myFunction1() override {}
};

class MyDerived : pub^lic MyIntermediate {
};
  )cpp";

  EXPECT_UNAVAILABLE(Test);
}

TEST_F(DeclarePureVirtualsTest, VirtualFunctionWithDefaultParameters) {
  const char *Test = R"cpp(
class MyBase {
  virtual void myFunction(int x, int y = 42) = 0;
};

class MyDerived : pub^lic MyBase {
};
  )cpp";

  const char *Expected = R"cpp(
class MyBase {
  virtual void myFunction(int x, int y = 42) = 0;
};

class MyDerived : public MyBase {
void myFunction(int x, int y = 42) override;
};
  )cpp";

  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DeclarePureVirtualsTest, FunctionOverloading) {
  const char *Test = R"cpp(
class MyBase {
  virtual void myFunction(int x) = 0;
  virtual void myFunction(float x) = 0;
};

class MyDerived : pub^lic MyBase {
};
  )cpp";

  const char *Expected = R"cpp(
class MyBase {
  virtual void myFunction(int x) = 0;
  virtual void myFunction(float x) = 0;
};

class MyDerived : public MyBase {
void myFunction(int x) override;
void myFunction(float x) override;
};
  )cpp";

  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DeclarePureVirtualsTest, OverrideAlreadyExisting) {
  const char *Test = R"cpp(
class MyBase2 {
  virtual void foo() = 0;
};

class C : My^Base2 {
  void foo() override;
};
  )cpp";

  EXPECT_UNAVAILABLE(Test);
}

TEST_F(DeclarePureVirtualsTest, PureVirtualRedeclarationAlreadyExisting) {
  const char *Test = R"cpp(
class MyBase1 {
  virtual void foo() = 0;
};

class MyBase2 : MyBase1 {
  virtual void foo() = 0;
};

class C : My^Base2 {
  virtual void foo() = 0;
};
  )cpp";

  EXPECT_UNAVAILABLE(Test);
}

TEST_F(DeclarePureVirtualsTest, TwoOverloadsOnlyOnePureVirtual) {
  const char *Test = R"cpp(
class MyBase {
  virtual void myFunction(int x) = 0;
  virtual void myFunction(float x);
};

class MyDerived : pub^lic MyBase {
};
  )cpp";

  const char *Expected = R"cpp(
class MyBase {
  virtual void myFunction(int x) = 0;
  virtual void myFunction(float x);
};

class MyDerived : public MyBase {
void myFunction(int x) override;
};
  )cpp";

  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DeclarePureVirtualsTest, DerivedClassHasNonOverridingFunction) {
  const char *Test = R"cpp(
class MyBase {
  virtual void myFunction(int x) = 0;
};

class MyDerived : pub^lic MyBase {
  void myFunction(float x);
};
  )cpp";

  const char *Expected = R"cpp(
class MyBase {
  virtual void myFunction(int x) = 0;
};

class MyDerived : public MyBase {
  void myFunction(float x);
void myFunction(int x) override;
};
  )cpp";

  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DeclarePureVirtualsTest, PureVirtualFunctionWithNoexcept) {
  const char *Test = R"cpp(
class MyBase {
  virtual void myFunction() noexcept = 0;
};

class MyDerived : pub^lic MyBase {
};
  )cpp";

  const char *Expected = R"cpp(
class MyBase {
  virtual void myFunction() noexcept = 0;
};

class MyDerived : public MyBase {
void myFunction() noexcept override;
};
  )cpp";

  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DeclarePureVirtualsTest, MixOfVirtualAndNonVirtualMemberFunctions) {
  const char *Test = R"cpp(
class MyBase {
  virtual void myFunction1() = 0;
    void myFunction2();
};

class MyDerived : pub^lic MyBase {
};
  )cpp";

  const char *Expected = R"cpp(
class MyBase {
  virtual void myFunction1() = 0;
    void myFunction2();
};

class MyDerived : public MyBase {
void myFunction1() override;
};
  )cpp";

  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DeclarePureVirtualsTest, PureVirtualFunctionWithBody) {
  const char *Test = R"cpp(
class MyBase {
  virtual void myFunction1() = 0;
};

void MyBase::myFunction1() {
}

class MyDerived : pub^lic MyBase {
};
  )cpp";

  const char *Expected = R"cpp(
class MyBase {
  virtual void myFunction1() = 0;
};

void MyBase::myFunction1() {
}

class MyDerived : public MyBase {
void myFunction1() override;
};
  )cpp";

  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DeclarePureVirtualsTest, TraversalOrder) {
  const char *Test = R"cpp(
class MyBase0 {
  virtual void myFunction0() = 0;
};

class MyBase1 {
  virtual void myFunction1() = 0;
};

class MyBase2 : MyBase0, MyBase1 {
  virtual void myFunction2() = 0;
};

class MyBase3 {
  virtual void myFunction3() = 0;
};

class MyBase4 : MyBase2, MyBase3 {
  virtual void myFunction4() = 0;
};

class MyDerived : pub^lic MyBase4 {
};
  )cpp";

  const char *Expected = R"cpp(
class MyBase0 {
  virtual void myFunction0() = 0;
};

class MyBase1 {
  virtual void myFunction1() = 0;
};

class MyBase2 : MyBase0, MyBase1 {
  virtual void myFunction2() = 0;
};

class MyBase3 {
  virtual void myFunction3() = 0;
};

class MyBase4 : MyBase2, MyBase3 {
  virtual void myFunction4() = 0;
};

class MyDerived : public MyBase4 {
void myFunction0() override;
void myFunction1() override;
void myFunction2() override;
void myFunction3() override;
void myFunction4() override;
};
  )cpp";

  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DeclarePureVirtualsTest, AllBaseClassSpecifiers) {
  const char *Test = R"cpp(
class MyBase0 {
  virtual void myFunction0() = 0;
};

class MyBase1 {
  virtual void myFunction1() = 0;
};

class My^Derived : MyBase0, MyBase1 {
};
  )cpp";

  const char *Expected = R"cpp(
class MyBase0 {
  virtual void myFunction0() = 0;
};

class MyBase1 {
  virtual void myFunction1() = 0;
};

class MyDerived : MyBase0, MyBase1 {
void myFunction0() override;
void myFunction1() override;
};
  )cpp";

  EXPECT_EQ(apply(Test), Expected);
}

// TODO: I'd like to test the `[[noreturn]]` form as well but there doesn't seem
// to be a way at the moment to escape the `[[` which is interpreted by
// `apply()` as selection.
// TODO: implement attribute removal
//TEST_F(DeclarePureVirtualsTest, Attributes) {
//  const char *Test = R"cpp(
//class MyBase {
//  __attribute__((noreturn)) virtual void myFunction() = 0;
//};
//
//class MyDerived : pub^lic MyBase {
//};
//  )cpp";
//
//  // I don't think attributes should be copied. They're not required to
//  // override, and whether they semantically belong there depends on the
//  // particular attribute - we can leave this up to the user.
//  const char *Expected = R"cpp(
//class MyBase {
//  __attribute__((noreturn)) virtual void myFunction() = 0;
//};
//
//class MyDerived : public MyBase {
//void myFunction() override;
//};
//  )cpp";
//
//  EXPECT_EQ(apply(Test), Expected);
//}

TEST_F(DeclarePureVirtualsTest, NoWhitespaceBeforePureVirtSpecifier) {
  const char *Test = R"cpp(
class MyBase {
  virtual void myFunction()= 0;
};

class MyDerived : pub^lic MyBase {
};
  )cpp";

  const char *Expected = R"cpp(
class MyBase {
  virtual void myFunction()= 0;
};

class MyDerived : public MyBase {
void myFunction() override;
};
  )cpp";

  EXPECT_EQ(apply(Test), Expected);
}

TEST_F(DeclarePureVirtualsTest, MultiplePureVirtualsInDifferentSubobjects) {
  const char *Test = R"cpp(
class MyBase {
  virtual void myFunction() = 0;
};

class A : MyBase {
  virtual void myFunction() = 0;
};

class B : MyBase {
  virtual void myFunction() = 0;
};

class My^Derived : A, B {
};
  )cpp";

  const char *Expected = R"cpp(
class MyBase {
  virtual void myFunction() = 0;
};

class A : MyBase {
  virtual void myFunction() = 0;
};

class B : MyBase {
  virtual void myFunction() = 0;
};

class MyDerived : A, B {
void myFunction() override;
};
  )cpp";

  EXPECT_EQ(apply(Test), Expected);
}

// TODO: it's not clear how to identify that these two functions A::myFunction, B::myFunction will be overridden by the same function in MyDerived.
TEST_F(DeclarePureVirtualsTest, MultiplePureVirtualsOfSameNameButSeparateBases) {
  const char *Test = R"cpp(
class A {
  virtual void myFunction() = 0;
};

class B {
  virtual void myFunction() = 0;
};

class My^Derived : A, B {
};
  )cpp";

  const char *Expected = R"cpp(
class A {
  virtual void myFunction() = 0;
};

class B {
  virtual void myFunction() = 0;
};

class MyDerived : A, B {
  void myFunction() override;
};
  )cpp";

  EXPECT_EQ(apply(Test), Expected);
}

// This test should fail since MyBase is incomplete. No idea how to test that.
// TEST_F(DeclarePureVirtualsTest, IncompleteBassClass) {
//   const char *Test = R"cpp(
// class MyBase;
//
// class MyDerived : My^Base {
// };
//   )cpp";
//
//   const char *Expected = R"cpp(
// class MyBase;
//
// class MyDerived : MyBase {
// };
//   )cpp";
//
//   EXPECT_EQ(apply(Test), Expected);
// }

// This test should fail since MyBase is incomplete. No idea how to test that.
// This is a different failure mode since it probably fails to apply but not to
// prepare.
// TEST_F(DeclarePureVirtualsTest, IncompleteBassClass) {
//   const char *Test = R"cpp(
// class MyBase;
//
// class My^Derived : MyBase {
// };
//   )cpp";
//
//   const char *Expected = R"cpp(
// class MyBase;
//
// class MyDerived : MyBase {
// };
//   )cpp";
//
//   EXPECT_EQ(apply(Test), Expected);
// }

// This test should fail since MyBase is unknown. No idea how to test that.
// TEST_F(DeclarePureVirtualsTest, UnknownBaseSpecifier) {
//   const char *Test = R"cpp(
// class MyDerived : My^Base {
// };
//   )cpp";
//
//   const char *Expected = R"cpp(
// class MyDerived : MyBase {
// };
//   )cpp";
//
//   EXPECT_EQ(apply(Test), Expected);
// }

// This test should fail since MyBase is unknown. No idea how to test that.
// It's a different failure mode since it will probably fail to apply but not to
// prepare.
// TEST_F(DeclarePureVirtualsTest, UnknownBaseSpecifier) {
//   const char *Test = R"cpp(
// class My^Derived : MyBase {
// };
//   )cpp";
//
//   const char *Expected = R"cpp(
// class MyDerived : MyBase {
// };
//   )cpp";
//
//   EXPECT_EQ(apply(Test), Expected);
// }

#endif

} // namespace
} // namespace clangd
} // namespace clang
