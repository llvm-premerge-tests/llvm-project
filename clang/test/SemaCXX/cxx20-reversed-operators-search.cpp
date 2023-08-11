// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify %s
// expected-no-diagnostics


namespace members {

struct X {
  virtual ~X();
  virtual bool operator ==(X);
  bool operator !=(X);
};

struct Y {
  virtual ~Y();
  virtual bool operator ==(Y);
  bool operator !=(Y);
};

struct Z : X, Y {
  virtual bool operator==(Z);
  bool operator==(X) override;
  bool operator==(Y) override;
};

void test() {
  bool b = Z() == Z();
}

} // namespace members

