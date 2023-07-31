// RUN: %clang_cc1 -std=c++2b -fsyntax-only -verify %s

namespace std {
constexpr inline bool
  is_constant_evaluated() noexcept {
    if consteval { return true; } else { return false; }
  }
} // namespace std

namespace P1938 {
  constexpr int f1() {
  if constexpr (!std::is_constant_evaluated() && sizeof(int) == 4) { // expected-warning {{always evaluate to true}}
    return 0;
  }
  if (std::is_constant_evaluated()) {
    return 42;
  } else {
    if constexpr (std::is_constant_evaluated()) { // expected-warning {{always evaluate to true}}
      return 0;
    }
  }
  return 7;
}


consteval int f2() {
  if (std::is_constant_evaluated() && f1()) { // expected-warning {{always evaluate to true}}
    return 42;
  }
  return 7;
}


int f3() {
  if (std::is_constant_evaluated() && f1()) { // expected-warning {{always evaluate to false}}
    return 42;
  }
  return 7;
}
}

void non_qual() {
  int ff = std::is_constant_evaluated(); // expected-warning {{always evaluate to false}}
  const int aa = std::is_constant_evaluated();
  constexpr int tt = std::is_constant_evaluated(); // expected-warning {{always evaluate to true}}
  static int bb = std::is_constant_evaluated();
  constexpr int cc = [](){
    if consteval {return 8;} // expected-warning {{always true}}
  }();
  auto lamda = []() {
    if consteval {return 8;}
    else {return 4;}
  };
  auto lamda_const = []() consteval {
    if consteval {return 8;} // expected-warning {{always true}}
    else {return 4;}
  };
  if consteval { // expected-warning {{always false}}
    int b = std::is_constant_evaluated(); // expected-warning {{always evaluate to true}}
  }
}

constexpr void in_constexpr() {
  int aa = std::is_constant_evaluated();
  constexpr int bb = std::is_constant_evaluated(); // expected-warning {{always evaluate to true}}
  const int cc = std::is_constant_evaluated();
  if consteval {
    int dd = std::is_constant_evaluated(); // expected-warning {{always evaluate to true}}
    constexpr int ee = std::is_constant_evaluated(); // expected-warning {{always evaluate to true}}
    const int ff = std::is_constant_evaluated(); // expected-warning {{always evaluate to true}}
  } else {
    int dd = std::is_constant_evaluated(); // expected-warning {{always evaluate to false}}
    constexpr int ee = std::is_constant_evaluated(); // expected-warning {{always evaluate to true}}
    const int ff = std::is_constant_evaluated();
    const int qq = std::is_constant_evaluated() ? dd : 3;
  }

  if consteval {
    if consteval {} // expected-warning {{always true}}
    if !consteval {} // expected-warning {{always false}}
  } else {
    if consteval {} // expected-warning {{always false}}
    if !consteval {} // expected-warning {{always true}}
  }
  if !consteval {
    if consteval {} // expected-warning {{always false}}
    if !consteval {} // expected-warning {{always true}}
  } else {
    if consteval {} // expected-warning {{always true}}
    if !consteval {} // expected-warning {{always false}}
  }
}

consteval void in_consteval() {
  int aa = std::is_constant_evaluated(); // expected-warning {{always evaluate to true}}
  constexpr int bb = std::is_constant_evaluated(); // expected-warning {{always evaluate to true}}
  const int cc = std::is_constant_evaluated(); // expected-warning {{always evaluate to true}}
  auto lambda = []() {
  int a(std::is_constant_evaluated()); // expected-warning {{always evaluate to true}}
  constexpr int b = std::is_constant_evaluated(); // expected-warning {{always evaluate to true}}
  const int c = std::is_constant_evaluated(); // expected-warning {{always evaluate to true}}
  };
  if !consteval {} // expected-warning {{always false}}
}

static_assert(std::is_constant_evaluated()); // expected-warning {{always evaluate to true}}
static_assert(__builtin_is_constant_evaluated()); // expected-warning {{always evaluate to true}}

template <bool b>
void templ() {
  if constexpr(std::is_constant_evaluated()) {} // expected-warning {{always evaluate to true}}
  constexpr bool c = std::is_constant_evaluated(); // expected-warning {{always evaluate to true}}
  if consteval {} // expected-warning {{always false}}
}

template <> void templ<std::is_constant_evaluated()>() { // expected-warning {{always evaluate to true}}
  if constexpr(std::is_constant_evaluated()) {} // expected-warning {{always evaluate to true}}
  constexpr bool c = std::is_constant_evaluated(); // expected-warning {{always evaluate to true}}
  if consteval {} // expected-warning {{always false}}
  templ<false>();
}

static_assert([] {
    if consteval { // expected-warning {{always true}}
      return 0;
    } else {
      return 1;
    }
  }() == 0);
constexpr bool b = __builtin_is_constant_evaluated(); // expected-warning {{always evaluate to true}}
constexpr bool c = std::is_constant_evaluated(); // expected-warning {{always evaluate to true}}
constinit bool d = std::is_constant_evaluated(); // expected-warning {{always evaluate to true}}
int p = __builtin_is_constant_evaluated();
const int q = __builtin_is_constant_evaluated();

template <bool c = std::is_constant_evaluated()> // expected-warning {{always evaluate to true}}
void vvv() {
  return;
}

template<> void vvv<true>() {}
template<> void vvv<false>() {}

template<typename T> concept C = __builtin_is_constant_evaluated();// expected-warning {{always evaluate to true}}

struct Foo {
  static constexpr bool ce = std::is_constant_evaluated(); // expected-warning {{always evaluate to true}}
  const static bool nonce = std::is_constant_evaluated();
  bool b = std::is_constant_evaluated();

  Foo() {
    if constexpr(std::is_constant_evaluated()) {} // expected-warning {{always evaluate to true}}
    bool aa = std::is_constant_evaluated(); // expected-warning {{always evaluate to false}}
    static bool bb = std::is_constant_evaluated();
    constexpr bool cc = std::is_constant_evaluated(); // expected-warning {{always evaluate to true}}
    if consteval {} // expected-warning {{always false}}
  }
  constexpr Foo(int) {
    if constexpr(std::is_constant_evaluated()) {} // expected-warning {{always evaluate to true}}
    bool aa = std::is_constant_evaluated();
    static bool bb = std::is_constant_evaluated();
    constexpr bool cc = std::is_constant_evaluated(); // expected-warning {{always evaluate to true}}
  }
  consteval Foo(int *) {
    if constexpr(std::is_constant_evaluated()) {} // expected-warning {{always evaluate to true}}
    bool aa = std::is_constant_evaluated(); // expected-warning {{always evaluate to true}}
    static bool bb = std::is_constant_evaluated(); // expected-warning {{always evaluate to true}}
    constexpr bool cc = std::is_constant_evaluated(); // expected-warning {{always evaluate to true}}
  }
};
