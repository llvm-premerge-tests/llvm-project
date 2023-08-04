// RUN: %clang_cc1 -fsyntax-only -verify %s


template<typename T>
constexpr bool isNan(T V) {
  return __builtin_isnan(V);
}

constexpr double Signaling = __builtin_nans("");
static_assert(isNan(Signaling), "");
constexpr double Quiet = __builtin_nan("");
static_assert(isNan(Quiet), "");


static_assert(Signaling + 1 == 0, ""); // expected-error {{not an integral constant expression}} \
                                       // expected-note {{NaN input to a floating point operation}}
static_assert(Signaling - 1 == 0, ""); // expected-error {{not an integral constant expression}} \
                                       // expected-note {{NaN input to a floating point operation}}
static_assert(Signaling / 1 == 0, ""); // expected-error {{not an integral constant expression}} \
                                       // expected-note {{NaN input to a floating point operation}}
static_assert(Signaling * 1 == 0, ""); // expected-error {{not an integral constant expression}} \
                                       // expected-note {{NaN input to a floating point operation}}

static_assert(Quiet + 1 != 0, "");
static_assert(isNan(Quiet + 1), "");
static_assert(-Signaling != 0, "");
static_assert(+Signaling != 0, "");
