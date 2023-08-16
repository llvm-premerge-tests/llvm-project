// RUN: %clang_cc1 -std=c++14 -triple x86_64-linux-gnu -Wno-unused-value %s

using FourCharsVecSize __attribute__((vector_size(4))) = char;
using FourIntsVecSize __attribute__((vector_size(16))) = int;
using FourLongLongsVecSize __attribute__((vector_size(32))) = long long;
using FourFloatsVecSize __attribute__((vector_size(16))) = float;
using FourDoublesVecSize __attribute__((vector_size(32))) = double;
using FourI128VecSize __attribute__((vector_size(64))) = __int128;

using FourCharsExtVec __attribute__((ext_vector_type(4))) = char;
using FourIntsExtVec __attribute__((ext_vector_type(4))) = int;
using FourLongLongsExtVec __attribute__((ext_vector_type(4))) = long long;
using FourFloatsExtVec __attribute__((ext_vector_type(4))) = float;
using FourDoublesExtVec __attribute__((ext_vector_type(4))) = double;
using FourI128ExtVec __attribute__((ext_vector_type(4))) = __int128;


// Next a series of tests to make sure these operations are usable in
// constexpr functions. Template instantiations don't emit Winvalid-constexpr,
// so we have to do these as macros.
#define MathShiftOps(Type)                            \
  constexpr auto MathShiftOps##Type(Type a, Type b) { \
    a = a + b;                                        \
    a = a - b;                                        \
    a = a * b;                                        \
    a = a / b;                                        \
    b = a + 1;                                        \
    b = a - 1;                                        \
    b = a * 1;                                        \
    b = a / 1;                                        \
    a += a;                                           \
    a -= a;                                           \
    a *= a;                                           \
    a /= a;                                           \
    b += a;                                           \
    b -= a;                                           \
    b *= a;                                           \
    b /= a;                                           \
    a < b;                                            \
    a > b;                                            \
    a <= b;                                           \
    a >= b;                                           \
    a == b;                                           \
    a != b;                                           \
    a &&b;                                            \
    a || b;                                           \
    a += a[1];                                        \
    a[1] += 1;                                        \
    a[1] = 1;                                         \
    a[1] = b[1];                                      \
    a[1]++;                                           \
    ++a[1];                                           \
    auto c = (a, b);                                  \
    return c;                                         \
  }

// Ops specific to Integers.
#define MathShiftOpsInts(Type)                            \
  constexpr auto MathShiftopsInts##Type(Type a, Type b) { \
    a = a << b;                                           \
    a = a >> b;                                           \
    a = a << 3;                                           \
    a = a >> 3;                                           \
    a = 3 << b;                                           \
    a = 3 >> b;                                           \
    a <<= b;                                              \
    a >>= b;                                              \
    a <<= 3;                                              \
    a >>= 3;                                              \
    a = a % b;                                            \
    a &b;                                                 \
    a | b;                                                \
    a ^ b;                                                \
    return a;                                             \
  }

MathShiftOps(FourCharsVecSize);
MathShiftOps(FourIntsVecSize);
MathShiftOps(FourLongLongsVecSize);
MathShiftOps(FourFloatsVecSize);
MathShiftOps(FourDoublesVecSize);
MathShiftOps(FourCharsExtVec);
MathShiftOps(FourIntsExtVec);
MathShiftOps(FourLongLongsExtVec);
MathShiftOps(FourFloatsExtVec);
MathShiftOps(FourDoublesExtVec);

MathShiftOpsInts(FourCharsVecSize);
MathShiftOpsInts(FourIntsVecSize);
MathShiftOpsInts(FourLongLongsVecSize);
MathShiftOpsInts(FourCharsExtVec);
MathShiftOpsInts(FourIntsExtVec);
MathShiftOpsInts(FourLongLongsExtVec);

template<typename T, typename U>
constexpr bool VectorsEqual(T a, U b) {
  for (unsigned I = 0; I < 4; ++I) {
    if (a[I] != b[I])
      return false;
  }
  return true;
}

template <typename T, typename U>
constexpr auto CmpMul(T t, U u) {
  t *= u;
  return t;
}
template <typename T, typename U>
constexpr auto CmpDiv(T t, U u) {
  t /= u;
  return t;
}
template <typename T, typename U>
constexpr auto CmpRem(T t, U u) {
  t %= u;
  return t;
}

template <typename T, typename U>
constexpr auto CmpAdd(T t, U u) {
  t += u;
  return t;
}

template <typename T, typename U>
constexpr auto CmpSub(T t, U u) {
  t -= u;
  return t;
}

template <typename T, typename U>
constexpr auto CmpLSH(T t, U u) {
  t <<= u;
  return t;
}

template <typename T, typename U>
constexpr auto CmpRSH(T t, U u) {
  t >>= u;
  return t;
}

template <typename T, typename U>
constexpr auto CmpBinAnd(T t, U u) {
  t &= u;
  return t;
}

template <typename T, typename U>
constexpr auto CmpBinXOr(T t, U u) {
  t ^= u;
  return t;
}

template <typename T, typename U>
constexpr auto CmpBinOr(T t, U u) {
  t |= u;
  return t;
}

template <typename T, typename U>
constexpr auto UpdateElementsInPlace(T t, U u) {
  t[0] += u[0];
  t[1]++;
  t[2] = 1;
  return t;
}

// Only int vs float makes a difference here, so we only need to test 1 of each.
// Test Char to make sure the mixed-nature of shifts around char is evident.
void CharUsage() {
  constexpr auto a = FourCharsVecSize{6, 3, 2, 1} +
                     FourCharsVecSize{12, 15, 5, 7};
  static_assert(VectorsEqual(a, FourCharsVecSize{18, 18, 7, 8}), "");

  constexpr auto b = FourCharsVecSize{19, 15, 13, 12} -
                     FourCharsVecSize{13, 14, 5, 3};
  static_assert(VectorsEqual(b, FourCharsVecSize{6, 1, 8, 9}), "");

  constexpr auto c = FourCharsVecSize{8, 4, 2, 1} *
                     FourCharsVecSize{3, 4, 5, 6};
  static_assert(VectorsEqual(c, FourCharsVecSize{24, 16, 10, 6}), "");

  constexpr auto d = FourCharsVecSize{12, 12, 10, 10} /
                     FourCharsVecSize{6, 4, 5, 2};
  static_assert(VectorsEqual(d, FourCharsVecSize{2, 3, 2, 5}), "");

  constexpr auto e = FourCharsVecSize{12, 12, 10, 10} %
                     FourCharsVecSize{6, 4, 4, 3};
  static_assert(VectorsEqual(e, FourCharsVecSize{0, 0, 2, 1}), "");

  constexpr auto f = FourCharsVecSize{6, 3, 2, 1} + 3;
  static_assert(VectorsEqual(f, FourCharsVecSize{9, 6, 5, 4}), "");

  constexpr auto g = FourCharsVecSize{19, 15, 12, 10} - 3;
  static_assert(VectorsEqual(g, FourCharsVecSize{16, 12, 9, 7}), "");

  constexpr auto h = FourCharsVecSize{8, 4, 2, 1} * 3;
  static_assert(VectorsEqual(h, FourCharsVecSize{24, 12, 6, 3}), "");

  constexpr auto j = FourCharsVecSize{12, 15, 18, 21} / 3;
  static_assert(VectorsEqual(j, FourCharsVecSize{4, 5, 6, 7}), "");

  constexpr auto k = FourCharsVecSize{12, 17, 19, 22} % 3;
  static_assert(VectorsEqual(k, FourCharsVecSize{0, 2, 1, 1}), "");

  constexpr auto l = 3 + FourCharsVecSize{6, 3, 2, 1};
  static_assert(VectorsEqual(l, FourCharsVecSize{9, 6, 5, 4}), "");

  constexpr auto m = 20 - FourCharsVecSize{19, 15, 12, 10};
  static_assert(VectorsEqual(m, FourCharsVecSize{1, 5, 8, 10}), "");

  constexpr auto n = 3 * FourCharsVecSize{8, 4, 2, 1};
  static_assert(VectorsEqual(n, FourCharsVecSize{24, 12, 6, 3}), "");

  constexpr auto o = 100 / FourCharsVecSize{12, 15, 18, 21};
  static_assert(VectorsEqual(o, FourCharsVecSize{8, 6, 5, 4}), "");

  constexpr auto p = 100 % FourCharsVecSize{12, 15, 18, 21};
  static_assert(VectorsEqual(p, FourCharsVecSize{4, 10, 10, 16}), "");

  constexpr auto q = FourCharsVecSize{6, 3, 2, 1} << FourCharsVecSize{1, 1, 2, 2};
  static_assert(VectorsEqual(q, FourCharsVecSize{12, 6, 8, 4}), "");

  constexpr auto r = FourCharsVecSize{19, 15, 12, 10} >>
                     FourCharsVecSize{1, 1, 2, 2};
  static_assert(VectorsEqual(r, FourCharsVecSize{9, 7, 3, 2}), "");

  constexpr auto s = FourCharsVecSize{6, 3, 5, 10} << 1;
  static_assert(VectorsEqual(s, FourCharsVecSize{12, 6, 10, 20}), "");

  constexpr auto t = FourCharsVecSize{19, 15, 10, 20} >> 1;
  static_assert(VectorsEqual(t, FourCharsVecSize{9, 7, 5, 10}), "");

  //TODO: These are being auto-deduced as FourCharsExtVec instead of FourCharsVecSize
  constexpr FourCharsVecSize u = 12 << FourCharsVecSize{1, 2, 3, 3};
  static_assert(VectorsEqual(u, FourCharsVecSize{24, 48, 96, 96}), "");

  constexpr FourCharsVecSize v = 12 >> FourCharsVecSize{1, 2, 2, 1};
  static_assert(VectorsEqual(v, FourCharsVecSize{6, 3, 3, 6}), "");

  constexpr auto w = FourCharsVecSize{1, 2, 3, 4} <
                     FourCharsVecSize{4, 3, 2, 1};
  static_assert(VectorsEqual(w, FourCharsVecSize{-1, -1, 0, 0}), "");

  constexpr auto x = FourCharsVecSize{1, 2, 3, 4} >
                     FourCharsVecSize{4, 3, 2, 1};
  static_assert(VectorsEqual(x, FourCharsVecSize{0, 0, -1, -1}), "");

  constexpr auto y = FourCharsVecSize{1, 2, 3, 4} <=
                     FourCharsVecSize{4, 3, 3, 1};
  static_assert(VectorsEqual(y, FourCharsVecSize{-1, -1, -1, 0}), "");

  constexpr auto z = FourCharsVecSize{1, 2, 3, 4} >=
                     FourCharsVecSize{4, 3, 3, 1};
  static_assert(VectorsEqual(z, FourCharsVecSize{0, 0, -1, -1}), "");

  constexpr auto A = FourCharsVecSize{1, 2, 3, 4} ==
                     FourCharsVecSize{4, 3, 3, 1};
  static_assert(VectorsEqual(A, FourCharsVecSize{0, 0, -1, 0}), "");

  constexpr auto B = FourCharsVecSize{1, 2, 3, 4} !=
                     FourCharsVecSize{4, 3, 3, 1};
  static_assert(VectorsEqual(B, FourCharsVecSize{-1, -1, 0, -1}), "");

  constexpr auto C = FourCharsVecSize{1, 2, 3, 4} < 3;
  static_assert(VectorsEqual(C, FourCharsVecSize{-1, -1, 0, 0}), "");

  constexpr auto D = FourCharsVecSize{1, 2, 3, 4} > 3;
  static_assert(VectorsEqual(D, FourCharsVecSize{0, 0, 0, -1}), "");

  constexpr auto E = FourCharsVecSize{1, 2, 3, 4} <= 3;
  static_assert(VectorsEqual(E, FourCharsVecSize{-1, -1, -1, 0}), "");

  constexpr auto F = FourCharsVecSize{1, 2, 3, 4} >= 3;
  static_assert(VectorsEqual(F, FourCharsVecSize{0, 0, -1, -1}), "");

  constexpr auto G = FourCharsVecSize{1, 2, 3, 4} == 3;
  static_assert(VectorsEqual(G, FourCharsVecSize{0, 0, -1, 0}), "");

  constexpr auto H = FourCharsVecSize{1, 2, 3, 4} != 3;
  static_assert(VectorsEqual(H, FourCharsVecSize{-1, -1, 0, -1}), "");

  constexpr auto I = FourCharsVecSize{1, 2, 3, 4} &
                     FourCharsVecSize{4, 3, 2, 1};
  static_assert(VectorsEqual(I, FourCharsVecSize{0, 2, 2, 0}), "");

  constexpr auto J = FourCharsVecSize{1, 2, 3, 4} ^
                     FourCharsVecSize { 4, 3, 2, 1 };
  static_assert(VectorsEqual(J, FourCharsVecSize{5, 1, 1, 5}), "");

  constexpr auto K = FourCharsVecSize{1, 2, 3, 4} |
                     FourCharsVecSize{4, 3, 2, 1};
  static_assert(VectorsEqual(K, FourCharsVecSize{5, 3, 3, 5}), "");

  constexpr auto L = FourCharsVecSize{1, 2, 3, 4} & 3;
  static_assert(VectorsEqual(L, FourCharsVecSize{1, 2, 3, 0}), "");

  constexpr auto M = FourCharsVecSize{1, 2, 3, 4} ^ 3;
  static_assert(VectorsEqual(M, FourCharsVecSize{2, 1, 0, 7}), "");

  constexpr auto N = FourCharsVecSize{1, 2, 3, 4} | 3;
  static_assert(VectorsEqual(N, FourCharsVecSize{3, 3, 3, 7}), "");

  constexpr auto O = FourCharsVecSize{5, 0, 6, 0} &&
                     FourCharsVecSize{5, 5, 0, 0};
  static_assert(VectorsEqual(O, FourCharsVecSize{1, 0, 0, 0}), "");

  constexpr auto P = FourCharsVecSize{5, 0, 6, 0} ||
                     FourCharsVecSize{5, 5, 0, 0};
  static_assert(VectorsEqual(P, FourCharsVecSize{1, 1, 1, 0}), "");

  constexpr auto Q = FourCharsVecSize{5, 0, 6, 0} && 3;
  static_assert(VectorsEqual(Q, FourCharsVecSize{1, 0, 1, 0}), "");

  constexpr auto R = FourCharsVecSize{5, 0, 6, 0} || 3;
  static_assert(VectorsEqual(R, FourCharsVecSize{1, 1, 1, 1}), "");

  constexpr auto T = CmpMul(a, b);
  static_assert(VectorsEqual(T, FourCharsVecSize{108, 18, 56, 72}), "");

  constexpr auto U = CmpDiv(a, b);
  static_assert(VectorsEqual(U, FourCharsVecSize{3, 18, 0, 0}), "");

  constexpr auto V = CmpRem(a, b);
  static_assert(VectorsEqual(V, FourCharsVecSize{0, 0, 7, 8}), "");

  constexpr auto X = CmpAdd(a, b);
  static_assert(VectorsEqual(X, FourCharsVecSize{24, 19, 15, 17}), "");

  constexpr auto Y = CmpSub(a, b);
  static_assert(VectorsEqual(Y, FourCharsVecSize{12, 17, -1, -1}), "");

  constexpr auto InvH = -H;
  static_assert(VectorsEqual(InvH, FourCharsVecSize{1, 1, 0, 1}), "");

  constexpr auto Z = CmpLSH(a, InvH);
  static_assert(VectorsEqual(Z, FourCharsVecSize{36, 36, 7, 16}), "");

  constexpr auto aa = CmpRSH(a, InvH);
  static_assert(VectorsEqual(aa, FourCharsVecSize{9, 9, 7, 4}), "");

  constexpr auto ab = CmpBinAnd(a, b);
  static_assert(VectorsEqual(ab, FourCharsVecSize{2, 0, 0, 8}), "");

  constexpr auto ac = CmpBinXOr(a, b);
  static_assert(VectorsEqual(ac, FourCharsVecSize{20, 19, 15, 1}), "");

  constexpr auto ad = CmpBinOr(a, b);
  static_assert(VectorsEqual(ad, FourCharsVecSize{22, 19, 15, 9}), "");

  constexpr auto ae = ~FourCharsVecSize{1, 2, 10, 20};
  static_assert(VectorsEqual(ae, FourCharsVecSize{-2, -3, -11, -21}), "");

  constexpr auto af = !FourCharsVecSize{0, 1, 8, -1};
  static_assert(VectorsEqual(af, FourCharsVecSize{-1, 0, 0, 0}), "");

  constexpr auto ag = UpdateElementsInPlace(
                    FourCharsVecSize{3, 3, 0, 0}, FourCharsVecSize{3, 0, 0, 0});
  static_assert(VectorsEqual(ag, FourCharsVecSize{6, 4, 1, 0}), "");
}

void CharExtVecUsage() {
  constexpr auto a = FourCharsExtVec{6, 3, 2, 1} +
                     FourCharsExtVec{12, 15, 5, 7};
  static_assert(VectorsEqual(a, FourCharsExtVec{18, 18, 7, 8}), "");

  constexpr auto b = FourCharsExtVec{19, 15, 13, 12} -
                     FourCharsExtVec{13, 14, 5, 3};
  static_assert(VectorsEqual(b, FourCharsExtVec{6, 1, 8, 9}), "");

  constexpr auto c = FourCharsExtVec{8, 4, 2, 1} *
                     FourCharsExtVec{3, 4, 5, 6};
  static_assert(VectorsEqual(c, FourCharsExtVec{24, 16, 10, 6}), "");

  constexpr auto d = FourCharsExtVec{12, 12, 10, 10} /
                     FourCharsExtVec{6, 4, 5, 2};
  static_assert(VectorsEqual(d, FourCharsExtVec{2, 3, 2, 5}), "");

  constexpr auto e = FourCharsExtVec{12, 12, 10, 10} %
                     FourCharsExtVec{6, 4, 4, 3};
  static_assert(VectorsEqual(e, FourCharsExtVec{0, 0, 2, 1}), "");

  constexpr auto f = FourCharsExtVec{6, 3, 2, 1} + 3;
  static_assert(VectorsEqual(f, FourCharsExtVec{9, 6, 5, 4}), "");

  constexpr auto g = FourCharsExtVec{19, 15, 12, 10} - 3;
  static_assert(VectorsEqual(g, FourCharsExtVec{16, 12, 9, 7}), "");

  constexpr auto h = FourCharsExtVec{8, 4, 2, 1} * 3;
  static_assert(VectorsEqual(h, FourCharsExtVec{24, 12, 6, 3}), "");

  constexpr auto j = FourCharsExtVec{12, 15, 18, 21} / 3;
  static_assert(VectorsEqual(j, FourCharsExtVec{4, 5, 6, 7}), "");

  constexpr auto k = FourCharsExtVec{12, 17, 19, 22} % 3;
  static_assert(VectorsEqual(k, FourCharsExtVec{0, 2, 1, 1}), "");

  constexpr auto l = 3 + FourCharsExtVec{6, 3, 2, 1};
  static_assert(VectorsEqual(l, FourCharsExtVec{9, 6, 5, 4}), "");

  constexpr auto m = 20 - FourCharsExtVec{19, 15, 12, 10};
  static_assert(VectorsEqual(m, FourCharsExtVec{1, 5, 8, 10}), "");

  constexpr auto n = 3 * FourCharsExtVec{8, 4, 2, 1};
  static_assert(VectorsEqual(n, FourCharsExtVec{24, 12, 6, 3}), "");

  constexpr auto o = 100 / FourCharsExtVec{12, 15, 18, 21};
  static_assert(VectorsEqual(o, FourCharsExtVec{8, 6, 5, 4}), "");

  constexpr auto p = 100 % FourCharsExtVec{12, 15, 18, 21};
  static_assert(VectorsEqual(p, FourCharsExtVec{4, 10, 10, 16}), "");

  constexpr auto q = FourCharsExtVec{6, 3, 2, 1} << FourCharsExtVec{1, 1, 2, 2};
  static_assert(VectorsEqual(q, FourCharsExtVec{12, 6, 8, 4}), "");

  constexpr auto r = FourCharsExtVec{19, 15, 12, 10} >>
                     FourCharsExtVec{1, 1, 2, 2};
  static_assert(VectorsEqual(r, FourCharsExtVec{9, 7, 3, 2}), "");

  constexpr auto s = FourCharsExtVec{6, 3, 5, 10} << 1;
  static_assert(VectorsEqual(s, FourCharsExtVec{12, 6, 10, 20}), "");

  constexpr auto t = FourCharsExtVec{19, 15, 10, 20} >> 1;
  static_assert(VectorsEqual(t, FourCharsExtVec{9, 7, 5, 10}), "");

  constexpr auto u = 12 << FourCharsExtVec{1, 2, 3, 3};
  static_assert(VectorsEqual(u, FourCharsExtVec{24, 48, 96, 96}), "");

  constexpr auto v = 12 >> FourCharsExtVec{1, 2, 2, 1};
  static_assert(VectorsEqual(v, FourCharsExtVec{6, 3, 3, 6}), "");

  constexpr auto w = FourCharsExtVec{1, 2, 3, 4} <
                     FourCharsExtVec{4, 3, 2, 1};
  static_assert(VectorsEqual(w, FourCharsExtVec{-1, -1, 0, 0}), "");

  constexpr auto x = FourCharsExtVec{1, 2, 3, 4} >
                     FourCharsExtVec{4, 3, 2, 1};
  static_assert(VectorsEqual(x, FourCharsExtVec{0, 0, -1, -1}), "");

  constexpr auto y = FourCharsExtVec{1, 2, 3, 4} <=
                     FourCharsExtVec{4, 3, 3, 1};
  static_assert(VectorsEqual(y, FourCharsExtVec{-1, -1, -1, 0}), "");

  constexpr auto z = FourCharsExtVec{1, 2, 3, 4} >=
                     FourCharsExtVec{4, 3, 3, 1};
  static_assert(VectorsEqual(z, FourCharsExtVec{0, 0, -1, -1}), "");

  constexpr auto A = FourCharsExtVec{1, 2, 3, 4} ==
                     FourCharsExtVec{4, 3, 3, 1};
  static_assert(VectorsEqual(A, FourCharsExtVec{0, 0, -1, 0}), "");

  constexpr auto B = FourCharsExtVec{1, 2, 3, 4} !=
                     FourCharsExtVec{4, 3, 3, 1};
  static_assert(VectorsEqual(B, FourCharsExtVec{-1, -1, 0, -1}), "");

  constexpr auto C = FourCharsExtVec{1, 2, 3, 4} < 3;
  static_assert(VectorsEqual(C, FourCharsExtVec{-1, -1, 0, 0}), "");

  constexpr auto D = FourCharsExtVec{1, 2, 3, 4} > 3;
  static_assert(VectorsEqual(D, FourCharsExtVec{0, 0, 0, -1}), "");

  constexpr auto E = FourCharsExtVec{1, 2, 3, 4} <= 3;
  static_assert(VectorsEqual(E, FourCharsExtVec{-1, -1, -1, 0}), "");

  constexpr auto F = FourCharsExtVec{1, 2, 3, 4} >= 3;
  static_assert(VectorsEqual(F, FourCharsExtVec{0, 0, -1, -1}), "");

  constexpr auto G = FourCharsExtVec{1, 2, 3, 4} == 3;
  static_assert(VectorsEqual(G, FourCharsExtVec{0, 0, -1, 0}), "");

  constexpr auto H = FourCharsExtVec{1, 2, 3, 4} != 3;
  static_assert(VectorsEqual(H, FourCharsExtVec{-1, -1, 0, -1}), "");

  constexpr auto I = FourCharsExtVec{1, 2, 3, 4} &
                     FourCharsExtVec{4, 3, 2, 1};
  static_assert(VectorsEqual(I, FourCharsExtVec{0, 2, 2, 0}), "");

  constexpr auto J = FourCharsExtVec{1, 2, 3, 4} ^
                     FourCharsExtVec { 4, 3, 2, 1 };
  static_assert(VectorsEqual(J, FourCharsExtVec{5, 1, 1, 5}), "");

  constexpr auto K = FourCharsExtVec{1, 2, 3, 4} |
                     FourCharsExtVec{4, 3, 2, 1};
  static_assert(VectorsEqual(K, FourCharsExtVec{5, 3, 3, 5}), "");

  constexpr auto L = FourCharsExtVec{1, 2, 3, 4} & 3;
  static_assert(VectorsEqual(L, FourCharsExtVec{1, 2, 3, 0}), "");

  constexpr auto M = FourCharsExtVec{1, 2, 3, 4} ^ 3;
  static_assert(VectorsEqual(M, FourCharsExtVec{2, 1, 0, 7}), "");

  constexpr auto N = FourCharsExtVec{1, 2, 3, 4} | 3;
  static_assert(VectorsEqual(N, FourCharsExtVec{3, 3, 3, 7}), "");

  constexpr auto O = FourCharsExtVec{5, 0, 6, 0} &&
                     FourCharsExtVec{5, 5, 0, 0};
  static_assert(VectorsEqual(O, FourCharsExtVec{1, 0, 0, 0}), "");

  constexpr auto P = FourCharsExtVec{5, 0, 6, 0} ||
                     FourCharsExtVec{5, 5, 0, 0};
  static_assert(VectorsEqual(P, FourCharsExtVec{1, 1, 1, 0}), "");

  constexpr auto Q = FourCharsExtVec{5, 0, 6, 0} && 3;
  static_assert(VectorsEqual(Q, FourCharsExtVec{1, 0, 1, 0}), "");

  constexpr auto R = FourCharsExtVec{5, 0, 6, 0} || 3;
  static_assert(VectorsEqual(R, FourCharsExtVec{1, 1, 1, 1}), "");

  constexpr auto T = CmpMul(a, b);
  static_assert(VectorsEqual(T, FourCharsExtVec{108, 18, 56, 72}), "");

  constexpr auto U = CmpDiv(a, b);
  static_assert(VectorsEqual(U, FourCharsExtVec{3, 18, 0, 0}), "");

  constexpr auto V = CmpRem(a, b);
  static_assert(VectorsEqual(V, FourCharsExtVec{0, 0, 7, 8}), "");

  constexpr auto X = CmpAdd(a, b);
  static_assert(VectorsEqual(X, FourCharsExtVec{24, 19, 15, 17}), "");

  constexpr auto Y = CmpSub(a, b);
  static_assert(VectorsEqual(Y, FourCharsExtVec{12, 17, -1, -1}), "");

  constexpr auto InvH = -H;
  static_assert(VectorsEqual(InvH, FourCharsExtVec{1, 1, 0, 1}), "");

  constexpr auto Z = CmpLSH(a, InvH);
  static_assert(VectorsEqual(Z, FourCharsExtVec{36, 36, 7, 16}), "");

  constexpr auto aa = CmpRSH(a, InvH);
  static_assert(VectorsEqual(aa, FourCharsExtVec{9, 9, 7, 4}), "");

  constexpr auto ab = CmpBinAnd(a, b);
  static_assert(VectorsEqual(ab, FourCharsExtVec{2, 0, 0, 8}), "");

  constexpr auto ac = CmpBinXOr(a, b);
  static_assert(VectorsEqual(ac, FourCharsExtVec{20, 19, 15, 1}), "");

  constexpr auto ad = CmpBinOr(a, b);
  static_assert(VectorsEqual(ad, FourCharsExtVec{22, 19, 15, 9}), "");

  constexpr auto ae = ~FourCharsExtVec{1, 2, 10, 20};
  static_assert(VectorsEqual(ae, FourCharsExtVec{-2, -3, -11, -21}), "");

  constexpr auto af = !FourCharsExtVec{0, 1, 8, -1};
  static_assert(VectorsEqual(af, FourCharsExtVec{-1, 0, 0, 0}), "");

  constexpr auto ag = UpdateElementsInPlace(
                  FourCharsExtVec{3, 3, 0, 0}, FourCharsExtVec{3, 0, 0, 0});
  static_assert(VectorsEqual(ag, FourCharsExtVec{6, 4, 1, 0}), "");
}

void FloatUsage() {
  constexpr auto a = FourFloatsVecSize{6, 3, 2, 1} +
                     FourFloatsVecSize{12, 15, 5, 7};
  static_assert(VectorsEqual(a, FourFloatsVecSize{1.800000e+01, 1.800000e+01, 7.000000e+00, 8.000000e+00}), "");

  constexpr auto b = FourFloatsVecSize{19, 15, 13, 12} -
                     FourFloatsVecSize{13, 14, 5, 3};
  static_assert(VectorsEqual(b, FourFloatsVecSize{6.000000e+00, 1.000000e+00, 8.000000e+00, 9.000000e+00}), "");

  constexpr auto c = FourFloatsVecSize{8, 4, 2, 1} *
                     FourFloatsVecSize{3, 4, 5, 6};
  static_assert(VectorsEqual(c, FourFloatsVecSize{2.400000e+01, 1.600000e+01, 1.000000e+01, 6.000000e+00}), "");

  constexpr auto d = FourFloatsVecSize{12, 12, 10, 10} /
                     FourFloatsVecSize{6, 4, 5, 2};
  static_assert(VectorsEqual(d, FourFloatsVecSize{2.000000e+00, 3.000000e+00, 2.000000e+00, 5.000000e+00}), "");

  constexpr auto f = FourFloatsVecSize{6, 3, 2, 1} + 3;
  static_assert(VectorsEqual(f, FourFloatsVecSize{9.000000e+00, 6.000000e+00, 5.000000e+00, 4.000000e+00}), "");

  constexpr auto g = FourFloatsVecSize{19, 15, 12, 10} - 3;
  static_assert(VectorsEqual(g, FourFloatsVecSize{1.600000e+01, 1.200000e+01, 9.000000e+00, 7.000000e+00}), "");

  constexpr auto h = FourFloatsVecSize{8, 4, 2, 1} * 3;
  static_assert(VectorsEqual(h, FourFloatsVecSize{2.400000e+01, 1.200000e+01, 6.000000e+00, 3.000000e+00}), "");

  constexpr auto j = FourFloatsVecSize{12, 15, 18, 21} / 3;
  static_assert(VectorsEqual(j, FourFloatsVecSize{4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00}), "");

  constexpr auto l = 3 + FourFloatsVecSize{6, 3, 2, 1};
  static_assert(VectorsEqual(l, FourFloatsVecSize{9.000000e+00, 6.000000e+00, 5.000000e+00, 4.000000e+00}), "");

  constexpr auto m = 20 - FourFloatsVecSize{19, 15, 12, 10};
  static_assert(VectorsEqual(m, FourFloatsVecSize{1.000000e+00, 5.000000e+00, 8.000000e+00, 1.000000e+01}), "");

  constexpr auto n = 3 * FourFloatsVecSize{8, 4, 2, 1};
  static_assert(VectorsEqual(n, FourFloatsVecSize{2.400000e+01, 1.200000e+01, 6.000000e+00, 3.000000e+00}), "");

  constexpr auto o = 100 / FourFloatsVecSize{12, 15, 18, 21};
  static_assert(VectorsEqual(o, FourFloatsVecSize{100.0f / 12.0f, 100.0f / 15.0f, 100.0f / 18.0f, 100.0f / 21.0f}), "");

  constexpr auto w = FourFloatsVecSize{1, 2, 3, 4} <
                     FourFloatsVecSize{4, 3, 2, 1};
  static_assert(VectorsEqual(w, FourIntsVecSize{-1, -1, 0, 0}), "");

  constexpr auto x = FourFloatsVecSize{1, 2, 3, 4} >
                     FourFloatsVecSize{4, 3, 2, 1};
  static_assert(VectorsEqual(x, FourIntsVecSize{0, 0, -1, -1}), "");

  constexpr auto y = FourFloatsVecSize{1, 2, 3, 4} <=
                     FourFloatsVecSize{4, 3, 3, 1};
  static_assert(VectorsEqual(y, FourIntsVecSize{-1, -1, -1, 0}), "");

  constexpr auto z = FourFloatsVecSize{1, 2, 3, 4} >=
                     FourFloatsVecSize{4, 3, 3, 1};
  static_assert(VectorsEqual(z, FourIntsVecSize{0, 0, -1, -1}), "");

  constexpr auto A = FourFloatsVecSize{1, 2, 3, 4} ==
                     FourFloatsVecSize{4, 3, 3, 1};
  static_assert(VectorsEqual(A, FourIntsVecSize{0, 0, -1, 0}), "");

  constexpr auto B = FourFloatsVecSize{1, 2, 3, 4} !=
                     FourFloatsVecSize{4, 3, 3, 1};
  static_assert(VectorsEqual(B, FourIntsVecSize{-1, -1, 0, -1}), "");

  constexpr auto C = FourFloatsVecSize{1, 2, 3, 4} < 3;
  static_assert(VectorsEqual(C, FourIntsVecSize{-1, -1, 0, 0}), "");

  constexpr auto D = FourFloatsVecSize{1, 2, 3, 4} > 3;
  static_assert(VectorsEqual(D, FourIntsVecSize{0, 0, 0, -1}), "");

  constexpr auto E = FourFloatsVecSize{1, 2, 3, 4} <= 3;
  static_assert(VectorsEqual(E, FourIntsVecSize{-1, -1, -1, 0}), "");

  constexpr auto F = FourFloatsVecSize{1, 2, 3, 4} >= 3;
  static_assert(VectorsEqual(F, FourIntsVecSize{0, 0, -1, -1}), "");

  constexpr auto G = FourFloatsVecSize{1, 2, 3, 4} == 3;
  static_assert(VectorsEqual(G, FourIntsVecSize{0, 0, -1, 0}), "");

  constexpr auto H = FourFloatsVecSize{1, 2, 3, 4} != 3;
  static_assert(VectorsEqual(H, FourIntsVecSize{-1, -1, 0, -1}), "");

  constexpr auto O = FourFloatsVecSize{5, 0, 6, 0} &&
                     FourFloatsVecSize{5, 5, 0, 0};
  static_assert(VectorsEqual(O, FourIntsVecSize{1, 0, 0, 0}), "");

  constexpr auto P = FourFloatsVecSize{5, 0, 6, 0} ||
                     FourFloatsVecSize{5, 5, 0, 0};
  static_assert(VectorsEqual(P, FourIntsVecSize{1, 1, 1, 0}), "");

  constexpr auto Q = FourFloatsVecSize{5, 0, 6, 0} && 3;
  static_assert(VectorsEqual(Q, FourIntsVecSize{1, 0, 1, 0}), "");

  constexpr auto R = FourFloatsVecSize{5, 0, 6, 0} || 3;
  static_assert(VectorsEqual(R, FourIntsVecSize{1, 1, 1, 1}), "");

  constexpr auto T = CmpMul(a, b);
  static_assert(VectorsEqual(T, FourFloatsVecSize{1.080000e+02, 1.800000e+01, 5.600000e+01, 7.200000e+01}), "");

  constexpr auto U = CmpDiv(a, b);
  static_assert(VectorsEqual(U, FourFloatsVecSize{3.000000e+00, 1.800000e+01, 8.750000e-01, a[3] / b[3]}), "");

  constexpr auto X = CmpAdd(a, b);
  static_assert(VectorsEqual(X, FourFloatsVecSize{2.400000e+01, 1.900000e+01, 1.500000e+01, 1.700000e+01}), "");

  constexpr auto Y = CmpSub(a, b);
  static_assert(VectorsEqual(Y, FourFloatsVecSize{1.200000e+01, 1.700000e+01, -1.000000e+00, -1.000000e+00}), "");

  constexpr auto Z = -Y;
  static_assert(VectorsEqual(Z, FourFloatsVecSize{-1.200000e+01, -1.700000e+01, 1.000000e+00, 1.000000e+00}), "");

  // Operator ~ is illegal on floats, so no test for that.
  constexpr auto af = !FourFloatsVecSize{0, 1, 8, -1};
  static_assert(VectorsEqual(af, FourIntsVecSize{-1, 0, 0, 0}), "");

  constexpr auto ag = UpdateElementsInPlace(
                    FourFloatsVecSize{3, 3, 0, 0}, FourFloatsVecSize{3, 0, 0, 0});
  static_assert(VectorsEqual(ag, FourFloatsVecSize{6, 4, 1, 0}), "");
}

void FloatVecUsage() {
  constexpr auto a = FourFloatsExtVec{6, 3, 2, 1} +
                     FourFloatsExtVec{12, 15, 5, 7};
  // CHECK: <4 x float> <float 1.800000e+01, float 1.800000e+01, float 7.000000e+00, float 8.000000e+00>
  constexpr auto b = FourFloatsExtVec{19, 15, 13, 12} -
                     FourFloatsExtVec{13, 14, 5, 3};
  static_assert(VectorsEqual(b, FourFloatsExtVec{6.000000e+00, 1.000000e+00, 8.000000e+00, 9.000000e+00}), "");

  constexpr auto c = FourFloatsExtVec{8, 4, 2, 1} *
                     FourFloatsExtVec{3, 4, 5, 6};
  static_assert(VectorsEqual(c, FourFloatsExtVec{2.400000e+01, 1.600000e+01, 1.000000e+01, 6.000000e+00}), "");

  constexpr auto d = FourFloatsExtVec{12, 12, 10, 10} /
                     FourFloatsExtVec{6, 4, 5, 2};
  static_assert(VectorsEqual(d, FourFloatsExtVec{2.000000e+00, 3.000000e+00, 2.000000e+00, 5.000000e+00}), "");

  constexpr auto f = FourFloatsExtVec{6, 3, 2, 1} + 3;
  static_assert(VectorsEqual(f, FourFloatsExtVec{9.000000e+00, 6.000000e+00, 5.000000e+00, 4.000000e+00}), "");

  constexpr auto g = FourFloatsExtVec{19, 15, 12, 10} - 3;
  static_assert(VectorsEqual(g, FourFloatsExtVec{1.600000e+01, 1.200000e+01, 9.000000e+00, 7.000000e+00}), "");

  constexpr auto h = FourFloatsExtVec{8, 4, 2, 1} * 3;
  static_assert(VectorsEqual(h, FourFloatsExtVec{2.400000e+01, 1.200000e+01, 6.000000e+00, 3.000000e+00}), "");

  constexpr auto j = FourFloatsExtVec{12, 15, 18, 21} / 3;
  static_assert(VectorsEqual(j, FourFloatsExtVec{4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00}), "");

  constexpr auto l = 3 + FourFloatsExtVec{6, 3, 2, 1};
  static_assert(VectorsEqual(l, FourFloatsExtVec{9.000000e+00, 6.000000e+00, 5.000000e+00, 4.000000e+00}), "");

  constexpr auto m = 20 - FourFloatsExtVec{19, 15, 12, 10};
  static_assert(VectorsEqual(m, FourFloatsExtVec{1.000000e+00, 5.000000e+00, 8.000000e+00, 1.000000e+01}), "");

  constexpr auto n = 3 * FourFloatsExtVec{8, 4, 2, 1};
  static_assert(VectorsEqual(n, FourFloatsExtVec{2.400000e+01, 1.200000e+01, 6.000000e+00, 3.000000e+00}), "");

  constexpr auto o = 100 / FourFloatsExtVec{12, 15, 18, 21};
  static_assert(VectorsEqual(o, FourFloatsExtVec{100.0f / 12.0f, 100.0f / 15.0f, 100.0f / 18.0f, 100.0f / 21.0f}), "");

  constexpr auto w = FourFloatsExtVec{1, 2, 3, 4} <
                     FourFloatsExtVec{4, 3, 2, 1};
  static_assert(VectorsEqual(w, FourIntsExtVec{-1, -1, 0, 0}), "");

  constexpr auto x = FourFloatsExtVec{1, 2, 3, 4} >
                     FourFloatsExtVec{4, 3, 2, 1};
  static_assert(VectorsEqual(x, FourIntsExtVec{0, 0, -1, -1}), "");

  constexpr auto y = FourFloatsExtVec{1, 2, 3, 4} <=
                     FourFloatsExtVec{4, 3, 3, 1};
  static_assert(VectorsEqual(y, FourIntsExtVec{-1, -1, -1, 0}), "");

  constexpr auto z = FourFloatsExtVec{1, 2, 3, 4} >=
                     FourFloatsExtVec{4, 3, 3, 1};
  static_assert(VectorsEqual(z, FourIntsExtVec{0, 0, -1, -1}), "");

  constexpr auto A = FourFloatsExtVec{1, 2, 3, 4} ==
                     FourFloatsExtVec{4, 3, 3, 1};
  static_assert(VectorsEqual(A, FourIntsExtVec{0, 0, -1, 0}), "");

  constexpr auto B = FourFloatsExtVec{1, 2, 3, 4} !=
                     FourFloatsExtVec{4, 3, 3, 1};
  static_assert(VectorsEqual(B, FourIntsExtVec{-1, -1, 0, -1}), "");

  constexpr auto C = FourFloatsExtVec{1, 2, 3, 4} < 3;
  static_assert(VectorsEqual(C, FourIntsExtVec{-1, -1, 0, 0}), "");

  constexpr auto D = FourFloatsExtVec{1, 2, 3, 4} > 3;
  static_assert(VectorsEqual(D, FourIntsExtVec{0, 0, 0, -1}), "");

  constexpr auto E = FourFloatsExtVec{1, 2, 3, 4} <= 3;
  static_assert(VectorsEqual(E, FourIntsExtVec{-1, -1, -1, 0}), "");

  constexpr auto F = FourFloatsExtVec{1, 2, 3, 4} >= 3;
  static_assert(VectorsEqual(F, FourIntsExtVec{0, 0, -1, -1}), "");

  constexpr auto G = FourFloatsExtVec{1, 2, 3, 4} == 3;
  static_assert(VectorsEqual(G, FourIntsExtVec{0, 0, -1, 0}), "");

  constexpr auto H = FourFloatsExtVec{1, 2, 3, 4} != 3;
  static_assert(VectorsEqual(H, FourIntsExtVec{-1, -1, 0, -1}), "");

  constexpr auto O = FourFloatsExtVec{5, 0, 6, 0} &&
                     FourFloatsExtVec{5, 5, 0, 0};
  static_assert(VectorsEqual(O, FourIntsExtVec{1, 0, 0, 0}), "");

  constexpr auto P = FourFloatsExtVec{5, 0, 6, 0} ||
                     FourFloatsExtVec{5, 5, 0, 0};
  static_assert(VectorsEqual(P, FourIntsExtVec{1, 1, 1, 0}), "");

  constexpr auto Q = FourFloatsExtVec{5, 0, 6, 0} && 3;
  static_assert(VectorsEqual(Q, FourIntsExtVec{1, 0, 1, 0}), "");

  constexpr auto R = FourFloatsExtVec{5, 0, 6, 0} || 3;
  static_assert(VectorsEqual(R, FourIntsExtVec{1, 1, 1, 1}), "");

  constexpr auto T = CmpMul(a, b);
  static_assert(VectorsEqual(T, FourFloatsExtVec{1.080000e+02, 1.800000e+01, 5.600000e+01, 7.200000e+01}), "");

  constexpr auto U = CmpDiv(a, b);
  static_assert(VectorsEqual(U, FourFloatsExtVec{3.000000e+00, 1.800000e+01, 8.750000e-01, a[3] / b[3]}), "");

  constexpr auto X = CmpAdd(a, b);
  static_assert(VectorsEqual(X, FourFloatsExtVec{2.400000e+01, 1.900000e+01, 1.500000e+01, 1.700000e+01}), "");

  constexpr auto Y = CmpSub(a, b);
  static_assert(VectorsEqual(Y, FourFloatsExtVec{1.200000e+01, 1.700000e+01, -1.000000e+00, -1.000000e+00}), "");

  constexpr auto Z = -Y;
  static_assert(VectorsEqual(Z, FourFloatsExtVec{-1.200000e+01, -1.700000e+01, 1.000000e+00, 1.000000e+00}), "");

  // Operator ~ is illegal on floats, so no test for that.
  constexpr auto af = !FourFloatsExtVec{0, 1, 8, -1};
  static_assert(VectorsEqual(af, FourIntsExtVec{-1, 0, 0, 0}), "");

  constexpr auto ag = UpdateElementsInPlace(
                    FourFloatsExtVec{3, 3, 0, 0}, FourFloatsExtVec{3, 0, 0, 0});
  static_assert(VectorsEqual(ag, FourFloatsExtVec{6, 4, 1, 0}), "");
}

void I128Usage() {
  constexpr auto a = FourI128VecSize{1, 2, 3, 4};
  static_assert(VectorsEqual(a, FourI128VecSize{1, 2, 3, 4}), "");

  constexpr auto b = a < 3;
  static_assert(VectorsEqual(b, FourI128VecSize{-1, -1, 0, 0}), "");

  // Operator ~ is illegal on floats, so no test for that.
  constexpr auto c = ~FourI128VecSize{1, 2, 10, 20};
  static_assert(VectorsEqual(c, FourI128VecSize{-2, -3, -11, -21}), "");

  constexpr auto d = !FourI128VecSize{0, 1, 8, -1};
  static_assert(VectorsEqual(d, FourI128VecSize{-1, 0, 0, 0}), "");
}

void I128VecUsage() {
  constexpr auto a = FourI128ExtVec{1, 2, 3, 4};
  static_assert(VectorsEqual(a, FourI128ExtVec{1, 2, 3, 4}), "");

  constexpr auto b = a < 3;
  static_assert(VectorsEqual(b, FourI128ExtVec{-1, -1, 0, 0}), "");

  // Operator ~ is illegal on floats, so no test for that.
  constexpr auto c = ~FourI128ExtVec{1, 2, 10, 20};
  static_assert(VectorsEqual(c, FourI128ExtVec{-2, -3, -11, -21}), "");

  constexpr auto d = !FourI128ExtVec{0, 1, 8, -1};
  static_assert(VectorsEqual(d, FourI128ExtVec{-1, 0, 0, 0}), "");
}

using FourBoolsExtVec __attribute__((ext_vector_type(4))) = bool;
void BoolVecUsage() {
  constexpr auto a = FourBoolsExtVec{true, false, true, false} <
                     FourBoolsExtVec{false, false, true, true};
  static_assert(VectorsEqual(a, FourBoolsExtVec{false, false, false, true}), "");

  constexpr auto b = FourBoolsExtVec{true, false, true, false} <=
                     FourBoolsExtVec{false, false, true, true};
  static_assert(VectorsEqual(b, FourBoolsExtVec{false, true, true, true}), "");

  constexpr auto c = FourBoolsExtVec{true, false, true, false} ==
                     FourBoolsExtVec{false, false, true, true};
  static_assert(VectorsEqual(c, FourBoolsExtVec{false, true, true, false}), "");

  constexpr auto d = FourBoolsExtVec{true, false, true, false} !=
                     FourBoolsExtVec{false, false, true, true};
  static_assert(VectorsEqual(d, FourBoolsExtVec{true, false, false, true}), "");

  constexpr auto e = FourBoolsExtVec{true, false, true, false} >=
                     FourBoolsExtVec{false, false, true, true};
  static_assert(VectorsEqual(e, FourBoolsExtVec{true, true, true, false}), "");

  constexpr auto f = FourBoolsExtVec{true, false, true, false} >
                     FourBoolsExtVec{false, false, true, true};
  static_assert(VectorsEqual(f, FourBoolsExtVec{true, false, false, false}), "");

  constexpr auto g = FourBoolsExtVec{true, false, true, false} &
                     FourBoolsExtVec{false, false, true, true};
  static_assert(VectorsEqual(g, FourBoolsExtVec{false, false, true, false}), "");

  constexpr auto h = FourBoolsExtVec{true, false, true, false} |
                     FourBoolsExtVec{false, false, true, true};
  static_assert(VectorsEqual(h, FourBoolsExtVec{true, false, true, true}), "");

  constexpr auto i = FourBoolsExtVec{true, false, true, false} ^
                     FourBoolsExtVec { false, false, true, true };
  static_assert(VectorsEqual(i, FourBoolsExtVec{true, false, false, true}), "");

  constexpr auto j = !FourBoolsExtVec{true, false, true, false};
  static_assert(VectorsEqual(j, FourBoolsExtVec{false, true, false, true}), "");

  constexpr auto k = ~FourBoolsExtVec{true, false, true, false};
  static_assert(VectorsEqual(k, FourBoolsExtVec{false, true, false, true}), "");
}
