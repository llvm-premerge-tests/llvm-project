// RUN: %clang_cc1 -triple x86_64-apple-darwin9 %s -std=c++20 \
// RUN:    -fsyntax-only -ast-dump | FileCheck %s

namespace std {
template <class Ret, typename... T>
struct coroutine_traits { using promise_type = typename Ret::promise_type; };

template <class Promise = void>
struct coroutine_handle {
  static coroutine_handle from_address(void *) noexcept;
  static coroutine_handle from_promise(Promise &promise);
  constexpr void* address() const noexcept;
};
template <>
struct coroutine_handle<void> {
  template <class PromiseType>
  coroutine_handle(coroutine_handle<PromiseType>) noexcept;
  static coroutine_handle from_address(void *);
  constexpr void* address() const noexcept;
};

struct suspend_always {
  bool await_ready() noexcept { return false; }
  void await_suspend(coroutine_handle<>) noexcept {}
  void await_resume() noexcept {}
};

struct suspend_never {
  bool await_ready() noexcept { return true; }
  void await_suspend(coroutine_handle<>) noexcept {}
  void await_resume() noexcept {}
};
} // namespace std

struct Chat {
  struct promise_type {
    std::suspend_always initial_suspend() { return {}; }
    Chat get_return_object() {
      return std::coroutine_handle<promise_type>::from_promise(*this);
    }
    std::suspend_always yield_value(int m) {
      return {};
    }
    std::suspend_always final_suspend() noexcept { return {}; }
    std::suspend_always return_value(int) noexcept { return {}; }
    void unhandled_exception() noexcept {}

    auto await_transform(int s) noexcept {
      struct awaiter {
        promise_type *promise;
        bool await_ready() const {
          return true;
        }
        int await_resume() const {
          return promise->message;
        }
        void await_suspend(std::coroutine_handle<>) {
        }
      };
      return awaiter{this};
    }
    int message;
  };

  Chat(std::coroutine_handle<promise_type> promise);

  std::coroutine_handle<promise_type> handle;
};

Chat f(int s) {
  // CHECK: CoyieldExpr {{.*}} <col:3, col:12>
  co_yield s;
  // CHECK: CoreturnStmt {{.*}} <line:74:3, col:13>
  co_return s;
  // CHECK: CoawaitExpr {{.*}} <col:3, col:12>
  co_await s;
}
