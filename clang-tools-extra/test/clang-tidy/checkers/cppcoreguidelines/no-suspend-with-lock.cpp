// RUN: %check_clang_tidy -std=c++20-or-later %s cppcoreguidelines-no-suspend-with-lock %t

// NOLINTBEGIN
namespace std {
template <typename T, typename... Args>
struct coroutine_traits {
  using promise_type = typename T::promise_type;
};
template <typename T = void>
struct coroutine_handle;
template <>
struct coroutine_handle<void> {
  coroutine_handle() noexcept;
  coroutine_handle(decltype(nullptr)) noexcept;
  static constexpr coroutine_handle from_address(void*);
};
template <typename T>
struct coroutine_handle {
  coroutine_handle() noexcept;
  coroutine_handle(decltype(nullptr)) noexcept;
  static constexpr coroutine_handle from_address(void*);
  operator coroutine_handle<>() const noexcept;
};

template <class Mutex>
class unique_lock {
public:
  unique_lock() noexcept;
  explicit unique_lock(Mutex &m);
  unique_lock& operator=(unique_lock&&);
  void unlock();
  Mutex* release() noexcept;
  Mutex* mutex() const noexcept;
  void swap(unique_lock& other) noexcept;
};

class mutex {
public:
  mutex() noexcept;
  ~mutex();
  mutex(const mutex &) = delete;
  mutex &operator=(const mutex &) = delete;

  void lock();
  void unlock();
};
} // namespace std

struct Awaiter {
  bool await_ready() noexcept;
  void await_suspend(std::coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

struct Coro {
  struct promise_type {
    Awaiter initial_suspend();
    Awaiter final_suspend() noexcept;
    void return_void();
    Coro get_return_object();
    void unhandled_exception();
    Awaiter yield_value(int);
  };
};
// NOLINTEND

std::mutex mtx;
std::mutex mtx2;

Coro awaits_with_lock() {
  std::unique_lock<std::mutex> lock(mtx);

  co_await Awaiter{};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: coroutine suspended with lock 'lock' held [cppcoreguidelines-no-suspend-with-lock]

  co_await Awaiter{};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: coroutine suspended with lock 'lock' held [cppcoreguidelines-no-suspend-with-lock]

  if (true) {
    co_await Awaiter{};
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: coroutine suspended with lock 'lock' held [cppcoreguidelines-no-suspend-with-lock]
  }

  if (true) {
    std::unique_lock<std::mutex> lock2;
    lock2.unlock();
    co_await Awaiter{};
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: coroutine suspended with lock 'lock2' held [cppcoreguidelines-no-suspend-with-lock]
  }
}

Coro awaits_with_lock_in_try() try {
  std::unique_lock<std::mutex> lock(mtx);
  co_await Awaiter{};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: coroutine suspended with lock 'lock' held [cppcoreguidelines-no-suspend-with-lock]
} catch (...) {}

Coro lock_possibly_unlocked() {
  // CppCoreGuideline CP.52's enforcement strictly requires flagging
  // code that suspends while any lock guard is not destructed.

  {
    std::unique_lock<std::mutex> lock(mtx);
    lock.unlock();
    co_await Awaiter{};
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: coroutine suspended with lock 'lock' held [cppcoreguidelines-no-suspend-with-lock]
  }

  {
    std::unique_lock<std::mutex> lock(mtx);
    lock.release();
    co_await Awaiter{};
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: coroutine suspended with lock 'lock' held [cppcoreguidelines-no-suspend-with-lock]
  }

  {
    std::unique_lock<std::mutex> lock(mtx);
    std::unique_lock<std::mutex> lock2;
    lock.swap(lock2);
    co_await Awaiter{};
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: coroutine suspended with lock 'lock' held [cppcoreguidelines-no-suspend-with-lock]
  }

  {
    std::unique_lock<std::mutex> lock(mtx);
    std::unique_lock<std::mutex> lock2{mtx2};
    lock.swap(lock2);
    co_await Awaiter{};
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: coroutine suspended with lock 'lock' held [cppcoreguidelines-no-suspend-with-lock]
  }

  {
    std::unique_lock<std::mutex> lock(mtx);
    lock = std::unique_lock<std::mutex>{};
    co_await Awaiter{};
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: coroutine suspended with lock 'lock' held [cppcoreguidelines-no-suspend-with-lock]
  }

  {
    std::unique_lock<std::mutex> lock(mtx);
    lock = std::unique_lock<std::mutex>{mtx2};
    co_await Awaiter{};
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: coroutine suspended with lock 'lock' held [cppcoreguidelines-no-suspend-with-lock]
  }
}

Coro await_with_underlying_mutex_unlocked() {
  std::unique_lock<std::mutex> lock(mtx);

  // Even though we unlock the mutex here, 'lock' is still active unless
  // there is a call to lock.unlock(). This is a bug in the program since
  // it will result in locking the mutex twice. The check does not track
  // unlock calls on the underlying mutex held by a lock guard object.
  mtx.unlock();

  co_await Awaiter{};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: coroutine suspended with lock 'lock' held [cppcoreguidelines-no-suspend-with-lock]
}

Coro await_with_empty_lock() {
  std::unique_lock<std::mutex> lock;
  co_await Awaiter{};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: coroutine suspended with lock 'lock' held [cppcoreguidelines-no-suspend-with-lock]
}

Coro await_before_lock() {
  co_await Awaiter{};
  std::unique_lock<std::mutex> lock(mtx);
}

Coro await_with_lock_different_scope() {
  {
    std::unique_lock<std::mutex> lock(mtx);
  }
  co_await Awaiter{};
}

Coro await_with_goto() {
first:
  co_await Awaiter{};
  std::unique_lock<std::mutex> lock(mtx);
  goto first;
}

Coro yields_with_lock() {
  std::unique_lock<std::mutex> lock(mtx);
  co_yield 0;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: coroutine suspended with lock 'lock' held [cppcoreguidelines-no-suspend-with-lock]
}
