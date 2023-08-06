// RUN: %check_clang_tidy %s bugprone-allocation-bool-conversion %t  -- -config="{CheckOptions: { bugprone-allocation-bool-conversion.portability-restrict-system-includes.AllocationFunctions: 'malloc;allocate;custom' }}"

typedef __SIZE_TYPE__ size_t;

void takeBool(bool);
void* operator new(size_t count);
void *malloc(size_t size);

template<typename T>
struct Allocator {
  typedef T* pointer;
  pointer allocate(size_t n, const void* hint = 0);
};

void* custom();
void* negative();

static Allocator<int> allocator;

void testImplicit() {
  takeBool(negative());

  takeBool(new int);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(new bool);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(operator new(10));
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'operator new' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(malloc(10));
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'malloc' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(allocator.allocate(1U));
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'allocate' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]

  takeBool(custom());

  bool value;

  value = negative();

  value = new int;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  value = new bool;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  value = operator new(10);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: result of the 'operator new' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  value = malloc(10);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: result of the 'malloc' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  value = allocator.allocate(1U);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: result of the 'allocate' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  value = custom();
}

void testExplicit() {
  takeBool(static_cast<bool>(negative()));

  takeBool(static_cast<bool>(new int));
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(static_cast<bool>(new bool));
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(static_cast<bool>(operator new(10)));
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: result of the 'operator new' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(static_cast<bool>(malloc(10)));
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: result of the 'malloc' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(static_cast<bool>(allocator.allocate(1U)));
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: result of the 'allocate' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(static_cast<bool>(custom()));
}

void testNegation() {
  takeBool(!negative());

  takeBool(!new bool);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(!new int);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(!operator new(10));
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: result of the 'operator new' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(!malloc(10));
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: result of the 'malloc' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(!allocator.allocate(1U));
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: result of the 'allocate' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(!custom());

  bool value;

  value = !negative();

  value = !new int;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  value = !new bool;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  value = !operator new(10);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'operator new' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  value = !malloc(10);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'malloc' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  value = !allocator.allocate(1U);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'allocate' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  value = !custom();
}
