// RUN: %check_clang_tidy %s bugprone-allocation-bool-conversion %t  -- -config="{CheckOptions: { \
// RUN:                      bugprone-allocation-bool-conversion.PointerReturningAllocators: 'ptr_custom',\
// RUN:                      bugprone-allocation-bool-conversion.IntegerReturningAllocators: 'int_custom'}}"

typedef __SIZE_TYPE__ size_t;

void takeBool(bool);
void* operator new(size_t count);
void *malloc(size_t size);
int open(const char *pathname, int flags);

template<typename T>
struct Allocator {
  typedef T* pointer;
  pointer allocate(size_t n, const void* hint = 0);
};

void* ptr_custom();
int  int_custom();
void* negative();
int negativeInt();

static Allocator<int> allocator;

void testImplicit() {
  takeBool(negative());
  takeBool(negativeInt());

  takeBool(new int);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(new bool);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(operator new(10));
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'operator new' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(malloc(10));
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'malloc' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(open("file", 0));
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'open' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(allocator.allocate(1U));
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'allocate' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(ptr_custom());
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'ptr_custom' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(int_custom());
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'int_custom' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]

  bool value;

  value = negative();
  value = negativeInt();

  value = new int;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  value = new bool;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  value = operator new(10);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: result of the 'operator new' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  value = malloc(10);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: result of the 'malloc' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  value = open("file", 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: result of the 'open' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  value = allocator.allocate(1U);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: result of the 'allocate' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  value = ptr_custom();
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: result of the 'ptr_custom' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  value = int_custom();
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: result of the 'int_custom' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
}

void testExplicit() {
  takeBool(static_cast<bool>(negative()));
  takeBool(static_cast<bool>(negativeInt()));

  takeBool(static_cast<bool>(new int));
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(static_cast<bool>(new bool));
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(static_cast<bool>(operator new(10)));
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: result of the 'operator new' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(static_cast<bool>(malloc(10)));
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: result of the 'malloc' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(static_cast<bool>(open("file", 0)));
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: result of the 'open' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(static_cast<bool>(allocator.allocate(1U)));
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: result of the 'allocate' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(static_cast<bool>(ptr_custom()));
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: result of the 'ptr_custom' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(static_cast<bool>(int_custom()));
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: result of the 'int_custom' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
}

void testNegation() {
  takeBool(!negative());
  takeBool(!negativeInt());

  takeBool(!new bool);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(!new int);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(!operator new(10));
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: result of the 'operator new' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(!malloc(10));
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: result of the 'malloc' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(!open("file", 0));
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: result of the 'open' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(!allocator.allocate(1U));
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: result of the 'allocate' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(!ptr_custom());
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: result of the 'ptr_custom' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  takeBool(!int_custom());
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: result of the 'int_custom' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]

  bool value;

  value = !negative();
  value = !negativeInt();

  value = !new int;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  value = !new bool;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  value = !operator new(10);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'operator new' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  value = !malloc(10);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'malloc' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  value = !open("file", 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'open' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  value = !allocator.allocate(1U);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'allocate' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  value = !ptr_custom();
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'ptr_custom' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
  value = !int_custom();
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'int_custom' call is being used as a boolean value, which may lead to unintended behavior or resource leaks [bugprone-allocation-bool-conversion]
}
