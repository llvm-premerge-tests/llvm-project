// RUN: %check_clang_tidy %s bugprone-new-bool-conversion %t

void takeBool(bool);

void testImplicit() {
  takeBool(new int);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or memory leaks [bugprone-new-bool-conversion]
  takeBool(new bool);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or memory leaks [bugprone-new-bool-conversion]

  bool value;

  value = new int;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or memory leaks [bugprone-new-bool-conversion]
  value = new bool;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or memory leaks [bugprone-new-bool-conversion]
}

void testExplicit() {
  takeBool(static_cast<bool>(new int));
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or memory leaks [bugprone-new-bool-conversion]
  takeBool(static_cast<bool>(new bool));
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or memory leaks [bugprone-new-bool-conversion]
}

void testNegation() {
  takeBool(!new bool);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or memory leaks [bugprone-new-bool-conversion]
  takeBool(!new int);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or memory leaks [bugprone-new-bool-conversion]

  bool value;

  value = !new int;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or memory leaks [bugprone-new-bool-conversion]
  value = !new bool;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: result of the 'new' expression is being used as a boolean value, which may lead to unintended behavior or memory leaks [bugprone-new-bool-conversion]
}
