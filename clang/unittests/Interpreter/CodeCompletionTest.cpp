#include "clang/Interpreter/CodeCompletion.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Interpreter/Interpreter.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "llvm/LineEditor/LineEditor.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace clang;
namespace {
auto CB = clang::IncrementalCompilerBuilder();

static std::unique_ptr<Interpreter> createInterpreter() {
  auto CI = cantFail(CB.CreateCpp());
  return cantFail(clang::Interpreter::create(std::move(CI)));
}

static std::vector<std::string> runComp(clang::Interpreter &MainInterp,
                                        llvm::StringRef Prefix,
                                        llvm::Error &ErrR) {
  auto CI = CB.CreateCpp();
  if (auto Err = CI.takeError()) {
    ErrR = std::move(Err);
    return {};
  }

  auto Interp = clang::Interpreter::create(std::move(*CI));
  if (auto Err = Interp.takeError()) {
    // log the error and returns an empty vector;
    ErrR = std::move(Err);

    return {};
  }

  std::vector<std::string> Results;
  std::vector<std::string> Comps;

  codeComplete(
      const_cast<clang::CompilerInstance *>((*Interp)->getCompilerInstance()),
      Prefix, /* Lines */ 1, Prefix.size(), MainInterp.getCompilerInstance(),
      Results);

  for (auto Res : Results)
    if (Res.find(Prefix) == 0)
      Comps.push_back(Res);

  return Comps;
}

TEST(CodeCompletionTest, Sanity) {
  auto Interp = createInterpreter();
  if (auto R = Interp->ParseAndExecute("int foo = 12;")) {
    consumeError(std::move(R));
    return;
  }
  auto Err = llvm::Error::success();
  auto comps = runComp(*Interp, "f", Err);
  EXPECT_EQ((size_t)2, comps.size()); // foo and float
  EXPECT_EQ(comps[0], std::string("foo"));
  EXPECT_EQ((bool)Err, false);
}

TEST(CodeCompletionTest, SanityNoneValid) {
  auto Interp = createInterpreter();
  if (auto R = Interp->ParseAndExecute("int foo = 12;")) {
    consumeError(std::move(R));
    return;
  }
  auto Err = llvm::Error::success();
  auto comps = runComp(*Interp, "babanana", Err);
  EXPECT_EQ((size_t)0, comps.size()); // foo and float
  EXPECT_EQ((bool)Err, false);
}

TEST(CodeCompletionTest, TwoDecls) {
  auto Interp = createInterpreter();
  if (auto R = Interp->ParseAndExecute("int application = 12;")) {
    consumeError(std::move(R));
    return;
  }
  if (auto R = Interp->ParseAndExecute("int apple = 12;")) {
    consumeError(std::move(R));
    return;
  }
  auto Err = llvm::Error::success();
  auto comps = runComp(*Interp, "app", Err);
  EXPECT_EQ((size_t)2, comps.size());
  EXPECT_EQ((bool)Err, false);
}

TEST(CodeCompletionTest, CompFunDeclsNoError) {
  auto Interp = createInterpreter();
  auto Err = llvm::Error::success();
  auto comps = runComp(*Interp, "void app(", Err);
  EXPECT_EQ((bool)Err, false);
}

TEST(CodeCompletionTest, TypedDirected) {
  auto Interp = createInterpreter();
  if (auto R = Interp->ParseAndExecute("int application = 12;")) {
    consumeError(std::move(R));
    return;
  }
  if (auto R = Interp->ParseAndExecute("char apple = '2';")) {
    consumeError(std::move(R));
    return;
  }
  if (auto R = Interp->ParseAndExecute("void add(int &SomeInt){}")) {
    consumeError(std::move(R));
    return;
  }
  {
    auto Err = llvm::Error::success();
    auto comps = runComp(*Interp, std::string("add("), Err);
    EXPECT_EQ((size_t)1, comps.size());
  }

  if (auto R = Interp->ParseAndExecute("int banana = 42;")) {
    consumeError(std::move(R));
    return;
  }

  {
    auto Err = llvm::Error::success();
    auto comps = runComp(*Interp, std::string("add("), Err);
    EXPECT_EQ((size_t)2, comps.size());
    EXPECT_EQ(comps[0], "application");
    EXPECT_EQ(comps[1], "banana");
  }

  {
    auto Err = llvm::Error::success();
    auto comps = runComp(*Interp, std::string("add(b"), Err);
    EXPECT_EQ((size_t)1, comps.size());
    EXPECT_EQ(comps[0], "anana");
  }
}

TEST(CodeCompletionTest, SanityClasses) {
  auto Interp = createInterpreter();
  if (auto R = Interp->ParseAndExecute("struct Apple{};")) {
    consumeError(std::move(R));
    return;
  }
  if (auto R = Interp->ParseAndExecute("void takeApple(Apple &a1){}")) {
    consumeError(std::move(R));
    return;
  }
  if (auto R = Interp->ParseAndExecute("Apple a1;")) {
    consumeError(std::move(R));
    return;
  }
  if (auto R = Interp->ParseAndExecute("void takeAppleCopy(Apple a1){}")) {
    consumeError(std::move(R));
    return;
  }

  {
    auto Err = llvm::Error::success();
    auto comps = runComp(*Interp, "takeApple(", Err);
    EXPECT_EQ((size_t)1, comps.size());
    EXPECT_EQ(comps[0], std::string("a1"));
  }
  {
    auto Err = llvm::Error::success();
    auto comps = runComp(*Interp, std::string("takeAppleCopy("), Err);
    EXPECT_EQ((size_t)1, comps.size());
    EXPECT_EQ(comps[0], std::string("a1"));
  }
}

TEST(CodeCompletionTest, SubClassing) {
  auto Interp = createInterpreter();
  if (auto R = Interp->ParseAndExecute("struct Fruit {};")) {
    consumeError(std::move(R));
    return;
  }
  if (auto R = Interp->ParseAndExecute("struct Apple : Fruit{};")) {
    consumeError(std::move(R));
    return;
  }
  if (auto R = Interp->ParseAndExecute("void takeFruit(Fruit &f){}")) {
    consumeError(std::move(R));
    return;
  }
  if (auto R = Interp->ParseAndExecute("Apple a1;")) {
    consumeError(std::move(R));
    return;
  }
  if (auto R = Interp->ParseAndExecute("Fruit f1;")) {
    consumeError(std::move(R));
    return;
  }
  auto Err = llvm::Error::success();
  auto comps = runComp(*Interp, std::string("takeFruit("), Err);
  EXPECT_EQ((size_t)2, comps.size());
  EXPECT_EQ(comps[0], std::string("a1"));
  EXPECT_EQ(comps[1], std::string("f1"));
}

} // anonymous namespace
