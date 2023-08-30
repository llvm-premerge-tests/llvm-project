set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangModuleBuildDaemon
  Protocol.cpp
  SocketSupport.cpp
  SocketMsgSupport.cpp
  )