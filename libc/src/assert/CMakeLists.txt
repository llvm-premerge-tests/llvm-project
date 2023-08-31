if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})

  add_entrypoint_object(
    __assert_fail
      ALIAS
      DEPENDS
      libc.src.assert.${LIBC_TARGET_OS}.__assert_fail
  )
  return()
endif()

add_entrypoint_object(
  __assert_fail
  SRCS
    __assert_fail.cpp
  HDRS
    __assert_fail.h
    assert.h
  DEPENDS
    libc.include.assert
    libc.src.__support.OSUtil.osutil
    libc.src.stdlib.abort
)
