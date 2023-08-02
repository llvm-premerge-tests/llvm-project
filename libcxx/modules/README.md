# The "module partitions" for the std module

The files in this directory contain the exported named declarations per header.
These files are used for the following purposes

 - During testing exported named declarations are tested against the named
   declarations in the associated header. This excludes reserved names; they
   are not exported.
 - Generate the module std.

These use cases require including the required headers for these "partitions"
at different locations. This means the user of these "partitions" are
responsible for including the proper header and validate whether the header can
be loaded in the current libc++ configuration. For example "include <locale>"
fails when locales are not available. The "partitions" use the libc++ feature
macros to export the declarations available in the current configuration. This
configuration is available if the user includes the `__config' header.

Originally the files in this directory were real C++ module partitions. It
turned out module partitions caused overhead when using the std module. This
currently used method is a lot faster. The drawback is the files in this
directory are no longer selfcontained. The performance issue was discussed on
[Discourse](https://discourse.llvm.org/t/alternatives-to-the-implementation-of-std-modules/71958).
