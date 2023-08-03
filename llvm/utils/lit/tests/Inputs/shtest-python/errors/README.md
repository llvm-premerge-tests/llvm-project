One goal of the tests in this directory is to check that diagnostics from the
python interpreter are faithfully copied to the lit user.  However, verifying
the exact text of those diagnostics makes the tests brittle across python
versions.

Please follow the guide below when adding new tests, and update it if you find
additional issues.  If you think of a better strategy for checking lit's
handling of python diagnostics, please propose it.

So far in the tests we have written, the following diagnostic text seems to be
stable across python versions, and it seems worthwhile to verify that it is
copied faithfully to the lit user:

  - The traceback.
  - When the erroneous python statement contains only one line, it is the last
    line that is quoted from the PYTHON block.
  - The presence of a caret (`^`) after the last line quoted from the PYTHON
    block.
  - The exception, if raised directly by our test code.

The following have proven unstable:

  - The last line that is quoted from the PYTHON block when the erroneous
    python statement contains multiple lines.
  - The number of context lines quoted from the PYTHON block.  Of course, if
    the PYTHON block has only one line, that number is stably zero.
  - The indentation of the last line quoted from the PYTHON block.  In
    particular, we have found that, independently of lit, at least python 3.8.0
    sometimes botches the indentation relative to other quoted lines and the
    caret, making the diagnostic confusing.
  - The indentation of the caret, including where it points in the last quoted
    line.
  - The exception, including its kind (e.g., `SyntaxError: invalid syntax`), if
    not raised directly by our test code.
