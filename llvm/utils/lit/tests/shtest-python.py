# We're using PYTHON to help us write tests for PYTHON.

# RUN: echo "-- Available Tests --" > %t.tests.actual.txt

# DEFINE: %{my-inputs} = %{inputs}/shtest-python

# PYTHON: def runTest(test, litPre="", prologue=None):
# PYTHON:     litCmd = litPre + " %{lit} -va"
# PYTHON:     if prologue:
# PYTHON:         litCmd += " -DaddPrologue=" + prologue
# PYTHON:     testAbs = "%{my-inputs}/" + test
# PYTHON:     lit.run(f"""
# PYTHON:         {litCmd} {testAbs} 2>&1 |
# PYTHON:           FileCheck {testAbs} -strict-whitespace -match-full-lines
# PYTHON:                               -dump-input-filter=all -vv -color &&
# PYTHON:         echo "  shtest-python :: {test}" >> %t.tests.actual.txt
# PYTHON:     """)
# PYTHON: def runPass(test, prologue=None):
# PYTHON:     runTest(test, prologue=prologue)
# PYTHON: def runFail(test, prologue=None):
# PYTHON:     runTest(test, litPre="not", prologue=prologue)

# PYTHON: runFail("errors/incomplete-python.txt")
# PYTHON: runFail("errors/inconsistent-indent-2-lines.txt")
# PYTHON: runFail("errors/inconsistent-indent-3-lines.txt")
# PYTHON: runFail("errors/internal-api-inaccessible.txt")
# PYTHON: runFail("errors/python-syntax-error-1-line.txt")
# PYTHON: runFail("errors/python-syntax-error-2-lines.txt")
# PYTHON: runFail("errors/python-syntax-error-3-lines.txt")
# PYTHON: runFail("errors/python-syntax-error-leading-indent.txt")
# PYTHON: runFail("errors/trace-compile-error.txt")
# PYTHON: runFail("errors/trace-exception.txt")
# PYTHON: runFail("errors/trace-prologue-exception.txt",
# PYTHON:         prologue="errors/lit.prologue.py")
# PYTHON: runFail("errors/trace-run-compile-error.txt")

# PYTHON: runPass("has.txt")
# PYTHON: runPass("import.txt")
# PYTHON: runPass("no-shell-commands.txt")
# PYTHON: runPass("prologue.txt", prologue="lit.prologue.py")
# PYTHON: runPass("shell-affects-python.txt")
# PYTHON: runPass("substs-affects-python.txt")
# PYTHON: runPass("trace.txt", prologue="lit.prologue.py")

# Make sure we didn't forget to run something.
#
# RUN: %{lit} --show-tests %{my-inputs} > %t.tests.expected.txt
# RUN: diff -u -w %t.tests.expected.txt %t.tests.actual.txt
