# If lit.cfg sets config.prologue to this file, this file executes immediately
# before the first PYTHON directive almost if it appears in a PYTHON directive.
# One notable difference is __file__ is set to the absolute location of this
# file to help with imports.

import os, sys

sys.path.append(os.path.dirname(__file__))

import exampleModule

exampleModule.lit = lit  # so the module can access the lit object

# The following helps us test the execution trace of config.prologue.

import inspect

print("config.prologue writes to stdout")
print("config.prologue writes to stderr", file=sys.stderr)
lit.run("true")  # writes nothing to stdout or stderr
loc = str(inspect.stack()[0].lineno + 1)
lit.run("echo config.prologue lit.run writes to stdout at line " + loc)
loc = str(inspect.stack()[0].lineno + 1)
lit.run("%{python} %S/write-to-stderr.py && echo at line " + loc)
loc = str(inspect.stack()[0].lineno + 1)
lit.run("%{python} %S/write-to-stdout-and-stderr.py && echo at line " + loc)


def localFnA():
    loc = str(inspect.stack()[0].lineno + 1)
    lit.run("echo config.prologue lit.run from func at line " + loc)
def localFnB():
    localFnA()  # must be at above lit.run line number + 2
localFnB()  # must be at above lit.run line number + 3
exampleModule.helloWorldFromLitRun()

del inspect, os, sys  # optional: just avoids unnecessary pollution across tests
