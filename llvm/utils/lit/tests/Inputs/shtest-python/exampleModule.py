# The importer should assign this to the 'lit' object, or uses of 'lit' below
# will fail.
lit = None

import inspect


def helloWorldFromLitRun():
    loc = f"{inspect.stack()[0].filename}:{inspect.stack()[0].lineno + 1}"
    lit.run(f"echo hello world from lit.run at {loc}")


def goodbyeWorldFromLitRun():
    loc = f"{inspect.stack()[0].filename}:{inspect.stack()[0].lineno + 1}"
    lit.run(f"echo goodbye world from lit.run at {loc}")
