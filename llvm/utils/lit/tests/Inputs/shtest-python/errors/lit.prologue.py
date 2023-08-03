import inspect, sys

print("config.prologue writes to stdout")
print("config.prologue writes to stderr", file=sys.stderr)
line = str(inspect.stack()[0].lineno + 1)
raise Exception(f"exception in config.prologue at line {line}")
