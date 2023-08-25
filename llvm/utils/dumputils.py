import re
import os
import tempfile
import pdb

class DumpFile(object):
  def __init__(self, path):
    self.passes = []
    self.lines = []
    f = open(path, 'r')
    for line in f.readlines():
      self.lines.append(line)
    f.close()
    pat_begin = re.compile('^# \*\*\* IR Dump [^\(]+ \(([a-zA-Z0-9\-]+)\) \*\*\*:$')
    begin = None
    begin_name = None
    for idx, line in enumerate(self.lines):
      if m := pat_begin.match(line):
        name = m.group(1)
        if begin:
          self.passes.append(Pass(self, begin_name, begin+1, idx))
        begin = idx
        begin_name = name
    self.passes.append(Pass(self, begin_name, begin+1, idx))

class Fragment(object):
  def __init__(self, dump, name, begin, end):
    self.dump = dump
    self.name = name
    self.begin = begin
    self.end = end

  def __eq__(self, other):
    return self.dump.lines[self.begin:self.end] == other.dump.lines[other.begin:other.end]

  def write(self, file):
    for idx in range(self.begin, self.end):
      file.write(self.dump.lines[idx])

class Pass(Fragment):
  def __init__(self, dump, name, begin, end):
    super().__init__(dump, name, begin, end)
    self.functions = {}

    pat_begin = re.compile('^# Machine code for function ([a-zA-Z_][a-zA-Z0-9_]+).*$')
    pat_end = re.compile('^# End machine code for function ([a-zA-Z_][a-zA-Z0-9_]+).*$')
    for idx in range(self.begin, self.end):
      line = self.dump.lines[idx]
      if m := pat_begin.match(line):
        begin = idx
      elif m := pat_end.match(line):
        name = m.group(1)
        self.functions[name] = Function(self.dump, name, begin, idx)


class Function(Fragment):
  def __init__(self, dump, name, begin, end):
    super().__init__(dump, name, begin, end)
    self.blocks = {}

    pat_begin = re.compile('^([0-9]+B)?\s*(bb\.[0-9]+)')
    begin = None
    begin_name = None
    for idx in range(self.begin, self.end):
      line = self.dump.lines[idx]
      if m := pat_begin.search(line):
        name = m.group(2)
        if begin:
          self.blocks[begin_name] = BasicBlock(self.dump, begin_name, begin, idx)
        begin = idx
        begin_name = name
    self.blocks[begin_name] = BasicBlock(self.dump, begin_name, begin, self.end)


class BasicBlock(Fragment):
  def __init__(self, dump, name, begin, end):
    super().__init__(dump, name, begin, end)

  def successors(self):
    pat_succs = re.compile('\s+successors:([^;]+).*$')
    pat_bb = re.compile('(bb.[0-9]+)')
    for idx in range(self.begin, self.end):
      line = self.dump.lines[idx]
      if m := pat_succs.match(line):
        return pat_bb.findall(m.group(1))
    return []

def diff(obj_a, obj_b):
  with tempfile.TemporaryDirectory() as tmpdirname:
    path_a = tmpdirname + '/a.txt'
    path_b = tmpdirname + '/b.txt'
    file_a = open(path_a, 'w')
    file_b = open(path_b, 'w')
    obj_a.write(file_a)
    obj_b.write(file_b)
    file_a.close()
    file_b.close()
    os.system('vimdiff {} {}'.format(path_a, path_b))

def view(obj_a):
  with tempfile.TemporaryDirectory() as tmpdirname:
    path_a = tmpdirname + '/a.txt'
    file_a = open(path_a, 'w')
    obj_a.write(file_a)
    file_a.close()
    os.system('vim {}'.format(path_a))

def graph(func):
  with tempfile.TemporaryDirectory() as tmpdirname:
    path = tmpdirname + '/cfg.dot'
    file = open(path, 'w')
    file.write('digraph CFG {\n')
    for _, b in func.blocks.items():
      for s in b.successors():
        file.write('  {} -> {}\n'.format(b.name.replace('.','_'), s.replace('.','_')))
    file.write('}\n')
    file.close()
    if os.system('dot -Tx11 {}'.format(path)) != 0:
      os.system('cat {}'.format(path))

def prune_passes(dump):
  mpasses = []
  mpasses.append(dump.passes[0])
  for p in dump.passes:
    if p != mpasses[-1]:
      mpasses.append(p)
  return mpasses
