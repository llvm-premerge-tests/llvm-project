================
LLDB Code Owners
================

This file is a list of the people responsible for ensuring that patches for a
particular part of LLDB are reviewed, either by themself or by someone else.
They are also the gatekeepers for their part of LLDB, with the final word on
what goes in or not.

.. contents::
   :depth: 2
   :local:

Current Code Owners
===================
The following people are the active code owners for the project. Please reach
out to them for code reviews, questions about their area of expertise, or other
assistance.

All parts of LLDB not covered by someone else
----------------------------------------------
| Jonas Devlieghere
| jonas\@devlieghere.com (email), jdevlieghere (Phabricator), jdevlieghere (GitHub)

Components
----------
These code owners are responsible for particular high-level components within
LLDB.

ABI
~~~
| Jason Molenda
| jmolenda\@apple.com (email), jasonmolenda (Phabricator), jasonmolenda (GitHub)

| David Spickett
| david.spickett\@linaro.org (email), DavidSpickett (Phabricator), DavidSpickett (GitHub)


Breakpoint
~~~~~~~~~~
| Jim Ingham
| jingham\@apple.com (email), jingham (Phabricator), jimingham (GitHub)

CMake & Build System
~~~~~~~~~~~~~~~~~~~~
| Jonas Devlieghere
| jonas\@devlieghere.com (email), jdevlieghere (Phabricator), jdevlieghere (GitHub)

| Alex Langford
| alangford\@apple.com (email), bulbazord (Phabricator), bulbazord (GitHub)

Commands
~~~~~~~~
| Jim Ingham
| jingham\@apple.com (email), jingham (Phabricator), jimingham (GitHub)

Expression Parser
~~~~~~~~~~~~~~~~~
| Michael Buch
| michaelbuch12\@gmail.com (email), Michael137 (Phabricator), Michael137 (GitHub)

| Jim Ingham
| jingham\@apple.com (email), jingham (Phabricator), jimingham (GitHub)

Interpreter
~~~~~~~~~~~
| Jim Ingham
| jingham\@apple.com (email), jingham (Phabricator), jimingham (GitHub)

| Greg Clayton
| gclayton\@fb.com (email), clayborg (Phabricator), clayborg (GitHub)


Lua
~~~
| Jonas Delvieghere
| jonas\@devlieghere.com (email), jdevlieghere (Phabricator), jdevlieghere (GitHub)

Python
~~~~~~
| Med Ismail Bennani
| ismail\@bennani.ma (email), mib (Phabricator), medismailben (GitHub)

Target/Process Control
~~~~~~~~~~~~~~~~~~~~~~
| Med Ismail Bennani
| ismail\@bennani.ma (email), mib (Phabricator), medismailben (GitHub)

| Jim Ingham
| jingham\@apple.com (email), jingham (Phabricator), jimingham (GitHub)

Test Suite
~~~~~~~~~~
| Jonas Devlieghere
| jonas\@devlieghere.com (email), jdevlieghere (Phabricator), jdevlieghere (GitHub)

| Pavel Labath
| pavel\@labath.sk (email), labath (Phabricator), labath (GitHub)

Trace
~~~~~
| Walter Erquinigo
| a20012251\@gmail.com (email), wallace (Phabricator), walter-erquinigo (GitHub)

Unwinding
~~~~~~~~~
| Jason Molenda
| jmolenda\@apple.com (email), jasonmolenda (Phabricator), jasonmolenda (GitHub)

Utility
~~~~~~~
| Pavel Labath
| pavel\@labath.sk (email), labath (Phabricator), labath (GitHub)

ValueObject
~~~~~~~~~~~
| Jim Ingham
| jingham\@apple.com (email), jingham (Phabricator), jimingham (GitHub)

Watchpoints
~~~~~~~~~~~
| Jason Molenda
| jmolenda\@apple.com (email), jasonmolenda (Phabricator), jasonmolenda (GitHub)

File Formats
------------
The following people are responsible for decisions involving file and debug
info formats.

(PE)COFF
~~~~~~~~
| Saleem Abdulrasool
| compnerd\@compnerd.org (email), compnerd (Phabricator), compnerd (GitHub)

Breakpad
~~~~~~~~
| Pavel Labath
| pavel\@labath.sk (email), labath (Phabricator), labath (GitHub)

CTF
~~~
| Jonas Devlieghere
| jonas\@devlieghere.com (email), jdevlieghere (Phabricator), jdevlieghere (GitHub)

DWARF
~~~~~
| Adrian Prantl
| aprantl\@apple.com (email), aprantl (Phabricator), adrian-prantl (GitHub)

| Greg Clayton
| gclayton\@fb.com (email), clayborg (Phabricator), clayborg (GitHub)

ELF
~~~
| Pavel Labath
| pavel\@labath.sk (email), labath (Phabricator), labath (GitHub)

JSON
~~~~
| Jonas Devlieghere
| jonas\@devlieghere.com (email), jdevlieghere (Phabricator), jdevlieghere (GitHub)

MachO
~~~~~
| Greg Clayton
| gclayton\@fb.com (email), clayborg (Phabricator), clayborg (GitHub)

| Jason Molenda
| jmolenda\@apple.com (email), jasonmolenda (Phabricator), jasonmolenda (GitHub)

PDB
~~~
| Zequan Wu
| zequanwu\@google.com (email), zequanwu (Phabricator), ZequanWu (GitHub)

Platforms
---------
The following people are responsible for decisions involving platforms.

Android
~~~~~~~
| Pavel Labath
| pavel\@labath.sk (email), labath (Phabricator), labath (GitHub)

Darwin
~~~~~~
| Jim Ingham
| jingham\@apple.com (email), jingham (Phabricator), jimingham (GitHub)

| Jason Molenda
| jmolenda\@apple.com (email), jasonmolenda (Phabricator), jasonmolenda (GitHub)

| Jonas Devlieghere
| jonas\@devlieghere.com (email), jdevlieghere (Phabricator), jdevlieghere (GitHub)

FreeBSD
~~~~~~~
| Ed Maste
| emaste\@freebsd.org (email), emaste (Phabricator), emaste (GitHub)

Linux
~~~~~
| Pavel Labath
| pavel\@labath.sk (email), labath (Phabricator), labath (GitHub)

| David Spickett
| david.spickett\@linaro.org (email), DavidSpickett (Phabricator), DavidSpickett (GitHub)

Windows
~~~~~~~
| Omair Javaid
| omair.javaid\@linaro.org (email), omjavaid (Phabricator), omjavaid (GitHub)


Tools
-----
The following people are responsible for decisions involving specific tools.

debugserver
~~~~~~~~~~~
| Jason Molenda
| jmolenda\@apple.com (email), jasonmolenda (Phabricator), jasonmolenda (GitHub)

lldb-server
~~~~~~~~~~~
| Pavel Labath
| pavel\@labath.sk (email), labath (Phabricator), labath (GitHub)

lldb-vscode
~~~~~~~~~~~
| Greg Clayton
| gclayton\@fb.com (email), clayborg (Phabricator), clayborg (GitHub)

| Walter Erquinigo
| a20012251\@gmail.com (email), wallace (Phabricator), walter-erquinigo (GitHub)

Former Code Owners
==================
The following people have graciously spent time performing code ownership
responsibilities but are no longer active in that role. Thank you for all your
help with the success of the project!

| Kamil Rytarowski (kamil\@netbsd.org)
| Zachary Turner (zturner\@google.com)
