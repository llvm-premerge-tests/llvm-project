=================
Time Zone Support
=================

Introduction
============

Starting with C++20 the ``<chrono>`` library has support for time zones.
These are available in the
`IANA Time Zone Database <https://data.iana.org/time-zones/tz-link.html>`_.
This page describes the design decisions and trade-offs made to implement this
feature. This page contains several links with more information regarding the
contents of the IANA database, this page assumes the reader is familiar with
this information.

Which version of the Time Zone Database to use
==============================================

The data of the database is available in several platforms in different forms:

- Typically Unix systems ship the database as
  `TZif files <https://www.rfc-editor.org/rfc/rfc8536.html>`_. This format has
  3 versions and the ``time_zone_link`` information is not always available.
  If available, they are symlinks in the file system.
  These files don't provide the database version information. This information
  is needed for the functions ``std::chrono:: remote_version()`` and
  ``std::chrono::reload_tzdb()``.

- On several Unix systems the time zone source files are available. These files
  are stored in several regions, mainly the continents. This file contains a
  large amount of comment with historical information regarding time zones.
  The format is documented in the
  `IANA documentation <https://data.iana.org/time-zones/tz-how-to.html>`_
  and in the `man page <https://man7.org/linux/man-pages/man8/zic.8.html>`_ of zic.
  The disadvantage of this version is that at least Linux versions don't have
  the database version information. This information is needed for the functions
  ``std::chrono:: remote_version()`` and ``std::chrono::reload_tzdb()``.

- On Linux systems ``tzdata.zi`` is available. This contains the same
  information as the source files but in one file without the comments. This
  file uses the same format as the sources, but shortens the names. For example
  ``Rule`` is abbreviated to ``R``. This file contains the database version
  information.

The disadvantage of the binary data is that it's not possible on all platforms
to get the proper ``time_zone_link`` information. This information is in the
database, this difference is observable by users. Without the database version
information it's not possible to create a conforming implementation.

Since it's easier to parse one file than a set of files the libc++ team decided
to use the ``tzdata.zi``. The other benefit is that the ``tzdata.zi`` file
contains the database version information needed for a conforming
implementation.

The file ``tzdata.zi`` is not available on all platforms, so vendors need to
make changes for their platform.  Most vendors already ship the database, so
they need to adjust the packaging of their time zone package. On Windows this
is not available by default. However it's possible for Windows packagers to add
these files to their libc++ packages. The IANA databases can be
`downloaded <https://data.iana.org/time-zones/releases/>`_.

An alternative would be to ship the database with libc++, either as a file or
compiled in the dylib. The text file is about 112 KB. For now libc++ will not
ship this file. When it's hard to get vendors to ship these files we can
reconsider based on that information.

Leap seconds
------------

For the leap seconds libc++ will use the source file ``leap-seconds.list``.
This file is easier to parse than the ``leapseconds`` file. Both files are
present on Linux, but not always on other platforms. Since these platforms need
to change their packaging for ``tzdata.zi``, adding two instead of one files
seems a small change.


Selecting the Time Zone Database location
=========================================

There are several mechanisms to select the database location

- Use the location known to libc++. At the moment of writing not all platforms
  ship the database. Those that ship the database have the path configured in
  ``include/__config``.
- The CMake configuration option ``LIBCXX_TIME_ZONE_DB``. This makes it
  possible for vendors to ship the database when its location is unknown to
  libc++. The vendor may opt to notify the libc++ development team when the
  location is hard coded so the default location can be set. However when the
  location is configurable this offers a flexible way to configure the location
  system wide.
- Overriding ``std::chrono::__libcpp_tzdb_directory()``. This allows
  applications to specify their own database path and implement their own time
  zone update policy. This may be important for applications that require more
  up to date information than their operating system provides. This is also
  heavily used to test the parsing of the time zone database in libc++.
  This includes testing with corrupt databases.


Updating the Time Zone Database
===============================

Per `[time.zone.db.remote]/1 <http://eel.is/c++draft/time.zone#db.remote-1>`_

.. code-block:: text

  The local time zone database is that supplied by the implementation when the
  program first accesses the database, for example via current_zone(). While the
  program is running, the implementation may choose to update the time zone
  database. This update shall not impact the program in any way unless the
  program calls the functions in this subclause. This potentially updated time
  zone database is referred to as the remote time zone database.

There is an update mechanism in libc++, however this is not done automatically.
Invoking the function ``std::chrono:: remote_version()`` will parse the version
information of the ``tzdata.zi`` file and returns that information. Similar
``std::chrono::reload_tzdb()`` will parse the ``tzdata.zi`` and
``leap-seconds.list`` again. This makes is possible to update the database if
needed by the application and gives the user full power over the update policy.

This approach has several advantages:

- It is simple to implement.
- The library does not need to start a periodic background process to poll
  changes to the filesystem. When using a background process, it may become
  active when the application is busy with its core task, taking away resources
  from that task.
- When there is no threading available this polling
  becomes more involved. For example, query the file every *x* calls to
  ``std::chrono::get_tzdb()``. This mean calls to ``std::chrono::get_tzdb()``
  has different performance characteristics.
- Even when the automatic update is implemented, user may want
  ``std::chrono:: remote_version()`` to load the most recent information. For
  example since they use ``std::chrono::__libcpp_tzdb_directory()`` to have
  full control over the time zone information in their application.
  That means ``std::chrono:: remote_version()`` should always query.

The small drawback is:

- On platforms with threading enabled updating the database may take longer.
  On these platforms the remote database could have been loaded in a background
  process.

Another issue with the automatic update is, that is may not be considered
Standard compliant "This update shall not impact the program in any way". Using
resources could be considered impacting the program.

