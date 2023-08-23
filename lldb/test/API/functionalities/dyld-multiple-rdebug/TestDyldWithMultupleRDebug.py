"""
Test that LLDB can launch a linux executable through the dynamic loader where
the main executable has an extra exported "_r_debug" symbol that used to mess
up shared library loading with DYLDRendezvous and the POSIX dynamic loader
plug-in. What used to happen is that any shared libraries other than the main
executable and the dynamic loader and VSDO would not get loaded. This test
checks to make sure that we still load libraries correctly when we have
multiple "_r_debug" symbols. See comments in the main.cpp source file for full
details on what the problem is.
"""

import lldb
import os

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestDyldWithMultipleRDebug(TestBase):
    @skipIf(oslist=no_match(["linux"]))
    @no_debug_info_test
    @skipIf(oslist=["linux"], archs=["arm"])
    def test(self):
        self.build()

        # Extracts path of the interpreter.
        exe = self.getBuildArtifact("a.out")
        print(exe)
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set breakpoints both on shared library function as well as on
        # main. Both of them will be pending breakpoints.
        breakpoint_main = target.BreakpointCreateBySourceRegex(
            "// Break here", lldb.SBFileSpec("main.cpp")
        )
        breakpoint_shared_library = target.BreakpointCreateBySourceRegex(
            "// Library break here", lldb.SBFileSpec("library_file.cpp")
        )
        args = []
        launch_info = lldb.SBLaunchInfo(args)
        cwd = self.get_process_working_directory()
        launch_info.SetWorkingDirectory(cwd)
        launch_info.SetEnvironmentEntries(['LD_LIBRARY_PATH=' + cwd], True)
        error = lldb.SBError()
        process = target.Launch(launch_info, error)
        self.assertSuccess(error)

        # Stopped on main here. This ensures that we were able to load the
        # main executable and resolve breakpoints within it.
        self.assertState(process.GetState(), lldb.eStateStopped)
        thread = process.GetSelectedThread()
        self.assertIn("main",
                      thread.GetFrameAtIndex(0).GetDisplayFunctionName())
        process.Continue()

        # Make sure we stop next a the library breakpoint. If the dynamic
        # loader doesn't load the library correctly, this breakpoint won't get
        # hit
        self.assertState(process.GetState(), lldb.eStateStopped)
        self.assertIn(
            "library_function",
            thread.GetFrameAtIndex(0).GetDisplayFunctionName()
        )
