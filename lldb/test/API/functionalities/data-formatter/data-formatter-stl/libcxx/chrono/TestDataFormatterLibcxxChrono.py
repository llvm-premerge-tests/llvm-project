"""
Test lldb data formatter subsystem.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LibcxxChronoDataFormatterTestCase(TestBase):
    @add_test_categories(["libc++"])
    def test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        self.build()
        (self.target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp", False)
        )

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd("type format clear", check=False)
            self.runCmd("type summary clear", check=False)
            self.runCmd("type filter clear", check=False)
            self.runCmd("type synth clear", check=False)
            self.runCmd("settings set target.max-children-count 256", check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        # empty vectors (and storage pointers SHOULD BOTH BE NULL..)
        self.expect("frame variable ns", substrs=["ns = 0"])
        self.expect("frame variable us", substrs=["us = 0"])
        self.expect("frame variable ms", substrs=["ms = 0"])
        self.expect("frame variable s", substrs=["s = 0"])
        self.expect("frame variable min", substrs=["min = 0"])
        self.expect("frame variable h", substrs=["h = 0"])

        self.expect("frame variable d", substrs=["d = 0"])
        self.expect("frame variable w", substrs=["w = 0"])
        self.expect("frame variable m", substrs=["m = 0"])
        self.expect("frame variable y", substrs=["y = 0"])

        lldbutil.continue_to_breakpoint(process, bkpt)
        self.expect("frame variable ns", substrs=["ns = 1"])
        self.expect("frame variable us", substrs=["us = 12"])
        self.expect("frame variable ms", substrs=["ms = 123"])
        self.expect("frame variable s", substrs=["s = 1234"])
        self.expect("frame variable min", substrs=["min = 12345"])
        self.expect("frame variable h", substrs=["h = 123456"])

        self.expect("frame variable d", substrs=["d = 654321"])
        self.expect("frame variable w", substrs=["w = 54321"])
        self.expect("frame variable m", substrs=["m = 4321"])
        self.expect("frame variable y", substrs=["y = 321"])

