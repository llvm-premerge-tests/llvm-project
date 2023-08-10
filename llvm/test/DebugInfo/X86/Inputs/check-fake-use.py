# Parsing dwarfdump's output to determine whether the location list for the
# parameter "b" covers all of the function. The script is written in form of a
# state machine and expects that dwarfdump output adheres to a certain order:
# 1) The .debug_info section must appear before the .debug_loc section.
# 2) The DW_AT_location attribute must appear before the parameter's name in the
#    formal parameter DIE.
#
import re
import sys

DebugInfoPattern = r"\.debug_info contents:"
SubprogramPattern = r"^0x[0-9a-f]+:\s+DW_TAG_subprogram"
HighPCPattern = r"DW_AT_high_pc.*0x([0-9a-f]+)"
FormalPattern = r"^0x[0-9a-f]+:\s+DW_TAG_formal_parameter"
LocationPattern = r"DW_AT_location\s+\[DW_FORM_sec_offset\].*0x([a-f0-9]+)"
DebugLocPattern = r'\[0x([a-f0-9]+),\s+0x([a-f0-9]+)\) ".text":'

# States
LookingForDebugInfo = 0
LookingForSubProgram = LookingForDebugInfo + 1  # 1
LookingForHighPC = LookingForSubProgram + 1  # 2
LookingForFormal = LookingForHighPC + 1  # 3
LookingForLocation = LookingForFormal + 1  # 4
DebugLocations = LookingForLocation + 1  # 5
AllDone = DebugLocations + 1  # 6

# For each state, the state table contains 3-item sublists with the following
# entries:
# 1) The regex pattern we use in each state.
# 2) The state we enter when we have a successful match for the current pattern.
# 3) The state we enter when we do not have a successful match for the
#    current pattern.
StateTable = [
    # LookingForDebugInfo
    [DebugInfoPattern, LookingForSubProgram, LookingForDebugInfo],
    # LookingForSubProgram
    [SubprogramPattern, LookingForHighPC, LookingForSubProgram],
    # LookingForHighPC
    [HighPCPattern, LookingForFormal, LookingForHighPC],
    # LookingForFormal
    [FormalPattern, LookingForLocation, LookingForFormal],
    # LookingForLocation
    [LocationPattern, DebugLocations, LookingForFormal],
    # DebugLocations
    [DebugLocPattern, DebugLocations, AllDone],
    # AllDone
    [None, AllDone, AllDone],
]

# Symbolic indices
StatePattern = 0
NextState = 1
FailState = 2

State = LookingForDebugInfo
FirstBeginOffset = -1

# Read output from file provided as command arg
with open(sys.argv[1], "r") as dwarf_dump_file:
    for line in dwarf_dump_file:
        if State == AllDone:
            break
        Pattern = StateTable[State][StatePattern]
        # print "State: %d - Searching '%s' for '%s'" % (State, line, Pattern)
        m = re.search(Pattern, line)
        if m:
            # Match. Depending on the state, we extract various values.
            if State == LookingForHighPC:
                HighPC = int(m.group(1), 16)
            elif State == DebugLocations:
                # Extract the range values
                if FirstBeginOffset == -1:
                    FirstBeginOffset = int(m.group(1), 16)
                    # print "FirstBeginOffset set to %d" % FirstBeginOffset
                EndOffset = int(m.group(2), 16)
                # print "EndOffset set to %d" % EndOffset
            State = StateTable[State][NextState]
        else:
            State = StateTable[State][FailState]

Success = True

# Check that the first entry start with 0 and that the last ending address
# in our location list is close to the high pc of the subprogram.
if State != AllDone:
    print("Error in expected sequence of DWARF information:")
    print(" State = %d\n" % State)
    Success = False
elif FirstBeginOffset == -1:
    print("Location list for 'b' not found, did the debug info format change?")
    Success = False
elif FirstBeginOffset != 0 or abs(EndOffset - HighPC) > 16:
    print("Location list for 'b' does not cover the whole function:")
    print(
        "Location starts at 0x%x, ends at 0x%x, HighPC = 0x%x"
        % (FirstBeginOffset, EndOffset, HighPC)
    )
    Success = False

sys.exit(not Success)
