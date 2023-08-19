//===- bolt/Rewrite/RewriteInstance.cpp - ELF rewriter --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Rewrite/RewriteInstance.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryEmitter.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/BinarySection.h"
#include "bolt/Core/DebugData.h"
#include "bolt/Core/Exceptions.h"
#include "bolt/Core/FunctionLayout.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "bolt/Core/ParallelUtilities.h"
#include "bolt/Core/Relocation.h"
#include "bolt/Passes/CacheMetrics.h"
#include "bolt/Passes/ReorderFunctions.h"
#include "bolt/Profile/BoltAddressTranslation.h"
#include "bolt/Profile/DataAggregator.h"
#include "bolt/Profile/DataReader.h"
#include "bolt/Profile/YAMLProfileReader.h"
#include "bolt/Profile/YAMLProfileWriter.h"
#include "bolt/Rewrite/BinaryPassManager.h"
#include "bolt/Rewrite/DWARFRewriter.h"
#include "bolt/Rewrite/ExecutableFileMemoryManager.h"
#include "bolt/Rewrite/JITLinkLinker.h"
#include "bolt/Rewrite/MetadataRewriters.h"
#include "bolt/RuntimeLibs/HugifyRuntimeLibrary.h"
#include "bolt/RuntimeLibs/InstrumentationRuntimeLibrary.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "bolt/Utils/Utils.h"
#include "llvm/ADT/AddressRanges.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugFrame.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <system_error>

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt"

using namespace llvm;
using namespace object;
using namespace bolt;

extern cl::opt<uint32_t> X86AlignBranchBoundary;
extern cl::opt<bool> X86AlignBranchWithin32BBoundaries;

namespace opts {

extern cl::opt<MacroFusionType> AlignMacroOpFusion;
extern cl::list<std::string> HotTextMoveSections;
extern cl::opt<bool> Hugify;
extern cl::opt<bool> Instrument;
extern cl::opt<JumpTableSupportLevel> JumpTables;
extern cl::list<std::string> ReorderData;
extern cl::opt<bolt::ReorderFunctions::ReorderType> ReorderFunctions;
extern cl::opt<bool> TimeBuild;

cl::opt<bool> AllowStripped("allow-stripped",
                            cl::desc("allow processing of stripped binaries"),
                            cl::Hidden, cl::cat(BoltCategory));

static cl::opt<bool> ForceToDataRelocations(
    "force-data-relocations",
    cl::desc("force relocations to data sections to always be processed"),

    cl::Hidden, cl::cat(BoltCategory));

cl::opt<std::string>
    BoltID("bolt-id",
           cl::desc("add any string to tag this execution in the "
                    "output binary via bolt info section"),
           cl::cat(BoltCategory));

cl::opt<bool> DumpDotAll(
    "dump-dot-all",
    cl::desc("dump function CFGs to graphviz format after each stage;"
             "enable '-print-loops' for color-coded blocks"),
    cl::Hidden, cl::cat(BoltCategory));

static cl::list<std::string>
ForceFunctionNames("funcs",
  cl::CommaSeparated,
  cl::desc("limit optimizations to functions from the list"),
  cl::value_desc("func1,func2,func3,..."),
  cl::Hidden,
  cl::cat(BoltCategory));

static cl::opt<std::string>
FunctionNamesFile("funcs-file",
  cl::desc("file with list of functions to optimize"),
  cl::Hidden,
  cl::cat(BoltCategory));

static cl::list<std::string> ForceFunctionNamesNR(
    "funcs-no-regex", cl::CommaSeparated,
    cl::desc("limit optimizations to functions from the list (non-regex)"),
    cl::value_desc("func1,func2,func3,..."), cl::Hidden, cl::cat(BoltCategory));

static cl::opt<std::string> FunctionNamesFileNR(
    "funcs-file-no-regex",
    cl::desc("file with list of functions to optimize (non-regex)"), cl::Hidden,
    cl::cat(BoltCategory));

cl::opt<bool>
KeepTmp("keep-tmp",
  cl::desc("preserve intermediate .o file"),
  cl::Hidden,
  cl::cat(BoltCategory));

cl::opt<bool> Lite("lite", cl::desc("skip processing of cold functions"),
                   cl::cat(BoltCategory));

static cl::opt<unsigned>
LiteThresholdPct("lite-threshold-pct",
  cl::desc("threshold (in percent) for selecting functions to process in lite "
            "mode. Higher threshold means fewer functions to process. E.g "
            "threshold of 90 means only top 10 percent of functions with "
            "profile will be processed."),
  cl::init(0),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltOptCategory));

static cl::opt<unsigned> LiteThresholdCount(
    "lite-threshold-count",
    cl::desc("similar to '-lite-threshold-pct' but specify threshold using "
             "absolute function call count. I.e. limit processing to functions "
             "executed at least the specified number of times."),
    cl::init(0), cl::Hidden, cl::cat(BoltOptCategory));

static cl::opt<unsigned>
    MaxFunctions("max-funcs",
                 cl::desc("maximum number of functions to process"), cl::Hidden,
                 cl::cat(BoltCategory));

static cl::opt<unsigned> MaxDataRelocations(
    "max-data-relocations",
    cl::desc("maximum number of data relocations to process"), cl::Hidden,
    cl::cat(BoltCategory));

cl::opt<bool> PrintAll("print-all",
                       cl::desc("print functions after each stage"), cl::Hidden,
                       cl::cat(BoltCategory));

cl::opt<bool> PrintProfile("print-profile",
                           cl::desc("print functions after attaching profile"),
                           cl::Hidden, cl::cat(BoltCategory));

cl::opt<bool> PrintCFG("print-cfg",
                       cl::desc("print functions after CFG construction"),
                       cl::Hidden, cl::cat(BoltCategory));

cl::opt<bool> PrintDisasm("print-disasm",
                          cl::desc("print function after disassembly"),
                          cl::Hidden, cl::cat(BoltCategory));

static cl::opt<bool>
    PrintGlobals("print-globals",
                 cl::desc("print global symbols after disassembly"), cl::Hidden,
                 cl::cat(BoltCategory));

extern cl::opt<bool> PrintSections;

static cl::opt<bool> PrintLoopInfo("print-loops",
                                   cl::desc("print loop related information"),
                                   cl::Hidden, cl::cat(BoltCategory));

static cl::opt<cl::boolOrDefault> RelocationMode(
    "relocs", cl::desc("use relocations in the binary (default=autodetect)"),
    cl::cat(BoltCategory));

static cl::opt<std::string>
SaveProfile("w",
  cl::desc("save recorded profile to a file"),
  cl::cat(BoltOutputCategory));

static cl::list<std::string>
SkipFunctionNames("skip-funcs",
  cl::CommaSeparated,
  cl::desc("list of functions to skip"),
  cl::value_desc("func1,func2,func3,..."),
  cl::Hidden,
  cl::cat(BoltCategory));

static cl::opt<std::string>
SkipFunctionNamesFile("skip-funcs-file",
  cl::desc("file with list of functions to skip"),
  cl::Hidden,
  cl::cat(BoltCategory));

cl::opt<bool>
TrapOldCode("trap-old-code",
  cl::desc("insert traps in old function bodies (relocation mode)"),
  cl::Hidden,
  cl::cat(BoltCategory));

static cl::opt<std::string> DWPPathName("dwp",
                                        cl::desc("Path and name to DWP file."),
                                        cl::Hidden, cl::init(""),
                                        cl::cat(BoltCategory));

static cl::opt<bool>
    UseGnuStack("use-gnu-stack",
                cl::desc("deprecated and not supported anymore: use GNU_STACK "
                         "program header for new segment (workaround for "
                         "issues with strip/objcopy)"),
                cl::ZeroOrMore, cl::cat(BoltCategory),
                cl::callback([](const bool &Option) {
                  errs() << "BOLT-WARNING: use-gnu-stack is deprecated, please "
                            "use -rewrite to "
                            "avoid stripping issues\n";
                }));

static cl::opt<bool>
    TimeRewrite("time-rewrite",
                cl::desc("print time spent in rewriting passes"), cl::Hidden,
                cl::cat(BoltCategory));

static cl::opt<bool>
SequentialDisassembly("sequential-disassembly",
  cl::desc("performs disassembly sequentially"),
  cl::init(false),
  cl::cat(BoltOptCategory));

static cl::opt<bool> WriteBoltInfoSection(
    "bolt-info", cl::desc("write bolt info section in the output binary"),
    cl::init(true), cl::Hidden, cl::cat(BoltOutputCategory));

} // namespace opts

// FIXME: implement a better way to mark sections for replacement.
constexpr const char *RewriteInstance::SectionsToOverwrite[];
std::vector<std::string> RewriteInstance::DebugSectionsToOverwrite = {
    ".debug_abbrev", ".debug_aranges",  ".debug_line",   ".debug_line_str",
    ".debug_loc",    ".debug_loclists", ".debug_ranges", ".debug_rnglists",
    ".gdb_index",    ".debug_addr",     ".debug_abbrev", ".debug_info",
    ".debug_types",  ".pseudo_probe"};

const char RewriteInstance::TimerGroupName[] = "rewrite";
const char RewriteInstance::TimerGroupDesc[] = "Rewrite passes";

namespace llvm {
namespace bolt {

extern const char *BoltRevision;

MCPlusBuilder *createMCPlusBuilder(const Triple::ArchType Arch,
                                   const MCInstrAnalysis *Analysis,
                                   const MCInstrInfo *Info,
                                   const MCRegisterInfo *RegInfo) {
#ifdef X86_AVAILABLE
  if (Arch == Triple::x86_64)
    return createX86MCPlusBuilder(Analysis, Info, RegInfo);
#endif

#ifdef AARCH64_AVAILABLE
  if (Arch == Triple::aarch64)
    return createAArch64MCPlusBuilder(Analysis, Info, RegInfo);
#endif

#ifdef RISCV_AVAILABLE
  if (Arch == Triple::riscv64)
    return createRISCVMCPlusBuilder(Analysis, Info, RegInfo);
#endif

  llvm_unreachable("architecture unsupported by MCPlusBuilder");
}

} // namespace bolt
} // namespace llvm

using ELF64LEPhdrTy = ELF64LEFile::Elf_Phdr;

namespace {

bool refersToReorderedSection(ErrorOr<BinarySection &> Section) {
  return llvm::any_of(opts::ReorderData, [&](const std::string &SectionName) {
    return Section && Section->getName() == SectionName;
  });
}

} // anonymous namespace

Expected<std::unique_ptr<RewriteInstance>>
RewriteInstance::create(ELFObjectFileBase *File, const int Argc,
                        const char *const *Argv, StringRef ToolPath) {
  Error Err = Error::success();
  auto RI = std::make_unique<RewriteInstance>(File, Argc, Argv, ToolPath, Err);
  if (Err)
    return std::move(Err);
  return std::move(RI);
}

RewriteInstance::RewriteInstance(ELFObjectFileBase *File, const int Argc,
                                 const char *const *Argv, StringRef ToolPath,
                                 Error &Err)
    : InputFile(File), Argc(Argc), Argv(Argv), ToolPath(ToolPath),
      SHStrTab(StringTableBuilder::ELF) {
  ErrorAsOutParameter EAO(&Err);
  auto ELF64LEFile = dyn_cast<ELF64LEObjectFile>(InputFile);
  if (!ELF64LEFile) {
    Err = createStringError(errc::not_supported,
                            "Only 64-bit LE ELF binaries are supported");
    return;
  }

  bool IsPIC = false;
  const ELFFile<ELF64LE> &Obj = ELF64LEFile->getELFFile();
  if (Obj.getHeader().e_type != ELF::ET_EXEC) {
    outs() << "BOLT-INFO: shared object or position-independent executable "
              "detected\n";
    IsPIC = true;
  }

  auto BCOrErr = BinaryContext::createBinaryContext(
      File, IsPIC,
      DWARFContext::create(*File, DWARFContext::ProcessDebugRelocations::Ignore,
                           nullptr, opts::DWPPathName,
                           WithColor::defaultErrorHandler,
                           WithColor::defaultWarningHandler));
  if (Error E = BCOrErr.takeError()) {
    Err = std::move(E);
    return;
  }
  BC = std::move(BCOrErr.get());
  BC->initializeTarget(std::unique_ptr<MCPlusBuilder>(createMCPlusBuilder(
      BC->TheTriple->getArch(), BC->MIA.get(), BC->MII.get(), BC->MRI.get())));

  BAT = std::make_unique<BoltAddressTranslation>();

  if (opts::UpdateDebugSections)
    DebugInfoRewriter = std::make_unique<DWARFRewriter>(*BC);

  if (opts::Instrument)
    BC->setRuntimeLibrary(std::make_unique<InstrumentationRuntimeLibrary>());
  else if (opts::Hugify)
    BC->setRuntimeLibrary(std::make_unique<HugifyRuntimeLibrary>());
}

RewriteInstance::~RewriteInstance() {}

Error RewriteInstance::setProfile(StringRef Filename) {
  if (!sys::fs::exists(Filename))
    return errorCodeToError(make_error_code(errc::no_such_file_or_directory));

  if (ProfileReader) {
    // Already exists
    return make_error<StringError>(Twine("multiple profiles specified: ") +
                                       ProfileReader->getFilename() + " and " +
                                       Filename,
                                   inconvertibleErrorCode());
  }

  // Spawn a profile reader based on file contents.
  if (DataAggregator::checkPerfDataMagic(Filename))
    ProfileReader = std::make_unique<DataAggregator>(Filename);
  else if (YAMLProfileReader::isYAML(Filename))
    ProfileReader = std::make_unique<YAMLProfileReader>(Filename);
  else
    ProfileReader = std::make_unique<DataReader>(Filename);

  return Error::success();
}

/// Return true if the function \p BF should be disassembled.
static bool shouldDisassemble(const BinaryFunction &BF) {
  if (BF.isPseudo())
    return false;

  if (opts::processAllFunctions())
    return true;

  return !BF.isIgnored();
}

// Return if a section stored in the image falls into a segment address space.
// If not, Set \p Overlap to true if there's a partial overlap.
template <class ELFT>
static bool checkOffsets(const typename ELFT::Phdr &Phdr,
                         const typename ELFT::Shdr &Sec, bool &Overlap) {
  // SHT_NOBITS sections don't need to have an offset inside the segment.
  if (Sec.sh_type == ELF::SHT_NOBITS)
    return true;

  // Only non-empty sections can be at the end of a segment.
  uint64_t SectionSize = Sec.sh_size ? Sec.sh_size : 1;
  AddressRange SectionAddressRange(Sec.sh_offset, Sec.sh_offset + SectionSize);
  AddressRange SegmentAddressRange(Phdr.p_offset,
                                   Phdr.p_offset + Phdr.p_filesz);
  if (SegmentAddressRange.contains(SectionAddressRange))
    return true;

  Overlap = SegmentAddressRange.intersects(SectionAddressRange);
  return false;
}

// Check that an allocatable section belongs to a virtual address
// space of a segment.
template <class ELFT>
static bool checkVMA(const typename ELFT::Phdr &Phdr,
                     const typename ELFT::Shdr &Sec, bool &Overlap) {
  // Only non-empty sections can be at the end of a segment.
  uint64_t SectionSize = Sec.sh_size ? Sec.sh_size : 1;
  AddressRange SectionAddressRange(Sec.sh_addr, Sec.sh_addr + SectionSize);
  AddressRange SegmentAddressRange(Phdr.p_vaddr, Phdr.p_vaddr + Phdr.p_memsz);

  if (SegmentAddressRange.contains(SectionAddressRange))
    return true;
  Overlap = SegmentAddressRange.intersects(SectionAddressRange);
  return false;
}

void RewriteInstance::markGnuRelroSections() {
  using ELFT = ELF64LE;
  using ELFShdrTy = typename ELFObjectFile<ELFT>::Elf_Shdr;
  auto ELF64LEFile = cast<ELF64LEObjectFile>(InputFile);
  const ELFFile<ELFT> &Obj = ELF64LEFile->getELFFile();

  auto handleSection = [&](const ELFT::Phdr &Phdr, SectionRef SecRef) {
    BinarySection *BinarySection = BC->getSectionForSectionRef(SecRef);
    // If the section is non-allocatable, ignore it for GNU_RELRO purposes:
    // it can't be made read-only after runtime relocations processing.
    if (!BinarySection || !BinarySection->isAllocatable())
      return;
    const ELFShdrTy *Sec = cantFail(Obj.getSection(SecRef.getIndex()));
    bool ImageOverlap{false}, VMAOverlap{false};
    bool ImageContains = checkOffsets<ELFT>(Phdr, *Sec, ImageOverlap);
    bool VMAContains = checkVMA<ELFT>(Phdr, *Sec, VMAOverlap);
    if (ImageOverlap) {
      if (opts::Verbosity >= 1)
        errs() << "BOLT-WARNING: GNU_RELRO segment has partial file offset "
               << "overlap with section " << BinarySection->getName() << '\n';
      return;
    }
    if (VMAOverlap) {
      if (opts::Verbosity >= 1)
        errs() << "BOLT-WARNING: GNU_RELRO segment has partial VMA overlap "
               << "with section " << BinarySection->getName() << '\n';
      return;
    }
    if (!ImageContains || !VMAContains)
      return;
    BinarySection->setRelro();
    if (opts::Verbosity >= 1)
      outs() << "BOLT-INFO: marking " << BinarySection->getName()
             << " as GNU_RELRO\n";
  };

  for (const ELFT::Phdr &Phdr : cantFail(Obj.program_headers()))
    if (Phdr.p_type == ELF::PT_GNU_RELRO)
      for (SectionRef SecRef : InputFile->sections())
        handleSection(Phdr, SecRef);
}

Error RewriteInstance::discoverStorage() {
  NamedRegionTimer T("discoverStorage", "discover storage", TimerGroupName,
                     TimerGroupDesc, opts::TimeRewrite);

  auto ELF64LEFile = cast<ELF64LEObjectFile>(InputFile);
  const ELFFile<ELF64LE> &Obj = ELF64LEFile->getELFFile();

  BC->StartFunctionAddress = Obj.getHeader().e_entry;

  Expected<ELF64LE::PhdrRange> PHsOrErr = Obj.program_headers();
  if (Error E = PHsOrErr.takeError())
    return E;

  ELF64LE::PhdrRange PHs = PHsOrErr.get();
  unsigned Phnum = PHs.end() - PHs.begin();
  uint64_t FirstAllocOffset = ~0ULL;
  uint64_t EndOfLoadablePartOffset = 0;
  for (const ELF64LE::Phdr &Phdr : PHs) {
    switch (Phdr.p_type) {
    case ELF::PT_LOAD:
      BC->FirstAllocAddress = std::min(BC->FirstAllocAddress,
                                       static_cast<uint64_t>(Phdr.p_vaddr));
      FirstAllocOffset =
          std::min(FirstAllocOffset, static_cast<uint64_t>(Phdr.p_offset));
      BC->InputAddressSpaceEnd =
          std::max(BC->InputAddressSpaceEnd,
                   alignTo(Phdr.p_vaddr + Phdr.p_memsz, BC->PageAlign));
      EndOfLoadablePartOffset =
          std::max(EndOfLoadablePartOffset,
                   alignTo(Phdr.p_offset + Phdr.p_memsz, BC->PageAlign));
      BC->SegmentMapInfo[Phdr.p_vaddr] = ProgramHeader(Phdr);
      if (Phdr.p_flags == ELF::PF_R)
        HasReadOnlySegment = true;
      break;
    case ELF::PT_INTERP:
      BC->HasInterpHeader = true;
      break;
    case ELF::PT_PHDR:
      HasProgramHeaderSegment = true;
      break;
    }
    BC->InputSegments.push_back(Phdr);
  }

  // Address of a first byte in a first segment (usually beginning of a file)
  BaseAddress = BC->FirstAllocAddress - FirstAllocOffset;
  for (const SectionRef &Section : InputFile->sections()) {
    Expected<StringRef> SectionNameOrErr = Section.getName();
    if (Error E = SectionNameOrErr.takeError())
      return E;
    StringRef SectionName = SectionNameOrErr.get();
    if (SectionName == ".text") {
      BC->OldTextSectionAddress = Section.getAddress();
    }

    if (!opts::HeatmapMode &&
        !(opts::AggregateOnly && BAT->enabledFor(InputFile)) &&
        (SectionName.startswith(getOrgSecPrefix()) ||
         SectionName == getBOLTTextSectionName()))
      return createStringError(
          errc::function_not_supported,
          "BOLT-ERROR: input file was processed by BOLT. Cannot re-optimize");
  }

  outs() << "BOLT-INFO: first alloc address is 0x"
         << Twine::utohexstr(BC->FirstAllocAddress) << '\n';

  if (!opts::Rewrite && !opts::UseOldText) {
    // when not rewriting, compute the new text section and segment addresses in
    // advance. Needed for LongJMP.
    FirstNonAllocatableOffset = EndOfLoadablePartOffset;
    uint64_t NextAvailableOffset = EndOfLoadablePartOffset;
    NextAvailableAddress = BC->InputAddressSpaceEnd;
    if (NextAvailableOffset <= NextAvailableAddress - BC->FirstAllocAddress)
      NextAvailableOffset = NextAvailableAddress - BC->FirstAllocAddress;
    else
      NextAvailableAddress = NextAvailableOffset + BC->FirstAllocAddress;

    assert(NextAvailableOffset ==
               NextAvailableAddress - BC->FirstAllocAddress &&
           "PHDR table address calculation error");

    BC->OutputAddressToOffsetMap[NextAvailableAddress] = NextAvailableOffset;

    unsigned MaxPhnum = Phnum + !HasProgramHeaderSegment + 2;

    BC->MaxPHDRSize = MaxPhnum * sizeof(ELF64LE::Phdr);

    PHDRTableAddress = NextAvailableAddress;
    PHDRTableOffset = NextAvailableOffset;
    NextAvailableAddress += BC->MaxPHDRSize;
    NextAvailableOffset += BC->MaxPHDRSize;

    NextAvailableAddress = alignTo(
        NextAvailableAddress, std::max(opts::AlignText, opts::AlignFunctions));
    BC->NewTextSectionAddress = NextAvailableAddress;
    outs() << "BOLT-INFO: creating new program header table at address 0x"
           << Twine::utohexstr(PHDRTableAddress) << ", offset 0x"
           << Twine::utohexstr(PHDRTableOffset) << '\n';
  }
  // Tools such as objcopy can strip section contents but leave header
  // entries. Check that at least .text is mapped in the file.
  if (!getFileOffsetForAddress(BC->OldTextSectionAddress))
    return createStringError(errc::executable_format_error,
                             "BOLT-ERROR: input binary is not a valid ELF "
                             "executable as its text section is not "
                             "mapped to a valid segment");
  return Error::success();
}

void RewriteInstance::parseBuildID() {
  if (!BuildIDSection)
    return;

  StringRef Buf = BuildIDSection->getContents();

  // Reading notes section (see Portable Formats Specification, Version 1.1,
  // pg 2-5, section "Note Section").
  DataExtractor DE = DataExtractor(Buf, true, 8);
  uint64_t Offset = 0;
  if (!DE.isValidOffset(Offset))
    return;
  uint32_t NameSz = DE.getU32(&Offset);
  if (!DE.isValidOffset(Offset))
    return;
  uint32_t DescSz = DE.getU32(&Offset);
  if (!DE.isValidOffset(Offset))
    return;
  uint32_t Type = DE.getU32(&Offset);

  LLVM_DEBUG(dbgs() << "NameSz = " << NameSz << "; DescSz = " << DescSz
                    << "; Type = " << Type << "\n");

  // Type 3 is a GNU build-id note section
  if (Type != 3)
    return;

  StringRef Name = Buf.slice(Offset, Offset + NameSz);
  Offset = alignTo(Offset + NameSz, 4);
  if (Name.substr(0, 3) != "GNU")
    return;

  BuildID = Buf.slice(Offset, Offset + DescSz);
}

std::optional<std::string> RewriteInstance::getPrintableBuildID() const {
  if (BuildID.empty())
    return std::nullopt;

  std::string Str;
  raw_string_ostream OS(Str);
  const unsigned char *CharIter = BuildID.bytes_begin();
  while (CharIter != BuildID.bytes_end()) {
    if (*CharIter < 0x10)
      OS << "0";
    OS << Twine::utohexstr(*CharIter);
    ++CharIter;
  }
  return OS.str();
}

void RewriteInstance::patchBuildID() {
  raw_fd_ostream &OS = Out->os();

  if (BuildID.empty())
    return;

  size_t IDOffset = BuildIDSection->getContents().rfind(BuildID);
  assert(IDOffset != StringRef::npos && "failed to patch build-id");

  uint64_t FileOffset =
      getOutputFileOffsetForAddress(BuildIDSection->getOutputAddress());
  if (!FileOffset) {
    errs() << "BOLT-WARNING: Non-allocatable build-id will not be updated.\n";
    return;
  }

  char LastIDByte = BuildID[BuildID.size() - 1];
  LastIDByte ^= 1;
  OS.pwrite(&LastIDByte, 1, FileOffset + IDOffset + BuildID.size() - 1);

  outs() << "BOLT-INFO: patched build-id (flipped last bit)\n";
}

Error RewriteInstance::run() {
  assert(BC && "failed to create a binary context");

  outs() << "BOLT-INFO: Target architecture: "
         << Triple::getArchTypeName(
                (llvm::Triple::ArchType)InputFile->getArch())
         << "\n";
  outs() << "BOLT-INFO: BOLT version: " << BoltRevision << "\n";

  if (Error E = discoverStorage())
    return E;
  if (Error E = readSpecialSections())
    return E;
  adjustCommandLineOptions();
  discoverFileObjects();
  preprocessProfileData();

  // Skip disassembling if we have a translation table and we are running an
  // aggregation job.
  if (opts::AggregateOnly && BAT->enabledFor(InputFile)) {
    processProfileData();
    return Error::success();
  }

  selectFunctionsToProcess();

  readDebugInfo();

  disassembleFunctions();

  processMetadataPreCFG();

  buildFunctionsCFG();

  processProfileData();

  postProcessFunctions();

  processMetadataPostCFG();

  if (opts::DiffOnly)
    return Error::success();

  renameAndPreregisterSections();

  runOptimizationPasses();

  emitAndLink();

  updateMetadata();

  if (opts::LinuxKernelMode) {
    errs() << "BOLT-WARNING: not writing the output file for Linux Kernel\n";
    return Error::success();
  } else if (opts::OutputFilename == "/dev/null") {
    outs() << "BOLT-INFO: skipping writing final binary to disk\n";
    return Error::success();
  }

  // Rewrite allocatable contents and copy non-allocatable parts with mods.
  rewriteFile();
  return Error::success();
}

void RewriteInstance::discoverFileObjects() {
  NamedRegionTimer T("discoverFileObjects", "discover file objects",
                     TimerGroupName, TimerGroupDesc, opts::TimeRewrite);
  FileSymRefs.clear();
  BC->getBinaryFunctions().clear();
  BC->clearBinaryData();

  // For local symbols we want to keep track of associated FILE symbol name for
  // disambiguation by combined name.
  StringRef FileSymbolName;
  bool SeenFileName = false;
  struct SymbolRefHash {
    size_t operator()(SymbolRef const &S) const {
      return std::hash<decltype(DataRefImpl::p)>{}(S.getRawDataRefImpl().p);
    }
  };
  std::unordered_map<SymbolRef, StringRef, SymbolRefHash> SymbolToFileName;
  for (const ELFSymbolRef &Symbol : InputFile->symbols()) {
    Expected<StringRef> NameOrError = Symbol.getName();
    if (NameOrError && NameOrError->startswith("__asan_init")) {
      errs() << "BOLT-ERROR: input file was compiled or linked with sanitizer "
                "support. Cannot optimize.\n";
      exit(1);
    }
    if (NameOrError && NameOrError->startswith("__llvm_coverage_mapping")) {
      errs() << "BOLT-ERROR: input file was compiled or linked with coverage "
                "support. Cannot optimize.\n";
      exit(1);
    }

    if (cantFail(Symbol.getFlags()) & SymbolRef::SF_Undefined)
      continue;

    if (cantFail(Symbol.getType()) == SymbolRef::ST_File) {
      StringRef Name =
          cantFail(std::move(NameOrError), "cannot get symbol name for file");
      // Ignore Clang LTO artificial FILE symbol as it is not always generated,
      // and this uncertainty is causing havoc in function name matching.
      if (Name == "ld-temp.o")
        continue;
      FileSymbolName = Name;
      SeenFileName = true;
      continue;
    }
    if (!FileSymbolName.empty() &&
        !(cantFail(Symbol.getFlags()) & SymbolRef::SF_Global))
      SymbolToFileName[Symbol] = FileSymbolName;
  }

  // Sort symbols in the file by value. Ignore symbols from non-allocatable
  // sections.
  auto isSymbolInMemory = [this](const SymbolRef &Sym) {
    if (cantFail(Sym.getType()) == SymbolRef::ST_File)
      return false;
    if (cantFail(Sym.getFlags()) & SymbolRef::SF_Absolute)
      return true;
    if (cantFail(Sym.getFlags()) & SymbolRef::SF_Undefined)
      return false;
    BinarySection Section(*BC, *cantFail(Sym.getSection()));
    return Section.isAllocatable();
  };
  std::vector<SymbolRef> SortedFileSymbols;
  llvm::copy_if(InputFile->symbols(), std::back_inserter(SortedFileSymbols),
                isSymbolInMemory);
  auto CompareSymbols = [this](const SymbolRef &A, const SymbolRef &B) {
    // Marker symbols have the highest precedence, while
    // SECTIONs have the lowest.
    auto AddressA = cantFail(A.getAddress());
    auto AddressB = cantFail(B.getAddress());
    if (AddressA != AddressB)
      return AddressA < AddressB;

    bool AMarker = BC->isMarker(A);
    bool BMarker = BC->isMarker(B);
    if (AMarker || BMarker) {
      return AMarker && !BMarker;
    }

    auto AType = cantFail(A.getType());
    auto BType = cantFail(B.getType());
    if (AType == SymbolRef::ST_Function && BType != SymbolRef::ST_Function)
      return true;
    if (BType == SymbolRef::ST_Debug && AType != SymbolRef::ST_Debug)
      return true;

    return false;
  };

  llvm::stable_sort(SortedFileSymbols, CompareSymbols);

  auto LastSymbol = SortedFileSymbols.end();
  if (!SortedFileSymbols.empty())
    --LastSymbol;

  // For aarch64, the ABI defines mapping symbols so we identify data in the
  // code section (see IHI0056B). $d identifies data contents.
  // Compilers usually merge multiple data objects in a single $d-$x interval,
  // but we need every data object to be marked with $d. Because of that we
  // create a vector of MarkerSyms with all locations of data objects.

  struct MarkerSym {
    uint64_t Address;
    MarkerSymType Type;
  };

  std::vector<MarkerSym> SortedMarkerSymbols;
  auto addExtraDataMarkerPerSymbol =
      [this](const std::vector<SymbolRef> &SortedFileSymbols,
             std::vector<MarkerSym> &SortedMarkerSymbols) {
        bool IsData = false;
        uint64_t LastAddr = 0;
        for (auto Sym = SortedFileSymbols.begin();
             Sym < SortedFileSymbols.end(); ++Sym) {
          uint64_t Address = cantFail(Sym->getAddress());
          if (LastAddr == Address) // don't repeat markers
            continue;

          MarkerSymType MarkerType = BC->getMarkerType(*Sym);
          if (MarkerType != MarkerSymType::NONE) {
            SortedMarkerSymbols.push_back(MarkerSym{Address, MarkerType});
            LastAddr = Address;
            IsData = MarkerType == MarkerSymType::DATA;
            continue;
          }

          if (IsData) {
            SortedMarkerSymbols.push_back(
                MarkerSym{cantFail(Sym->getAddress()), MarkerSymType::DATA});
            LastAddr = Address;
          }
        }
      };

  if (BC->isAArch64() || BC->isRISCV()) {
    addExtraDataMarkerPerSymbol(SortedFileSymbols, SortedMarkerSymbols);
    LastSymbol = std::stable_partition(
        SortedFileSymbols.begin(), SortedFileSymbols.end(),
        [this](const SymbolRef &Symbol) { return !BC->isMarker(Symbol); });
    if (!SortedFileSymbols.empty())
      --LastSymbol;
  }

  BinaryFunction *PreviousFunction = nullptr;
  unsigned AnonymousId = 0;

  // Regex object for matching cold fragments.
  Regex ColdFragment(".*\\.cold(\\.[0-9]+)?");

  const auto SortedSymbolsEnd = LastSymbol == SortedFileSymbols.end()
                                    ? LastSymbol
                                    : std::next(LastSymbol);
  for (auto ISym = SortedFileSymbols.begin(); ISym != SortedSymbolsEnd;
       ++ISym) {
    const SymbolRef &Symbol = *ISym;
    // Keep undefined symbols for pretty printing?
    if (cantFail(Symbol.getFlags()) & SymbolRef::SF_Undefined)
      continue;

    const SymbolRef::Type SymbolType = cantFail(Symbol.getType());

    if (SymbolType == SymbolRef::ST_File)
      continue;

    StringRef SymName = cantFail(Symbol.getName(), "cannot get symbol name");
    uint64_t Address =
        cantFail(Symbol.getAddress(), "cannot get symbol address");
    if (Address == 0) {
      if (opts::Verbosity >= 1 && SymbolType == SymbolRef::ST_Function)
        errs() << "BOLT-WARNING: function with 0 address seen\n";
      continue;
    }

    // Ignore input hot markers
    if (SymName == "__hot_start" || SymName == "__hot_end")
      continue;

    FileSymRefs[Address] = Symbol;

    // Skip section symbols that will be registered by disassemblePLT().
    if ((cantFail(Symbol.getType()) == SymbolRef::ST_Debug)) {
      ErrorOr<BinarySection &> BSection = BC->getSectionForAddress(Address);
      if (BSection && getPLTSectionInfo(BSection->getName()))
        continue;
    }

    /// It is possible we are seeing a globalized local. LLVM might treat it as
    /// a local if it has a "private global" prefix, e.g. ".L". Thus we have to
    /// change the prefix to enforce global scope of the symbol.
    std::string Name = SymName.startswith(BC->AsmInfo->getPrivateGlobalPrefix())
                           ? "PG" + std::string(SymName)
                           : std::string(SymName);

    // Disambiguate all local symbols before adding to symbol table.
    // Since we don't know if we will see a global with the same name,
    // always modify the local name.
    //
    // NOTE: the naming convention for local symbols should match
    //       the one we use for profile data.
    std::string UniqueName;
    std::string AlternativeName;
    if (Name.empty()) {
      UniqueName = "ANONYMOUS." + std::to_string(AnonymousId++);
    } else if (cantFail(Symbol.getFlags()) & SymbolRef::SF_Global) {
      if (const BinaryData *BD = BC->getBinaryDataByName(Name)) {
        if (BD->getSize() == ELFSymbolRef(Symbol).getSize() &&
            BD->getAddress() == Address) {
          if (opts::Verbosity > 1)
            errs() << "BOLT-WARNING: ignoring duplicate global symbol " << Name
                   << "\n";
          // Ignore duplicate entry - possibly a bug in the linker
          continue;
        }
        errs() << "BOLT-ERROR: bad input binary, global symbol \"" << Name
               << "\" is not unique\n";
        exit(1);
      }
      UniqueName = Name;
    } else {
      // If we have a local file name, we should create 2 variants for the
      // function name. The reason is that perf profile might have been
      // collected on a binary that did not have the local file name (e.g. as
      // a side effect of stripping debug info from the binary):
      //
      //   primary:     <function>/<id>
      //   alternative: <function>/<file>/<id2>
      //
      // The <id> field is used for disambiguation of local symbols since there
      // could be identical function names coming from identical file names
      // (e.g. from different directories).
      std::string AltPrefix;
      auto SFI = SymbolToFileName.find(Symbol);
      if (SymbolType == SymbolRef::ST_Function && SFI != SymbolToFileName.end())
        AltPrefix = Name + "/" + std::string(SFI->second);

      UniqueName = NR.uniquify(Name);
      if (!AltPrefix.empty())
        AlternativeName = NR.uniquify(AltPrefix);
    }

    uint64_t SymbolSize = ELFSymbolRef(Symbol).getSize();
    uint64_t SymbolAlignment = Symbol.getAlignment();
    unsigned SymbolFlags = cantFail(Symbol.getFlags());

    auto registerName = [&](uint64_t FinalSize) {
      // Register names even if it's not a function, e.g. for an entry point.
      BC->registerNameAtAddress(UniqueName, Address, FinalSize, SymbolAlignment,
                                SymbolFlags);
      if (!AlternativeName.empty())
        BC->registerNameAtAddress(AlternativeName, Address, FinalSize,
                                  SymbolAlignment, SymbolFlags);
    };

    section_iterator Section =
        cantFail(Symbol.getSection(), "cannot get symbol section");
    if (Section == InputFile->section_end() || !Section->getName()) {
      // Could be an absolute symbol. Used on RISC-V for __global_pointer$ so we
      // need to record it to handle relocations against it. For other instances
      // of absolute symbols, we record for pretty printing.
      LLVM_DEBUG(if (opts::Verbosity > 1) {
        dbgs() << "BOLT-INFO: absolute sym " << UniqueName << "\n";
      });
      registerName(SymbolSize);
      continue;
    }

    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: considering symbol " << UniqueName
                      << " for function\n");

    if (SymbolType != SymbolRef::ST_Debug &&
        Address == Section->getAddress() + Section->getSize()) {
      assert(SymbolSize == 0 &&
             "unexpected non-zero sized symbol at end of section");

      if (auto BSec =
              BC->getUniqueSectionByName(cantFail(Section->getName()))) {
        // we don't register name because it can collide with data from the next
        // section on the same address. We'll instead create local MCSymbols
        // when we encounter relocations against such symbols
        BC->EndSymbols[Name] = &*BSec;
        LLVM_DEBUG(dbgs() << formatv("BOLT-INFO: {0} is in the end of {1}\n",
                                     Name, BSec->getName()));
      } else {
        LLVM_DEBUG(
            dbgs()
            << "BOLT-INFO: rejecting as symbol points to end of its section\n");
      }
      continue;
    }

    if (!Section->isText()) {
      assert(SymbolType != SymbolRef::ST_Function &&
             "unexpected function inside non-code section");
      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: rejecting as symbol is not in code\n");
      registerName(SymbolSize);
      continue;
    }

    // Assembly functions could be ST_NONE with 0 size. Check that the
    // corresponding section is a code section and they are not inside any
    // other known function to consider them.
    //
    // Sometimes assembly functions are not marked as functions and neither are
    // their local labels. The only way to tell them apart is to look at
    // symbol scope - global vs local.
    if (PreviousFunction && SymbolType != SymbolRef::ST_Function) {
      if (PreviousFunction->containsAddress(Address)) {
        if (PreviousFunction->isSymbolValidInScope(Symbol, SymbolSize)) {
          LLVM_DEBUG(dbgs()
                     << "BOLT-DEBUG: symbol is a function local symbol\n");
        } else if (Address == PreviousFunction->getAddress() && !SymbolSize) {
          LLVM_DEBUG(dbgs() << "BOLT-DEBUG: ignoring symbol as a marker\n");
        } else if (opts::Verbosity > 1) {
          errs() << "BOLT-WARNING: symbol " << UniqueName
                 << " seen in the middle of function " << *PreviousFunction
                 << ". Could be a new entry.\n";
        }
        registerName(SymbolSize);
        continue;
      } else if (PreviousFunction->getSize() == 0 &&
                 PreviousFunction->isSymbolValidInScope(Symbol, SymbolSize)) {
        LLVM_DEBUG(dbgs() << "BOLT-DEBUG: symbol is a function local symbol\n");
        registerName(SymbolSize);
        continue;
      }
    }

    if (PreviousFunction && PreviousFunction->containsAddress(Address) &&
        PreviousFunction->getAddress() != Address) {
      if (PreviousFunction->isSymbolValidInScope(Symbol, SymbolSize)) {
        if (opts::Verbosity >= 1)
          outs() << "BOLT-INFO: skipping possibly another entry for function "
                 << *PreviousFunction << " : " << UniqueName << '\n';
        registerName(SymbolSize);
      } else {
        outs() << "BOLT-INFO: using " << UniqueName << " as another entry to "
               << "function " << *PreviousFunction << '\n';

        registerName(0);

        PreviousFunction->addEntryPointAtOffset(Address -
                                                PreviousFunction->getAddress());

        // Remove the symbol from FileSymRefs so that we can skip it from
        // in the future.
        auto SI = FileSymRefs.find(Address);
        assert(SI != FileSymRefs.end() && "symbol expected to be present");
        assert(SI->second == Symbol && "wrong symbol found");
        FileSymRefs.erase(SI);
      }
      continue;
    }

    // Checkout for conflicts with function data from FDEs.
    bool IsSimple = true;
    auto FDEI = CFIRdWrt->getFDEs().lower_bound(Address);
    if (FDEI != CFIRdWrt->getFDEs().end()) {
      const dwarf::FDE &FDE = *FDEI->second;
      if (FDEI->first != Address) {
        // There's no matching starting address in FDE. Make sure the previous
        // FDE does not contain this address.
        if (FDEI != CFIRdWrt->getFDEs().begin()) {
          --FDEI;
          const dwarf::FDE &PrevFDE = *FDEI->second;
          uint64_t PrevStart = PrevFDE.getInitialLocation();
          uint64_t PrevLength = PrevFDE.getAddressRange();
          if (Address > PrevStart && Address < PrevStart + PrevLength) {
            errs() << "BOLT-ERROR: function " << UniqueName
                   << " is in conflict with FDE ["
                   << Twine::utohexstr(PrevStart) << ", "
                   << Twine::utohexstr(PrevStart + PrevLength)
                   << "). Skipping.\n";
            IsSimple = false;
          }
        }
      } else if (FDE.getAddressRange() != SymbolSize) {
        if (SymbolSize) {
          // Function addresses match but sizes differ.
          errs() << "BOLT-WARNING: sizes differ for function " << UniqueName
                 << ". FDE : " << FDE.getAddressRange()
                 << "; symbol table : " << SymbolSize << ". Using max size.\n";
        }
        SymbolSize = std::max(SymbolSize, FDE.getAddressRange());
        if (BC->getBinaryDataAtAddress(Address)) {
          BC->setBinaryDataSize(Address, SymbolSize);
        } else {
          LLVM_DEBUG(dbgs() << "BOLT-DEBUG: No BD @ 0x"
                            << Twine::utohexstr(Address) << "\n");
        }
      }
    }

    BinaryFunction *BF = nullptr;
    // Since function may not have yet obtained its real size, do a search
    // using the list of registered functions instead of calling
    // getBinaryFunctionAtAddress().
    auto BFI = BC->getBinaryFunctions().find(Address);
    if (BFI != BC->getBinaryFunctions().end()) {
      BF = &BFI->second;
      // Duplicate the function name. Make sure everything matches before we add
      // an alternative name.
      if (SymbolSize != BF->getSize()) {
        if (opts::Verbosity >= 1) {
          if (SymbolSize && BF->getSize())
            errs() << "BOLT-WARNING: size mismatch for duplicate entries "
                   << *BF << " and " << UniqueName << '\n';
          outs() << "BOLT-INFO: adjusting size of function " << *BF << " old "
                 << BF->getSize() << " new " << SymbolSize << "\n";
        }
        BF->setSize(std::max(SymbolSize, BF->getSize()));
        BC->setBinaryDataSize(Address, BF->getSize());
      }
      BF->addAlternativeName(UniqueName);
    } else {
      ErrorOr<BinarySection &> Section = BC->getSectionForAddress(Address);
      // Skip symbols from invalid sections
      if (!Section) {
        errs() << "BOLT-WARNING: " << UniqueName << " (0x"
               << Twine::utohexstr(Address) << ") does not have any section\n";
        continue;
      }

      // Skip symbols from zero-sized sections.
      if (!Section->getSize())
        continue;

      BF = BC->createBinaryFunction(UniqueName, *Section, Address, SymbolSize);
      if (!IsSimple)
        BF->setSimple(false);
    }

    // Check if it's a cold function fragment.
    if (ColdFragment.match(SymName)) {
      static bool PrintedWarning = false;
      if (!PrintedWarning) {
        PrintedWarning = true;
        errs() << "BOLT-WARNING: split function detected on input : "
               << SymName;
        if (BC->HasRelocations)
          errs() << ". The support is limited in relocation mode\n";
        else
          errs() << '\n';
      }
      BC->HasSplitFunctions = true;
      BF->IsFragment = true;
    }

    if (!AlternativeName.empty())
      BF->addAlternativeName(AlternativeName);

    registerName(SymbolSize);
    PreviousFunction = BF;
  }

  // Read dynamic relocation first as their presence affects the way we process
  // static relocations. E.g. we will ignore a static relocation at an address
  // that is a subject to dynamic relocation processing.
  processDynamicRelocations();

  // Process PLT section.
  disassemblePLT();

  // See if we missed any functions marked by FDE.
  for (const auto &FDEI : CFIRdWrt->getFDEs()) {
    const uint64_t Address = FDEI.first;
    const dwarf::FDE *FDE = FDEI.second;
    const BinaryFunction *BF = BC->getBinaryFunctionAtAddress(Address);
    if (BF)
      continue;

    BF = BC->getBinaryFunctionContainingAddress(Address);
    if (BF) {
      errs() << "BOLT-WARNING: FDE [0x" << Twine::utohexstr(Address) << ", 0x"
             << Twine::utohexstr(Address + FDE->getAddressRange())
             << ") conflicts with function " << *BF << '\n';
      continue;
    }

    if (opts::Verbosity >= 1)
      errs() << "BOLT-WARNING: FDE [0x" << Twine::utohexstr(Address) << ", 0x"
             << Twine::utohexstr(Address + FDE->getAddressRange())
             << ") has no corresponding symbol table entry\n";

    ErrorOr<BinarySection &> Section = BC->getSectionForAddress(Address);
    assert(Section && "cannot get section for address from FDE");
    std::string FunctionName =
        "__BOLT_FDE_FUNCat" + Twine::utohexstr(Address).str();
    BC->createBinaryFunction(FunctionName, *Section, Address,
                             FDE->getAddressRange());
  }

  BC->setHasSymbolsWithFileName(SeenFileName);

  // Now that all the functions were created - adjust their boundaries.
  adjustFunctionBoundaries();

  // Annotate functions with code/data markers in AArch64
  for (auto ISym = SortedMarkerSymbols.begin();
       ISym != SortedMarkerSymbols.end(); ++ISym) {

    auto *BF =
        BC->getBinaryFunctionContainingAddress(ISym->Address, true, true);

    if (!BF) {
      // Stray marker
      continue;
    }
    const auto EntryOffset = ISym->Address - BF->getAddress();
    if (ISym->Type == MarkerSymType::CODE) {
      BF->markCodeAtOffset(EntryOffset);
      continue;
    }
    if (ISym->Type == MarkerSymType::DATA) {
      BF->markDataAtOffset(EntryOffset);
      BC->AddressToConstantIslandMap[ISym->Address] = BF;
      continue;
    }
    llvm_unreachable("Unknown marker");
  }

  if (BC->isAArch64()) {
    // Check for dynamic relocations that might be contained in
    // constant islands.
    for (const BinarySection &Section : BC->allocatableSections()) {
      const uint64_t SectionAddress = Section.getAddress();
      for (const Relocation &Rel : Section.dynamicRelocations()) {
        const uint64_t RelAddress = SectionAddress + Rel.Offset;
        BinaryFunction *BF =
            BC->getBinaryFunctionContainingAddress(RelAddress,
                                                   /*CheckPastEnd*/ false,
                                                   /*UseMaxSize*/ true);
        if (BF) {
          assert(Rel.isRelative() && "Expected relative relocation for island");
          BF->markIslandDynamicRelocationAtAddress(RelAddress);
        }
      }
    }
  }

  if (!opts::LinuxKernelMode) {
    // Read all relocations now that we have binary functions mapped.
    createGOTPLTRelocations();
    processRelocations();
  }

  registerFragments();
}

void RewriteInstance::registerFragments() {
  if (!BC->HasSplitFunctions)
    return;

  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    if (!Function.isFragment())
      continue;
    unsigned ParentsFound = 0;
    for (StringRef Name : Function.getNames()) {
      StringRef BaseName, Suffix;
      std::tie(BaseName, Suffix) = Name.split('/');
      const size_t ColdSuffixPos = BaseName.find(".cold");
      if (ColdSuffixPos == StringRef::npos)
        continue;
      // For cold function with local (foo.cold/1) symbol, prefer a parent with
      // local symbol as well (foo/1) over global symbol (foo).
      std::string ParentName = BaseName.substr(0, ColdSuffixPos).str();
      const BinaryData *BD = BC->getBinaryDataByName(ParentName);
      if (Suffix != "") {
        ParentName.append(Twine("/", Suffix).str());
        const BinaryData *BDLocal = BC->getBinaryDataByName(ParentName);
        if (BDLocal || !BD)
          BD = BDLocal;
      }
      if (!BD) {
        if (opts::Verbosity >= 1)
          outs() << "BOLT-INFO: parent function not found for " << Name << "\n";
        continue;
      }
      const uint64_t Address = BD->getAddress();
      BinaryFunction *BF = BC->getBinaryFunctionAtAddress(Address);
      if (!BF) {
        if (opts::Verbosity >= 1)
          outs() << formatv("BOLT-INFO: parent function not found at {0:x}\n",
                            Address);
        continue;
      }
      BC->registerFragment(Function, *BF);
      ++ParentsFound;
    }
    if (!ParentsFound) {
      errs() << "BOLT-ERROR: parent function not found for " << Function
             << '\n';
      exit(1);
    }
  }
}

BinaryFunction *RewriteInstance::createPLTBinaryFunction(uint64_t TargetAddress,
                                                         uint64_t EntryAddress,
                                                         uint64_t EntrySize) {
  if (!TargetAddress)
    return nullptr;

  auto setPLTSymbol = [&](BinaryFunction *BF, StringRef Name) {
    const unsigned PtrSize = BC->AsmInfo->getCodePointerSize();
    MCSymbol *TargetSymbol = BC->registerNameAtAddress(
        Name.str() + "@GOT", TargetAddress, PtrSize, PtrSize);
    BF->setPLTSymbol(TargetSymbol);
    BF->setPseudo(true);
    BF->setIgnored();
  };

  BinaryFunction *BF = BC->getBinaryFunctionAtAddress(EntryAddress);
  if (BF && BC->isAArch64()) {
    // Handle IFUNC trampoline
    setPLTSymbol(BF, BF->getOneName());
    return BF;
  }

  MCSymbol *Symbol = [&]() -> MCSymbol * {
    const Relocation *Rel = BC->getDynamicRelocationAt(TargetAddress);
    if (Rel && Rel->Symbol)
      return Rel->Symbol;
    if (auto Addr =
            BC->getUnsignedValueAtAddress(TargetAddress, sizeof(uint64_t))) {
      if (auto *Data = BC->getBinaryDataAtAddress(*Addr))
        return Data->getSymbol();
    }
    return nullptr;
  }();

  ErrorOr<BinarySection &> Section = BC->getSectionForAddress(EntryAddress);
  assert(Section && "cannot get section for address");
  if (!Symbol) {
    assert(EntryAddress == Section->getAddress() &&
           "Unknown PLT entry after PLT header");
    if (!BF)
      BF = BC->createBinaryFunction("__BOLT_PSEUDO_" + Section->getName().str(),
                                    *Section, EntryAddress, 0, EntrySize,
                                    Section->getAlignment());
    BF->setPseudo(true);
    BF->setIgnored();
    BF->setPLTSymbol(BC->getOrCreateGlobalSymbol(TargetAddress, "DATAat"));
    return BF;
  }
  if (!BF)
    BF = BC->createBinaryFunction(Symbol->getName().str() + "@PLT", *Section,
                                  EntryAddress, 0, EntrySize,
                                  Section->getAlignment());
  else
    BF->addAlternativeName(Symbol->getName().str() + "@PLT");
  setPLTSymbol(BF, Symbol->getName());
  return BF;
}

void RewriteInstance::disassemblePLTSectionAArch64(BinarySection &Section) {
  const uint64_t SectionAddress = Section.getAddress();
  const uint64_t SectionSize = Section.getSize();
  StringRef PLTContents = Section.getContents();
  ArrayRef<uint8_t> PLTData(
      reinterpret_cast<const uint8_t *>(PLTContents.data()), SectionSize);

  auto disassembleInstruction = [&](uint64_t InstrOffset, MCInst &Instruction,
                                    uint64_t &InstrSize) {
    const uint64_t InstrAddr = SectionAddress + InstrOffset;
    if (!BC->DisAsm->getInstruction(Instruction, InstrSize,
                                    PLTData.slice(InstrOffset), InstrAddr,
                                    nulls())) {
      errs() << "BOLT-ERROR: unable to disassemble instruction in PLT section "
             << Section.getName() << " at offset 0x"
             << Twine::utohexstr(InstrOffset) << '\n';
      exit(1);
    }
  };

  uint64_t InstrOffset = 0;
  // Locate new plt entry
  while (InstrOffset < SectionSize) {
    InstructionListType Instructions;
    MCInst Instruction;
    uint64_t EntryOffset = InstrOffset;
    uint64_t EntrySize = 0;
    uint64_t InstrSize;
    // Loop through entry instructions
    while (InstrOffset < SectionSize) {
      disassembleInstruction(InstrOffset, Instruction, InstrSize);
      EntrySize += InstrSize;
      Instructions.emplace_back(Instruction);
      InstrOffset += InstrSize;
      if (BC->MIB->isIndirectBranch(Instruction)) {
        break;
      }
    }
    const uint64_t EntryAddress = SectionAddress + EntryOffset;
    const uint64_t TargetAddress = BC->MIB->analyzePLTEntry(
        Instructions.begin(), Instructions.end(), EntryAddress);
    BinaryFunction *BF =
        createPLTBinaryFunction(TargetAddress, EntryAddress, EntrySize);
    assert(BF && "Failed to create PLT function");
    if (opts::Rewrite) {
      assert(BF && BF->empty());
      BF->setSize(EntrySize);
      BF->updateState(BinaryFunction::State::Disassembled);
      BinaryBasicBlock *BB = BF->addBasicBlockAt(0, BF->getSymbol());
      for (auto &Inst : Instructions) {
        BB->addInstruction(Inst);
      }
      bool Ok = BF->handlePLT();
      assert(Ok && "Failed to handle PLT entry");
      (void)Ok;
      BF->updateState(BinaryFunction::State::CFG);
    }

    // Skip nops if any
    while (InstrOffset < SectionSize) {
      disassembleInstruction(InstrOffset, Instruction, InstrSize);
      if (!BC->MIB->isNoop(Instruction))
        break;

      InstrOffset += InstrSize;
      if (opts::Rewrite) {
        BF->getBasicBlockAtOffset(0)->addInstruction(Instruction);
        BF->setSize(BF->getSize() + InstrSize);
      }
    }
  }
}

void RewriteInstance::createGOTPLTRelocations() {
  const uint64_t RelType = Relocation::getAbs64();
  auto CreateRelocations = [this, RelType](BinarySection &Section, bool IsGot) {
    DataExtractor DE(Section.getContents(), BC->AsmInfo->isLittleEndian(),
                     BC->AsmInfo->getCodePointerSize());
    const uint64_t Size = Section.getSize();
    assert(Size % BC->AsmInfo->getCodePointerSize() == 0 && "Invalid GOT size");
    for (uint64_t DataOffset = 0; DataOffset < Size;) {
      const uint64_t Offset = DataOffset;
      const uint64_t Address = DE.getU64(&DataOffset);
      if (Address == 0 || Address == ~0ULL)
        continue;
      MCSymbol *Sym = BC->getOrCreateGlobalSymbol(Address, "SYMBOLat");
      Section.addRelocation(Offset, Sym, RelType, 0);
      if (!opts::Rewrite || !IsGot || !BC->isRISC())
        continue;
      if (BinaryData *BD = BC->getBinaryDataAtAddress(Address)) {
        for (auto *Sym : BD->getSymbols()) {
          StringRef Name = Sym->getName();
          Name = Name.substr(0, Name.find('/'));
          if (Name.empty())
            continue;
          GOTSymbolsByName[Name.str()] = Section.getAddress() + Offset;

          LLVM_DEBUG(
              outs() << formatv(
                  "BOLT-INFO: BD {0} with address {1:x} at got entry {2:x}\n",
                  Sym->getName(), Address, Section.getAddress() + Offset););
        }
      }
    }
  };

  auto GOTPLTSection = BC->getUniqueSectionByName(".got.plt");
  if (opts::Rewrite && GOTPLTSection)
    CreateRelocations(*GOTPLTSection, false);
  auto GOTSection = BC->getUniqueSectionByName(".got");
  if (GOTSection)
    CreateRelocations(*GOTSection, true);
}


void RewriteInstance::disassemblePLTSectionRISCV(BinarySection &Section) {
  const uint64_t SectionAddress = Section.getAddress();
  const uint64_t SectionSize = Section.getSize();
  StringRef PLTContents = Section.getContents();
  ArrayRef<uint8_t> PLTData(
      reinterpret_cast<const uint8_t *>(PLTContents.data()), SectionSize);

  auto disassembleInstruction = [&](uint64_t InstrOffset, MCInst &Instruction,
                                    uint64_t &InstrSize) {
    const uint64_t InstrAddr = SectionAddress + InstrOffset;
    if (!BC->DisAsm->getInstruction(Instruction, InstrSize,
                                    PLTData.slice(InstrOffset), InstrAddr,
                                    nulls())) {
      errs() << "BOLT-ERROR: unable to disassemble instruction in PLT section "
             << Section.getName() << " at offset 0x"
             << Twine::utohexstr(InstrOffset) << '\n';
      exit(1);
    }
  };

  uint64_t InstrOffset = 0;

  while (InstrOffset < SectionSize) {
    InstructionListType Instructions;
    MCInst Instruction;
    const uint64_t EntryOffset = InstrOffset;
    uint64_t EntrySize = EntryOffset == 0 ? 32 : 16;
    uint64_t InstrSize;

    while (InstrOffset < EntryOffset + EntrySize) {
      disassembleInstruction(InstrOffset, Instruction, InstrSize);
      Instructions.emplace_back(Instruction);
      InstrOffset += InstrSize;
    }

    const uint64_t EntryAddress = SectionAddress + EntryOffset;
    const uint64_t TargetAddress = BC->MIB->analyzePLTEntry(
        Instructions.begin(), Instructions.end(), EntryAddress);

    BinaryFunction *BF =
        createPLTBinaryFunction(TargetAddress, EntryAddress, EntrySize);
    assert(BF && "Failed to create PLT function");
    if (opts::Rewrite) {
      assert(BF && BF->empty());
      BF->setSize(EntrySize);
      BF->updateState(BinaryFunction::State::Disassembled);
      BinaryBasicBlock *BB = BF->addBasicBlockAt(0, BF->getSymbol());
      for (auto &Inst : Instructions) {
        BB->addInstruction(Inst);
      }
      bool Ok = BF->handlePLT();
      assert(Ok && "Failed to handle PLT entry");
      (void)Ok;
      BF->updateState(BinaryFunction::State::CFG);
    }
    // Skip nops if any
    while (InstrOffset < SectionSize) {
      disassembleInstruction(InstrOffset, Instruction, InstrSize);
      if (!BC->MIB->isNoop(Instruction))
        break;

      InstrOffset += InstrSize;
      if (opts::Rewrite) {
        BF->getBasicBlockAtOffset(0)->addInstruction(Instruction);
        BF->setSize(BF->getSize() + InstrSize);
      }
    }
  }
}

void RewriteInstance::disassemblePLTSectionX86(BinarySection &Section,
                                               uint64_t EntrySize) {
  const uint64_t SectionAddress = Section.getAddress();
  const uint64_t SectionSize = Section.getSize();
  StringRef PLTContents = Section.getContents();
  ArrayRef<uint8_t> PLTData(
      reinterpret_cast<const uint8_t *>(PLTContents.data()), SectionSize);

  auto disassembleInstruction = [&](uint64_t InstrOffset, MCInst &Instruction,
                                    uint64_t &InstrSize) {
    const uint64_t InstrAddr = SectionAddress + InstrOffset;
    if (!BC->DisAsm->getInstruction(Instruction, InstrSize,
                                    PLTData.slice(InstrOffset), InstrAddr,
                                    nulls())) {
      errs() << "BOLT-ERROR: unable to disassemble instruction in PLT section "
             << Section.getName() << " at offset 0x"
             << Twine::utohexstr(InstrOffset) << '\n';
      exit(1);
    }
  };

  for (uint64_t EntryOffset = 0; EntryOffset + EntrySize <= SectionSize;
       EntryOffset += EntrySize) {
    MCInst Instruction;
    uint64_t InstrSize, InstrOffset = EntryOffset;
    while (InstrOffset < EntryOffset + EntrySize) {
      uint64_t TargetAddress = 0;
      disassembleInstruction(InstrOffset, Instruction, InstrSize);
      // Check if the entry size needs adjustment.
      if (EntryOffset == 0 && BC->MIB->isTerminateBranch(Instruction) &&
          EntrySize == 8)
        EntrySize = 16;

      if (BC->MIB->hasPCRelOperand(Instruction)) {
        bool Ok = BC->MIB->evaluateMemOperandTarget(
            Instruction, TargetAddress, SectionAddress + InstrOffset,
            InstrSize);
        assert(Ok && "Failed to evaluate PLT operand!");
        (void)Ok;
        if (opts::Rewrite) {
          MCSymbol *Sym = BC->getOrCreateGlobalSymbol(TargetAddress, "DATAat");
          // This is a hack to work around the fact that we don't actually
          // disassemble plt. We bluntly assume immediate size of 4 and create
          // addend because the address is relative to the next instruction
          // This should be properly done by disassembling plt, replacing the
          // operand with symbol ref and emitting it back, but it's harder to
          // do on X86 because entries may terminate differently depending on
          // PLT type, so we just patch immediates - at least for now.
          const uint64_t SizeOfImm = 4;
          assert(InstrSize > SizeOfImm);
          const uint64_t OffsetOfImm = InstrOffset + InstrSize - SizeOfImm;
          const uint64_t Addend = -SizeOfImm;
          Section.addRelocation(OffsetOfImm, Sym, ELF::R_X86_64_PC32, Addend);
        }
      }
      if (BC->MIB->isIndirectBranch(Instruction)) {
        assert(TargetAddress);
        const uint64_t EntryAddress = SectionAddress + EntryOffset;
        BinaryFunction *BF =
            createPLTBinaryFunction(TargetAddress, EntryAddress, EntrySize);
        assert(BF && "Failed to create PLT function");
      }
      InstrOffset += InstrSize;
    }
  }
}

void RewriteInstance::disassemblePLT() {
  auto analyzeOnePLTSection = [&](BinarySection &Section, uint64_t EntrySize) {
    if (BC->isAArch64())
      return disassemblePLTSectionAArch64(Section);
    if (BC->isRISCV())
      return disassemblePLTSectionRISCV(Section);
    return disassemblePLTSectionX86(Section, EntrySize);
  };

  for (BinarySection &Section : BC->allocatableSections()) {
    const PLTSectionInfo *PLTSI = getPLTSectionInfo(Section.getName());
    if (!PLTSI)
      continue;

    analyzeOnePLTSection(Section, PLTSI->EntrySize);

    BinaryFunction *PltBF;
    auto BFIter = BC->getBinaryFunctions().find(Section.getAddress());
    assert(BFIter != BC->getBinaryFunctions().end() &&
           "Failed to create PLT header function");
    PltBF = &BFIter->second;
    PltBF->setPseudo(true);
  }
}

void RewriteInstance::adjustFunctionBoundaries() {
  for (auto BFI = BC->getBinaryFunctions().begin(),
            BFE = BC->getBinaryFunctions().end();
       BFI != BFE; ++BFI) {
    BinaryFunction &Function = BFI->second;
    const BinaryFunction *NextFunction = nullptr;
    if (std::next(BFI) != BFE)
      NextFunction = &std::next(BFI)->second;

    // Check if there's a symbol or a function with a larger address in the
    // same section. If there is - it determines the maximum size for the
    // current function. Otherwise, it is the size of a containing section
    // the defines it.
    //
    // NOTE: ignore some symbols that could be tolerated inside the body
    //       of a function.
    auto NextSymRefI = FileSymRefs.upper_bound(Function.getAddress());
    while (NextSymRefI != FileSymRefs.end()) {
      SymbolRef &Symbol = NextSymRefI->second;
      const uint64_t SymbolAddress = NextSymRefI->first;
      const uint64_t SymbolSize = ELFSymbolRef(Symbol).getSize();

      if (NextFunction && SymbolAddress >= NextFunction->getAddress())
        break;

      if (!Function.isSymbolValidInScope(Symbol, SymbolSize))
        break;

      // This is potentially another entry point into the function.
      uint64_t EntryOffset = NextSymRefI->first - Function.getAddress();
      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: adding entry point to function "
                        << Function << " at offset 0x"
                        << Twine::utohexstr(EntryOffset) << '\n');
      Function.addEntryPointAtOffset(EntryOffset);

      ++NextSymRefI;
    }

    // Function runs at most till the end of the containing section.
    uint64_t NextObjectAddress = Function.getOriginSection()->getEndAddress();
    // Or till the next object marked by a symbol.
    if (NextSymRefI != FileSymRefs.end())
      NextObjectAddress = std::min(NextSymRefI->first, NextObjectAddress);

    // Or till the next function not marked by a symbol.
    if (NextFunction)
      NextObjectAddress =
          std::min(NextFunction->getAddress(), NextObjectAddress);

    const uint64_t MaxSize = NextObjectAddress - Function.getAddress();
    if (MaxSize < Function.getSize()) {
      errs() << "BOLT-ERROR: symbol seen in the middle of the function "
             << Function << ". Skipping.\n";
      Function.setSimple(false);
      Function.setMaxSize(Function.getSize());
      continue;
    }
    Function.setMaxSize(MaxSize);
    if (!Function.getSize() && Function.isSimple()) {
      // Some assembly functions have their size set to 0, use the max
      // size as their real size.
      if (opts::Verbosity >= 1)
        outs() << "BOLT-INFO: setting size of function " << Function << " to "
               << Function.getMaxSize() << " (was 0)\n";
      Function.setSize(Function.getMaxSize());
    }
  }
}

void RewriteInstance::relocateEHFrameSection() {
  assert(EHFrameSection && "Non-empty .eh_frame section expected.");

  BinarySection *RelocatedEHFrameSection =
      getSection(".relocated" + getEHFrameSectionName());
  assert(RelocatedEHFrameSection &&
         "Relocated eh_frame section should be preregistered.");
  DWARFDataExtractor DE(EHFrameSection->getContents(),
                        BC->AsmInfo->isLittleEndian(),
                        BC->AsmInfo->getCodePointerSize());
  auto createReloc = [&](uint64_t Value, uint64_t Offset, uint64_t DwarfType) {
    if (DwarfType == dwarf::DW_EH_PE_omit)
      return;

    // Only fix references that are relative to other locations.
    if (!(DwarfType & dwarf::DW_EH_PE_pcrel) &&
        !(DwarfType & dwarf::DW_EH_PE_textrel) &&
        !(DwarfType & dwarf::DW_EH_PE_funcrel) &&
        !(DwarfType & dwarf::DW_EH_PE_datarel))
      return;

    if (!(DwarfType & dwarf::DW_EH_PE_sdata4))
      return;

    uint64_t RelType;
    switch (DwarfType & 0x0f) {
    default:
      llvm_unreachable("unsupported DWARF encoding type");
    case dwarf::DW_EH_PE_sdata4:
    case dwarf::DW_EH_PE_udata4:
      RelType = Relocation::getPC32();
      Offset -= 4;
      break;
    case dwarf::DW_EH_PE_sdata8:
    case dwarf::DW_EH_PE_udata8:
      RelType = Relocation::getPC64();
      Offset -= 8;
      break;
    }

    // Create a relocation against an absolute value since the goal is to
    // preserve the contents of the section independent of the new values
    // of referenced symbols.
    RelocatedEHFrameSection->addRelocation(Offset, nullptr, RelType, Value);
  };

  Error E = EHFrameParser::parse(DE, EHFrameSection->getAddress(), createReloc);
  check_error(std::move(E), "failed to patch EH frame");
}

ArrayRef<uint8_t> RewriteInstance::getLSDAData() {
  return ArrayRef<uint8_t>(LSDASection->getData(),
                           LSDASection->getContents().size());
}

uint64_t RewriteInstance::getLSDAAddress() { return LSDASection->getAddress(); }

Error RewriteInstance::readSpecialSections() {
  NamedRegionTimer T("readSpecialSections", "read special sections",
                     TimerGroupName, TimerGroupDesc, opts::TimeRewrite);

  bool HasTextRelocations = false;
  bool HasSymbolTable = false;
  bool HasDebugInfo = false;

  // Process special sections.
  for (const SectionRef &Section : InputFile->sections()) {
    Expected<StringRef> SectionNameOrErr = Section.getName();
    check_error(SectionNameOrErr.takeError(), "cannot get section name");
    StringRef SectionName = *SectionNameOrErr;

    if (Error E = Section.getContents().takeError())
      return E;
    BC->registerSection(Section);
    LLVM_DEBUG(
        dbgs() << "BOLT-DEBUG: registering section " << SectionName << " @ 0x"
               << Twine::utohexstr(Section.getAddress()) << ":0x"
               << Twine::utohexstr(Section.getAddress() + Section.getSize())
               << "\n");
    if (isDebugSection(SectionName))
      HasDebugInfo = true;
    if (isKSymtabSection(SectionName))
      opts::LinuxKernelMode = true;
  }

  // Set IsRelro section attribute based on PT_GNU_RELRO segment.
  markGnuRelroSections();

  if (HasDebugInfo && !opts::UpdateDebugSections && !opts::AggregateOnly) {
    errs() << "BOLT-WARNING: debug info will be stripped from the binary. "
              "Use -update-debug-sections to keep it.\n";
  }

  HasTextRelocations = (bool)BC->getUniqueSectionByName(".rela.text");
  HasSymbolTable = (bool)BC->getUniqueSectionByName(".symtab");
  LSDASection = BC->getUniqueSectionByName(".gcc_except_table");
  EHFrameSection = BC->getUniqueSectionByName(".eh_frame");
  BuildIDSection = BC->getUniqueSectionByName(".note.gnu.build-id");

  if (ErrorOr<BinarySection &> BATSec =
          BC->getUniqueSectionByName(BoltAddressTranslation::SECTION_NAME)) {
    // Do not read BAT when plotting a heatmap
    if (!opts::HeatmapMode) {
      if (std::error_code EC = BAT->parse(BATSec->getContents())) {
        errs() << "BOLT-ERROR: failed to parse BOLT address translation "
                  "table.\n";
        exit(1);
      }
    }
  }

  if (opts::PrintSections) {
    outs() << "BOLT-INFO: Sections from original binary:\n";
    BC->printSections(outs());
  }

  if (opts::RelocationMode == cl::BOU_TRUE && !HasTextRelocations) {
    errs() << "BOLT-ERROR: relocations against code are missing from the input "
              "file. Cannot proceed in relocations mode (-relocs).\n";
    exit(1);
  }

  BC->HasRelocations =
      HasTextRelocations && (opts::RelocationMode != cl::BOU_FALSE);

  BC->IsStripped = !HasSymbolTable;

  if (BC->IsStripped && !opts::AllowStripped) {
    errs() << "BOLT-ERROR: stripped binaries are not supported. If you know "
              "what you're doing, use --allow-stripped to proceed";
    exit(1);
  }

  // Force non-relocation mode for heatmap generation
  if (opts::HeatmapMode)
    BC->HasRelocations = false;

  if (BC->HasRelocations)
    outs() << "BOLT-INFO: enabling " << (opts::StrictMode ? "strict " : "")
           << "relocation mode\n";

  // Read EH frame for function boundaries info.
  Expected<const DWARFDebugFrame *> EHFrameOrError = BC->DwCtx->getEHFrame();
  if (!EHFrameOrError)
    report_error("expected valid eh_frame section", EHFrameOrError.takeError());
  CFIRdWrt.reset(new CFIReaderWriter(*EHFrameOrError.get()));

  // Parse build-id
  parseBuildID();
  if (std::optional<std::string> FileBuildID = getPrintableBuildID())
    BC->setFileBuildID(*FileBuildID);

  // Read .dynamic/PT_DYNAMIC.
  return readELFDynamic();
}

void RewriteInstance::adjustCommandLineOptions() {
  if (BC->isAArch64() && !BC->HasRelocations)
    errs() << "BOLT-WARNING: non-relocation mode for AArch64 is not fully "
              "supported\n";

  if (RuntimeLibrary *RtLibrary = BC->getRuntimeLibrary())
    RtLibrary->adjustCommandLineOptions(*BC);

  if (opts::AlignMacroOpFusion != MFT_NONE && !BC->isX86()) {
    outs() << "BOLT-INFO: disabling -align-macro-fusion on non-x86 platform\n";
    opts::AlignMacroOpFusion = MFT_NONE;
  }

  if (BC->isX86() && BC->MAB->allowAutoPadding()) {
    if (!BC->HasRelocations) {
      errs() << "BOLT-ERROR: cannot apply mitigations for Intel JCC erratum in "
                "non-relocation mode\n";
      exit(1);
    }
    outs() << "BOLT-WARNING: using mitigation for Intel JCC erratum, layout "
              "may take several minutes\n";
    opts::AlignMacroOpFusion = MFT_NONE;
  }

  if (opts::AlignMacroOpFusion != MFT_NONE && !BC->HasRelocations) {
    outs() << "BOLT-INFO: disabling -align-macro-fusion in non-relocation "
              "mode\n";
    opts::AlignMacroOpFusion = MFT_NONE;
  }

  if (opts::SplitEH && !BC->HasRelocations) {
    errs() << "BOLT-WARNING: disabling -split-eh in non-relocation mode\n";
    opts::SplitEH = false;
  }

  if (opts::StrictMode && !BC->HasRelocations) {
    errs() << "BOLT-WARNING: disabling strict mode (-strict) in non-relocation "
              "mode\n";
    opts::StrictMode = false;
  }

  if (BC->HasRelocations && opts::AggregateOnly &&
      !opts::StrictMode.getNumOccurrences()) {
    outs() << "BOLT-INFO: enabling strict relocation mode for aggregation "
              "purposes\n";
    opts::StrictMode = true;
  }

  if (BC->isX86() && BC->HasRelocations &&
      opts::AlignMacroOpFusion == MFT_HOT && !ProfileReader) {
    outs() << "BOLT-INFO: enabling -align-macro-fusion=all since no profile "
              "was specified\n";
    opts::AlignMacroOpFusion = MFT_ALL;
  }

  if (!BC->HasRelocations &&
      opts::ReorderFunctions != ReorderFunctions::RT_NONE) {
    errs() << "BOLT-ERROR: function reordering only works when "
           << "relocations are enabled\n";
    exit(1);
  }

  if (opts::ReorderFunctions != ReorderFunctions::RT_NONE &&
      !opts::HotText.getNumOccurrences()) {
    opts::HotText = true;
  } else if (opts::HotText && !BC->HasRelocations) {
    errs() << "BOLT-WARNING: hot text is disabled in non-relocation mode\n";
    opts::HotText = false;
  }

  if (opts::HotText && opts::HotTextMoveSections.getNumOccurrences() == 0) {
    opts::HotTextMoveSections.addValue(".stub");
    opts::HotTextMoveSections.addValue(".mover");
    opts::HotTextMoveSections.addValue(".never_hugify");
  }

  if (opts::Hugify) {
    opts::AlignText = BinaryContext::HugePageSize;
    BC->PageAlign = BinaryContext::HugePageSize;
  }

  if (opts::AlignText < opts::AlignFunctions)
    opts::AlignText = (unsigned)opts::AlignFunctions;

  if (opts::UseOldText) {
    errs() << "BOLT-WARNING: '-use-old-text' is deprecated, "
              "using -rewrite instead.\n";
    opts::UseOldText = false;
    opts::Rewrite = true;
  }

  if (opts::Rewrite && !BC->HasRelocations) {
    errs() << "BOLT-ERROR: cannot rewrite in non-relocation mode\n";
    exit(1);
  }

  if (opts::AlignText < opts::AlignFunctions)
    opts::AlignText = (unsigned)opts::AlignFunctions;

  if (BC->isX86() && opts::Lite.getNumOccurrences() == 0 && !opts::StrictMode &&
      !opts::Rewrite)
    opts::Lite = true;

  if (opts::Lite && opts::StrictMode) {
    errs() << "BOLT-ERROR: -strict and -lite cannot be used at the same time\n";
    exit(1);
  }

  if (opts::Lite && opts::Rewrite) {
    errs() << "BOLT-ERROR: cannot combine -lite with -rewrite.\n";
    exit(1);
  }

  if (opts::Lite)
    outs() << "BOLT-INFO: enabling lite mode\n";

  if (!opts::SaveProfile.empty() && BAT->enabledFor(InputFile)) {
    errs() << "BOLT-ERROR: unable to save profile in YAML format for input "
              "file processed by BOLT. Please remove -w option and use branch "
              "profile.\n";
    exit(1);
  }
}

namespace {
template <typename ELFT>
int64_t getRelocationAddend(const ELFObjectFile<ELFT> *Obj,
                            const RelocationRef &RelRef) {
  using ELFShdrTy = typename ELFT::Shdr;
  using Elf_Rela = typename ELFT::Rela;
  int64_t Addend = 0;
  const ELFFile<ELFT> &EF = Obj->getELFFile();
  DataRefImpl Rel = RelRef.getRawDataRefImpl();
  const ELFShdrTy *RelocationSection = cantFail(EF.getSection(Rel.d.a));
  switch (RelocationSection->sh_type) {
  default:
    llvm_unreachable("unexpected relocation section type");
  case ELF::SHT_REL:
    break;
  case ELF::SHT_RELA: {
    const Elf_Rela *RelA = Obj->getRela(Rel);
    Addend = RelA->r_addend;
    break;
  }
  }

  return Addend;
}

int64_t getRelocationAddend(const ELFObjectFileBase *Obj,
                            const RelocationRef &Rel) {
  return getRelocationAddend(cast<ELF64LEObjectFile>(Obj), Rel);
}

template <typename ELFT>
uint32_t getRelocationSymbol(const ELFObjectFile<ELFT> *Obj,
                             const RelocationRef &RelRef) {
  using ELFShdrTy = typename ELFT::Shdr;
  uint32_t Symbol = 0;
  const ELFFile<ELFT> &EF = Obj->getELFFile();
  DataRefImpl Rel = RelRef.getRawDataRefImpl();
  const ELFShdrTy *RelocationSection = cantFail(EF.getSection(Rel.d.a));
  switch (RelocationSection->sh_type) {
  default:
    llvm_unreachable("unexpected relocation section type");
  case ELF::SHT_REL:
    Symbol = Obj->getRel(Rel)->getSymbol(EF.isMips64EL());
    break;
  case ELF::SHT_RELA:
    Symbol = Obj->getRela(Rel)->getSymbol(EF.isMips64EL());
    break;
  }

  return Symbol;
}

uint32_t getRelocationSymbol(const ELFObjectFileBase *Obj,
                             const RelocationRef &Rel) {
  return getRelocationSymbol(cast<ELF64LEObjectFile>(Obj), Rel);
}
} // anonymous namespace

bool RewriteInstance::analyzeRelocation(
    const RelocationRef &Rel, uint64_t &RType, std::string &SymbolName,
    bool &IsSectionRelocation, uint64_t &SymbolAddress, int64_t &Addend,
    uint64_t &ExtractedValue, bool &Skip) const {
  Skip = false;
  if (!Relocation::isSupported(RType))
    return false;

  const bool IsAArch64 = BC->isAArch64();

  const size_t RelSize = Relocation::getSizeForType(RType);

  ErrorOr<uint64_t> Value =
      BC->getUnsignedValueAtAddress(Rel.getOffset(), RelSize);
  assert(Value && "failed to extract relocated value");
  Skip = Relocation::skipRelocationProcess(RType, *Value);
  if (Skip)
    return true;
  ExtractedValue = Relocation::extractValue(RType, *Value, Rel.getOffset());
  Addend = getRelocationAddend(InputFile, Rel);

  const bool IsPCRelative = Relocation::isPCRelative(RType);
  const uint64_t PCRelOffset = IsPCRelative && !IsAArch64 ? Rel.getOffset() : 0;
  bool SkipVerification = false;

  auto SymbolIter = Rel.getSymbol();
  if (SymbolIter == InputFile->symbol_end()) {
    SymbolAddress = ExtractedValue - Addend + PCRelOffset;
    MCSymbol *RelSymbol =
        BC->getOrCreateGlobalSymbol(SymbolAddress, "RELSYMat");
    SymbolName = std::string(RelSymbol->getName());
    IsSectionRelocation = false;
  } else {
    const SymbolRef &Symbol = *SymbolIter;
    SymbolName = std::string(cantFail(Symbol.getName()));

    SymbolAddress = cantFail(Symbol.getAddress());
    SkipVerification = (cantFail(Symbol.getType()) == SymbolRef::ST_Other);
    // Section symbols are marked as ST_Debug.
    IsSectionRelocation = (cantFail(Symbol.getType()) == SymbolRef::ST_Debug);
    // Check for PLT entry registered with symbol name
    if (opts::Rewrite && Relocation::isGOT(RType) && BC->isRISC()) {
      std::string StrippedName = SymbolName.substr(0, SymbolName.find("@"));
      if (auto It = GOTSymbolsByName.find(StrippedName);
          It != GOTSymbolsByName.end()) {
        SymbolAddress = It->second;
      } else {
        // we can't determine which .got entry is referenced by that
        // instruction, so we have to relax .got access to direct access. For
        // adrp it is enough to just change relocation type to load page address
        // of the symbol itself, while LDR will be handled during disassembly -
        // we'll change LDR to ADD when we see ldr with
        // R_AARCH64_ADD_ABS_LO12_NC relocation.
        switch (RType) {
        case ELF::R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC:
        case ELF::R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21:
          // for TLS relocs, we'll match instructions during disassembly to
          // determine .got entry.
          break;
        case ELF::R_AARCH64_ADR_GOT_PAGE:
          outs() << formatv("BOLT-WARNING: cannot find .got entry for symbol "
                            "{0} referenced by R_AARCH64_ADR_GOT_PAGE at {1:x}. Will relax to adrp+add\n",
                            StrippedName, Rel.getOffset());
          RType = ELF::R_AARCH64_ADR_PREL_PG_HI21;
          break;
        case ELF::R_AARCH64_LD64_GOT_LO12_NC:
          outs() << formatv("BOLT-WARNING: cannot find .got entry for symbol "
                            "{0} referenced by R_AARCH64_LD64_GOT_LO12_NC at {1:x}. Will relax to adrp+add\n",
                            StrippedName, Rel.getOffset());
          RType = ELF::R_AARCH64_ADD_ABS_LO12_NC;
          break;
        default:
          errs() << formatv("BOLT-ERROR: Unexpected .got access relocation at {0:x}: can't find .got entry for {1}!\n",
                            Rel.getOffset(), StrippedName);
          exit(1);
        }
      }
    }
    if (!SymbolAddress && IsAArch64) {
      const BinaryData *BD = BC->getPLTBinaryDataByName(SymbolName);
      SymbolAddress = BD ? BD->getAddress() : 0;
    }
  }
  // For PIE or dynamic libs, the linker may choose not to put the relocation
  // result at the address if it is a X86_64_64 one because it will emit a
  // dynamic relocation (X86_RELATIVE) for the dynamic linker and loader to
  // resolve it at run time. The static relocation result goes as the addend
  // of the dynamic relocation in this case. We can't verify these cases.
  // FIXME: perhaps we can try to find if it really emitted a corresponding
  // RELATIVE relocation at this offset with the correct value as the addend.
  if (!BC->HasFixedLoadAddress && RelSize == 8)
    SkipVerification = true;

  if (IsSectionRelocation && !IsAArch64) {
    ErrorOr<BinarySection &> Section = BC->getSectionForAddress(SymbolAddress);
    assert(Section && "section expected for section relocation");
    SymbolName = "section " + std::string(Section->getName());
    // Convert section symbol relocations to regular relocations inside
    // non-section symbols.
    if (Section->containsAddress(ExtractedValue) && !IsPCRelative) {
      SymbolAddress = ExtractedValue;
      Addend = 0;
    } else {
      Addend = ExtractedValue - (SymbolAddress - PCRelOffset);
    }
  }

  // If no symbol has been found or if it is a relocation requiring the
  // creation of a GOT entry, do not link against the symbol but against
  // whatever address was extracted from the instruction itself. We are
  // not creating a GOT entry as this was already processed by the linker.
  // For GOT relocs, do not subtract addend as the addend does not refer
  // to this instruction's target, but it refers to the target in the GOT
  // entry.
  if (Relocation::isGOT(RType) && (BC->isX86() || !opts::Rewrite)) {
    Addend = 0;
    SymbolAddress = ExtractedValue + PCRelOffset;
  } else if (Relocation::isTLS(RType)) {
    SkipVerification = true;
  } else if (!SymbolAddress) {
    assert(!IsSectionRelocation);
    if (ExtractedValue || Addend == 0 || IsPCRelative) {
      SymbolAddress =
          truncateToSize(ExtractedValue - Addend + PCRelOffset, RelSize);
    } else {
      // This is weird case.  The extracted value is zero but the addend is
      // non-zero and the relocation is not pc-rel.  Using the previous logic,
      // the SymbolAddress would end up as a huge number.  Seen in
      // exceptions_pic.test.
      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: relocation @ 0x"
                        << Twine::utohexstr(Rel.getOffset())
                        << " value does not match addend for "
                        << "relocation to undefined symbol.\n");
      return true;
    }
  }

  auto verifyExtractedValue = [&]() {
    if (SkipVerification)
      return true;

    if (BC->isRISC())
      return true;

    if (SymbolName == "__hot_start" || SymbolName == "__hot_end")
      return true;

    if (RType == ELF::R_X86_64_PLT32)
      return true;

    return truncateToSize(ExtractedValue, RelSize) ==
           truncateToSize(SymbolAddress + Addend - PCRelOffset, RelSize);
  };

  (void)verifyExtractedValue;
  assert(verifyExtractedValue() && "mismatched extracted relocation value");

  return true;
}

void RewriteInstance::processDynamicRelocations() {
  // Read .relr.dyn section containing compressed R_*_RELATIVE relocations.
  if (DynamicRelrSize > 0) {
    ErrorOr<BinarySection &> DynamicRelrSectionOrErr =
        BC->getSectionForAddress(*DynamicRelrAddress);
    if (!DynamicRelrSectionOrErr)
      report_error("unable to find section corresponding to DT_RELR",
                   DynamicRelrSectionOrErr.getError());
    if (DynamicRelrSectionOrErr->getSize() != DynamicRelrSize)
      report_error("section size mismatch for DT_RELRSZ",
                   errc::executable_format_error);
    readDynamicRelrRelocations(*DynamicRelrSectionOrErr);
  }

  // Read relocations for PLT - DT_JMPREL.
  if (PLTRelocationsSize > 0) {
    ErrorOr<BinarySection &> PLTRelSectionOrErr =
        BC->getSectionForAddress(*PLTRelocationsAddress);
    if (!PLTRelSectionOrErr)
      report_error("unable to find section corresponding to DT_JMPREL",
                   PLTRelSectionOrErr.getError());
    if (PLTRelSectionOrErr->getSize() != PLTRelocationsSize)
      report_error("section size mismatch for DT_PLTRELSZ",
                   errc::executable_format_error);
    readDynamicRelocations(PLTRelSectionOrErr->getSectionRef(),
                           /*IsJmpRel*/ true);
  }

  // The rest of dynamic relocations - DT_RELA.
  if (DynamicRelocationsSize > 0) {
    ErrorOr<BinarySection &> DynamicRelSectionOrErr =
        BC->getSectionForAddress(*DynamicRelocationsAddress);
    if (!DynamicRelSectionOrErr)
      report_error("unable to find section corresponding to DT_RELA",
                   DynamicRelSectionOrErr.getError());
    auto DynamicRelSectionSize = DynamicRelSectionOrErr->getSize();
    // On RISC-V DT_RELASZ seems to include both .rela.dyn and .rela.plt
    if (DynamicRelocationsSize == DynamicRelSectionSize + PLTRelocationsSize)
      DynamicRelocationsSize = DynamicRelSectionSize;
    if (DynamicRelSectionSize != DynamicRelocationsSize)
      report_error("section size mismatch for DT_RELASZ",
                   errc::executable_format_error);
    readDynamicRelocations(DynamicRelSectionOrErr->getSectionRef(),
                           /*IsJmpRel*/ false);
  }
}

void RewriteInstance::processRelocations() {
  if (!BC->HasRelocations)
    return;

  for (const SectionRef &Section : InputFile->sections()) {
    if (cantFail(Section.getRelocatedSection()) != InputFile->section_end() &&
        !BinarySection(*BC, Section).isAllocatable())
      readRelocations(Section);
  }

  if (NumFailedRelocations)
    errs() << "BOLT-WARNING: Failed to analyze " << NumFailedRelocations
           << " relocations\n";
}

void RewriteInstance::readDynamicRelocations(const SectionRef &Section,
                                             bool IsJmpRel) {
  assert(BinarySection(*BC, Section).isAllocatable() && "allocatable expected");

  LLVM_DEBUG({
    StringRef SectionName = cantFail(Section.getName());
    dbgs() << "BOLT-DEBUG: reading relocations for section " << SectionName
           << ":\n";
  });

  for (const RelocationRef &Rel : Section.relocations()) {
    const uint64_t RType = Rel.getType();
    if (Relocation::isNone(RType))
      continue;

    StringRef SymbolName = "<none>";
    MCSymbol *Symbol = nullptr;
    uint64_t SymbolAddress = 0;
    const uint64_t Addend = getRelocationAddend(InputFile, Rel);

    symbol_iterator SymbolIter = Rel.getSymbol();
    if (SymbolIter != InputFile->symbol_end()) {
      SymbolName = cantFail(SymbolIter->getName());
      BinaryData *BD = BC->getBinaryDataByName(SymbolName);
      Symbol = BD ? BD->getSymbol()
                  : BC->getOrCreateUndefinedGlobalSymbol(SymbolName);
      SymbolAddress = cantFail(SymbolIter->getAddress());
      (void)SymbolAddress;
    }

    LLVM_DEBUG(
      SmallString<16> TypeName;
      Rel.getTypeName(TypeName);
      dbgs() << "BOLT-DEBUG: dynamic relocation at 0x"
             << Twine::utohexstr(Rel.getOffset()) << " : " << TypeName
             << " : " << SymbolName << " : " <<  Twine::utohexstr(SymbolAddress)
             << " : + 0x" << Twine::utohexstr(Addend) << '\n'
    );

    if (IsJmpRel)
      IsJmpRelocation[RType] = true;

    if (Symbol)
      SymbolIndex[Symbol] = getRelocationSymbol(InputFile, Rel);

    auto IsGotLikeReloc = [this](const uint64_t RType) {
      return ((BC->isAArch64() && (RType == ELF::R_AARCH64_GLOB_DAT ||
                                   RType == ELF::R_AARCH64_RELATIVE ||
                                   RType == ELF::R_AARCH64_TLS_TPREL64)) ||
              (BC->isRISCV() && RType == ELF::R_RISCV_64));
    };

    if (opts::Rewrite && IsGotLikeReloc(RType)) {
      if (Symbol && !SymbolName.empty()) {
        GOTSymbolsByName[SymbolName.str()] = Rel.getOffset();
        LLVM_DEBUG(dbgs() << formatv("BOLT-INFO: GOT entry at {0:x} contains "
                                     "symbol {1} with address {2:x}\n",
                                     Rel.getOffset(), SymbolName, Addend););
      } else if (BinaryData *BD = BC->getBinaryDataAtAddress(Addend)) {
        LLVM_DEBUG(outs() << formatv("BOLT-INFO: GOT entry at {0:x} contains "
                                     "symbol {1} with address {2:x}\n",
                                     Rel.getOffset(), BD->getName(), Addend););

        for (MCSymbol *Sym : BD->getSymbols()) {
          StringRef Name = Sym->getName().substr(0, Sym->getName().find('/'));
          if (Name.empty())
            continue;
          GOTSymbolsByName[Name.str()] = Rel.getOffset();
        }
      }
    }
    BC->addDynamicRelocation(Rel.getOffset(), Symbol, RType, Addend);
  }
}

void RewriteInstance::readDynamicRelrRelocations(BinarySection &Section) {
  assert(Section.isAllocatable() && "allocatable expected");

  LLVM_DEBUG({
    StringRef SectionName = Section.getName();
    dbgs() << "BOLT-DEBUG: reading relocations in section " << SectionName
           << ":\n";
  });

  const uint64_t RType = Relocation::getRelative();
  const uint8_t PSize = BC->AsmInfo->getCodePointerSize();
  const uint64_t MaxDelta = ((CHAR_BIT * DynamicRelrEntrySize) - 1) * PSize;

  auto ExtractAddendValue = [&](uint64_t Address) -> uint64_t {
    ErrorOr<BinarySection &> Section = BC->getSectionForAddress(Address);
    assert(Section && "cannot get section for data address from RELR");
    DataExtractor DE = DataExtractor(Section->getContents(),
                                     BC->AsmInfo->isLittleEndian(), PSize);
    uint64_t Offset = Address - Section->getAddress();
    return DE.getUnsigned(&Offset, PSize);
  };

  auto AddRelocation = [&](uint64_t Address) {
    uint64_t Addend = ExtractAddendValue(Address);
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: R_*_RELATIVE relocation at 0x"
                      << Twine::utohexstr(Address) << " to 0x"
                      << Twine::utohexstr(Addend) << '\n';);
    BC->addDynamicRelocation(Address, nullptr, RType, Addend);
  };

  DataExtractor DE = DataExtractor(Section.getContents(),
                                   BC->AsmInfo->isLittleEndian(), PSize);
  uint64_t Offset = 0, Address = 0;
  uint64_t RelrCount = DynamicRelrSize / DynamicRelrEntrySize;
  while (RelrCount--) {
    assert(DE.isValidOffset(Offset));
    uint64_t Entry = DE.getUnsigned(&Offset, DynamicRelrEntrySize);
    if ((Entry & 1) == 0) {
      AddRelocation(Entry);
      Address = Entry + PSize;
    } else {
      const uint64_t StartAddress = Address;
      while (Entry >>= 1) {
        if (Entry & 1)
          AddRelocation(Address);

        Address += PSize;
      }

      Address = StartAddress + MaxDelta;
    }
  }
}

void RewriteInstance::printRelocationInfo(const RelocationRef &Rel,
                                          StringRef SymbolName,
                                          uint64_t SymbolAddress,
                                          uint64_t Addend,
                                          uint64_t ExtractedValue) const {
  SmallString<16> TypeName;
  Rel.getTypeName(TypeName);
  const uint64_t Address = SymbolAddress + Addend;
  const uint64_t Offset = Rel.getOffset();
  ErrorOr<BinarySection &> Section = BC->getSectionForAddress(SymbolAddress);
  BinaryFunction *Func =
      BC->getBinaryFunctionContainingAddress(Offset, false, BC->isAArch64());
  dbgs() << formatv("Relocation: offset = {0:x}; type = {1}; value = {2:x}; ",
                    Offset, TypeName, ExtractedValue)
         << formatv("symbol = {0} ({1}); symbol address = {2:x}; ", SymbolName,
                    Section ? Section->getName() : "", SymbolAddress)
         << formatv("addend = {0:x}; address = {1:x}; in = ", Addend, Address);
  if (Func)
    dbgs() << Func->getPrintName();
  else
    dbgs() << BC->getSectionForAddress(Rel.getOffset())->getName();
  dbgs() << '\n';
}

void RewriteInstance::readRelocations(const SectionRef &Section) {
  LLVM_DEBUG({
    StringRef SectionName = cantFail(Section.getName());
    dbgs() << "BOLT-DEBUG: reading relocations for section " << SectionName
           << ":\n";
  });
  if (BinarySection(*BC, Section).isAllocatable()) {
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: ignoring runtime relocations\n");
    return;
  }
  section_iterator SecIter = cantFail(Section.getRelocatedSection());
  assert(SecIter != InputFile->section_end() && "relocated section expected");
  SectionRef RelocatedSection = *SecIter;

  StringRef RelocatedSectionName = cantFail(RelocatedSection.getName());
  LLVM_DEBUG(dbgs() << "BOLT-DEBUG: relocated section is "
                    << RelocatedSectionName << '\n');
  auto RelocatedBinarySectionOrErr =
      BC->getUniqueSectionByName(RelocatedSectionName);
  assert(RelocatedBinarySectionOrErr);
  BinarySection &RelocatedBinarySection = *RelocatedBinarySectionOrErr;
  if (!RelocatedBinarySection.isAllocatable()) {
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: ignoring relocations against "
                      << "non-allocatable section\n");
    return;
  }
  const bool SkipRelocs = StringSwitch<bool>(RelocatedSectionName)
                              .Cases(".plt", ".rela.plt", ".got.plt",
                                     ".eh_frame", ".gcc_except_table", true)
                              .Default(false);
  if (SkipRelocs) {
    LLVM_DEBUG(
        dbgs() << "BOLT-DEBUG: ignoring relocations against known section\n");
    return;
  }

  for (const RelocationRef &Rel : Section.relocations())
    handleRelocation(RelocatedBinarySection, Rel);
}

void RewriteInstance::handleRelocation(BinarySection &RelocatedSection,
                                       const RelocationRef &Rel) {
  const bool IsAArch64 = BC->isAArch64();
  const bool IsFromCode = RelocatedSection.isText();

  SmallString<16> TypeName;
  Rel.getTypeName(TypeName);
  uint64_t RType = Rel.getType();
  if (Relocation::skipRelocationType(RType))
    return;

  // Adjust the relocation type as the linker might have skewed it.
  if (BC->isX86() && (RType & ELF::R_X86_64_converted_reloc_bit)) {
    if (opts::Verbosity >= 1)
      dbgs() << "BOLT-WARNING: ignoring R_X86_64_converted_reloc_bit\n";
    RType &= ~ELF::R_X86_64_converted_reloc_bit;
  }
  if (Relocation::isTLS(RType)) {
    // No special handling required for TLS relocations on X86.
    if (BC->isX86())
      return;

    // The non-got related TLS relocations on AArch64 also could be skipped.
    if (!Relocation::isGOT(RType))
      return;
  }


  std::string SymbolName;
  uint64_t SymbolAddress;
  int64_t Addend;
  uint64_t ExtractedValue;
  bool IsSectionRelocation;
  bool Skip;
  if (!analyzeRelocation(Rel, RType, SymbolName, IsSectionRelocation,
                         SymbolAddress, Addend, ExtractedValue, Skip)) {
    LLVM_DEBUG({
      dbgs() << "BOLT-WARNING: failed to analyze relocation @ offset = "
             << formatv("{0:x}; type name = {1}\n", Rel.getOffset(), TypeName);
    });
    ++NumFailedRelocations;
    return;
  }

  if (Skip) {
    LLVM_DEBUG({
      dbgs() << "BOLT-DEBUG: skipping relocation @ offset = "
             << formatv("{0:x}; type name = {1}\n", Rel.getOffset(), TypeName);
    });
    return;
  }

  const uint64_t Address = SymbolAddress + Addend;

  LLVM_DEBUG({
    dbgs() << "BOLT-DEBUG: ";
    printRelocationInfo(Rel, SymbolName, SymbolAddress, Addend, ExtractedValue);
  });

  BinaryFunction *ContainingBF = nullptr;
  if (IsFromCode) {
    ContainingBF =
        BC->getBinaryFunctionContainingAddress(Rel.getOffset(),
                                               /*CheckPastEnd*/ false,
                                               /*UseMaxSize*/ true);
    assert(ContainingBF && "cannot find function for address in code");
    if (!IsAArch64 && !ContainingBF->containsAddress(Rel.getOffset())) {
      if (opts::Verbosity >= 1)
        outs() << formatv("BOLT-INFO: {0} has relocations in padding area\n",
                          *ContainingBF);
      ContainingBF->setSize(ContainingBF->getMaxSize());
      ContainingBF->setSimple(false);
      return;
    }
  }

  MCSymbol *ReferencedSymbol = nullptr;
  bool IsToSectionEnd = false;
  if (auto It = BC->EndSymbols.find(SymbolName); It != BC->EndSymbols.end()) {
    // We need to create local symbol because using registerName will collide
    // with the next section if it's on the same address. This requires us to
    // check for EndSymbols in a lot of places, but what else can we do?
    ReferencedSymbol = BC->Ctx->getOrCreateSymbol(SymbolName);
    IsToSectionEnd = true;
  } else if (!IsSectionRelocation)
    if (BinaryData *BD = BC->getBinaryDataByName(SymbolName))
      ReferencedSymbol = BD->getSymbol();

  ErrorOr<BinarySection &> ReferencedSection{std::errc::bad_address};
  symbol_iterator SymbolIter = Rel.getSymbol();
  if (SymbolIter != InputFile->symbol_end()) {
    SymbolRef Symbol = *SymbolIter;
    section_iterator Section =
        cantFail(Symbol.getSection(), "cannot get symbol section");
    if (Section != InputFile->section_end()) {
      Expected<StringRef> SectionName = Section->getName();
      if (SectionName && !SectionName->empty())
        ReferencedSection = BC->getUniqueSectionByName(*SectionName);
    } else if (ReferencedSymbol &&
               (cantFail(Symbol.getFlags()) & SymbolRef::SF_Absolute)) {
      // This might be a relocation for an ABS symbols like __global_pointer$ on
      // RISC-V
      ContainingBF->addRelocation(Rel.getOffset(), ReferencedSymbol,
                                  Rel.getType(), 0,
                                  cantFail(Symbol.getValue()));
      return;
    }
  }

  if (!ReferencedSection)
    ReferencedSection = BC->getSectionForAddress(SymbolAddress);

  const bool IsToCode = ReferencedSection && ReferencedSection->isText();

  // Special handling of PC-relative relocations.
  if (!IsAArch64 && !BC->isRISCV() && Relocation::isPCRelative(RType)) {
    if (!IsFromCode && IsToCode) {
      // PC-relative relocations from data to code are tricky since the
      // original information is typically lost after linking, even with
      // '--emit-relocs'. Such relocations are normally used by PIC-style
      // jump tables and they reference both the jump table and jump
      // targets by computing the difference between the two. If we blindly
      // apply the relocation, it will appear that it references an arbitrary
      // location in the code, possibly in a different function from the one
      // containing the jump table.
      //
      // For that reason, we only register the fact that there is a
      // PC-relative relocation at a given address against the code.
      // The actual referenced label/address will be determined during jump
      // table analysis.
      BC->addPCRelativeDataRelocation(Rel.getOffset());
    } else if (ContainingBF && !IsSectionRelocation && ReferencedSymbol) {
      // If we know the referenced symbol, register the relocation from
      // the code. It's required  to properly handle cases where
      // "symbol + addend" references an object different from "symbol".
      ContainingBF->addRelocation(Rel.getOffset(), ReferencedSymbol, RType,
                                  Addend, ExtractedValue);
    } else if (opts::Rewrite && !IsToCode && !IsFromCode) {
      if (!ReferencedSymbol) {
        ReferencedSymbol =
            BC->getOrCreateGlobalSymbol(SymbolAddress, "SYMBOLat");
      }

      RelocatedSection.addRelocation(
          Rel.getOffset() - RelocatedSection.getAddress(), ReferencedSymbol,
          RType, Addend, ExtractedValue);
      LLVM_DEBUG({
        dbgs() << formatv("BOLT-DEBUG: creating PC-relative data relocation at "
                          "{0:x} for {1}\n",
                          Rel.getOffset(), SymbolName);
      });
    } else {
      LLVM_DEBUG({
        dbgs() << formatv("BOLT-DEBUG: ignoring PC-relative relocation at "
                          "{0:x} for {1}\n",
                          Rel.getOffset(), SymbolName);
      });
    }

    return;
  }

  bool ForceRelocation = BC->forceSymbolRelocations(SymbolName);
  if ((BC->isAArch64() || BC->isRISCV()) && Relocation::isGOT(RType))
    ForceRelocation = true;

  if (!ReferencedSection && !ForceRelocation) {
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: cannot determine referenced section.\n");
    return;
  }

  // Occasionally we may see a reference past the last byte of the function
  // typically as a result of __builtin_unreachable(). Check it here.
  BinaryFunction *ReferencedBF = BC->getBinaryFunctionContainingAddress(
      Address, /*CheckPastEnd*/ true, /*UseMaxSize*/ IsAArch64);

  if (!IsSectionRelocation) {
    if (BinaryFunction *BF =
            BC->getBinaryFunctionContainingAddress(SymbolAddress)) {
      if (BF != ReferencedBF) {
        // It's possible we are referencing a function without referencing any
        // code, e.g. when taking a bitmask action on a function address.
        errs() << "BOLT-WARNING: non-standard function reference (e.g. bitmask)"
               << formatv(" detected against function {0} from ", *BF);
        if (IsFromCode)
          errs() << formatv("function {0}\n", *ContainingBF);
        else
          errs() << formatv("data section at {0:x}\n", Rel.getOffset());
        LLVM_DEBUG(printRelocationInfo(Rel, SymbolName, SymbolAddress, Addend,
                                       ExtractedValue));
        ReferencedBF = BF;
      }
    }
  } else if (ReferencedBF) {
    assert(ReferencedSection && "section expected for section relocation");
    if (*ReferencedBF->getOriginSection() != *ReferencedSection) {
      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: ignoring false function reference\n");
      ReferencedBF = nullptr;
    }
  }

  // Workaround for a member function pointer de-virtualization bug. We check
  // if a non-pc-relative relocation in the code is pointing to (fptr - 1).
  if (IsToCode && ContainingBF && !Relocation::isPCRelative(RType) &&
      (!ReferencedBF || (ReferencedBF->getAddress() != Address))) {
    if (const BinaryFunction *RogueBF =
            BC->getBinaryFunctionAtAddress(Address + 1)) {
      // Do an extra check that the function was referenced previously.
      // It's a linear search, but it should rarely happen.
      auto CheckReloc = [&](const Relocation &Rel) {
        return Rel.Symbol == RogueBF->getSymbol() &&
               !Relocation::isPCRelative(Rel.Type);
      };
      bool Found = llvm::any_of(
          llvm::make_second_range(ContainingBF->Relocations), CheckReloc);

      if (Found) {
        errs() << "BOLT-WARNING: detected possible compiler de-virtualization "
                  "bug: -1 addend used with non-pc-relative relocation against "
               << formatv("function {0} in function {1}\n", *RogueBF,
                          *ContainingBF);
        return;
      }
    }
  }

  if (IsToSectionEnd) {
    // do nothing, add it as it is later
  } else if (ForceRelocation) {
    std::string Name = SymbolName;
    if (Relocation::isGOT(RType)) {
      if (opts::Rewrite) {
        Name = SymbolName + "@GOT";
      } else {
        Addend = Address;
        SymbolAddress = 0;
        Name = "__BOLT_got_zero";
      }
    }
    ReferencedSymbol = BC->registerNameAtAddress(Name, SymbolAddress, 0, 0);
    LLVM_DEBUG(
        dbgs() << formatv("BOLT-DEBUG: forcing relocation against symbol {0} "
                          "with addend {1:x} and address {2:x}\n",
                          SymbolName, Addend, SymbolAddress));
  } else if (ReferencedBF) {
    ReferencedSymbol = ReferencedBF->getSymbol();
    uint64_t RefFunctionOffset = 0;

    // Adjust the point of reference to a code location inside a function.
    if (ReferencedBF->containsAddress(Address, /*UseMaxSize = */ true)) {
      RefFunctionOffset = Address - ReferencedBF->getAddress();
      if (RefFunctionOffset) {
        if (ContainingBF && ContainingBF != ReferencedBF) {
          ReferencedSymbol =
              ReferencedBF->addEntryPointAtOffset(RefFunctionOffset);
        } else {
          ReferencedSymbol =
              ReferencedBF->getOrCreateLocalLabel(Address,
                                                  /*CreatePastEnd =*/true);

          // If ContainingBF != nullptr, it equals ReferencedBF (see
          // if-condition above) so we're handling a relocation from a function
          // to itself. RISC-V uses such relocations for branches, for example.
          // These should not be registered as externally references offsets.
          if (!ContainingBF)
            ReferencedBF->registerReferencedOffset(RefFunctionOffset);
        }
        if (opts::Verbosity > 1 && RelocatedSection.isWritable())
          errs() << "BOLT-WARNING: writable reference into the middle of the "
                 << formatv("function {0} detected at address {1:x}\n",
                            *ReferencedBF, Rel.getOffset());
      }
      SymbolAddress = Address;
      Addend = 0;
    }
    LLVM_DEBUG({
      dbgs() << "  referenced function " << *ReferencedBF;
      if (Address != ReferencedBF->getAddress())
        dbgs() << formatv(" at offset {0:x}", RefFunctionOffset);
      dbgs() << '\n';
    });
  } else {
    if (IsToCode && SymbolAddress) {
      // This can happen e.g. with PIC-style jump tables.
      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: no corresponding function for "
                           "relocation against code\n");
    }

    // In AArch64 there are zero reasons to keep a reference to the
    // "original" symbol plus addend. The original symbol is probably just a
    // section symbol. If we are here, this means we are probably accessing
    // data, so it is imperative to keep the original address.
    if (IsAArch64) {
      SymbolName = formatv("SYMBOLat{0:x}", Address);
      SymbolAddress = Address;
      Addend = 0;
    }

    if (BinaryData *BD = BC->getBinaryDataContainingAddress(SymbolAddress)) {
      // Note: this assertion is trying to check sanity of BinaryData objects
      // but AArch64 has inferred and incomplete object locations coming from
      // GOT/TLS or any other non-trivial relocation (that requires creation
      // of sections and whose symbol address is not really what should be
      // encoded in the instruction). So we essentially disabled this check
      // for AArch64 and live with bogus names for objects.
      assert((IsAArch64 || IsSectionRelocation ||
              BD->nameStartsWith(SymbolName) ||
              BD->nameStartsWith("PG" + SymbolName) ||
              (BD->nameStartsWith("ANONYMOUS") &&
               (BD->getSectionName().startswith(".plt") ||
                BD->getSectionName().endswith(".plt")))) &&
             "BOLT symbol names of all non-section relocations must match up "
             "with symbol names referenced in the relocation");

      if (IsSectionRelocation)
        BC->markAmbiguousRelocations(*BD, Address);

      ReferencedSymbol = BD->getSymbol();
      Addend += (SymbolAddress - BD->getAddress());
      SymbolAddress = BD->getAddress();
      assert(Address == SymbolAddress + Addend);
    } else {
      // These are mostly local data symbols but undefined symbols
      // in relocation sections can get through here too, from .plt.
      assert(
          (IsAArch64 || BC->isRISCV() || IsSectionRelocation ||
           BC->getSectionNameForAddress(SymbolAddress)->startswith(".plt")) &&
          "known symbols should not resolve to anonymous locals");

      if (IsSectionRelocation) {
        ReferencedSymbol =
            BC->getOrCreateGlobalSymbol(SymbolAddress, "SYMBOLat");
      } else {
        SymbolRef Symbol = *Rel.getSymbol();
        const uint64_t SymbolSize =
            IsAArch64 ? 0 : ELFSymbolRef(Symbol).getSize();
        const uint64_t SymbolAlignment = IsAArch64 ? 1 : Symbol.getAlignment();
        const uint32_t SymbolFlags = cantFail(Symbol.getFlags());
        std::string Name;
        if (SymbolFlags & SymbolRef::SF_Global) {
          Name = SymbolName;
        } else {
          if (StringRef(SymbolName)
                  .startswith(BC->AsmInfo->getPrivateGlobalPrefix()))
            Name = NR.uniquify("PG" + SymbolName);
          else
            Name = NR.uniquify(SymbolName);
        }
        ReferencedSymbol = BC->registerNameAtAddress(
            Name, SymbolAddress, SymbolSize, SymbolAlignment, SymbolFlags);
      }

      if (IsSectionRelocation) {
        BinaryData *BD = BC->getBinaryDataByName(ReferencedSymbol->getName());
        BC->markAmbiguousRelocations(*BD, Address);
      }
    }
  }

  auto checkMaxDataRelocations = [&]() {
    ++NumDataRelocations;
    LLVM_DEBUG(if (opts::MaxDataRelocations &&
                   NumDataRelocations + 1 == opts::MaxDataRelocations) {
      dbgs() << "BOLT-DEBUG: processing ending on data relocation "
             << NumDataRelocations << ": ";
      printRelocationInfo(Rel, ReferencedSymbol->getName(), SymbolAddress,
                          Addend, ExtractedValue);
    });

    return (!opts::MaxDataRelocations ||
            NumDataRelocations < opts::MaxDataRelocations);
  };

  if (opts::Rewrite ||
      (ReferencedSection && refersToReorderedSection(ReferencedSection)) ||
      (opts::ForceToDataRelocations && checkMaxDataRelocations()) ||
      BC->isRISCV())
    ForceRelocation = true;

  if (IsFromCode) {
    ContainingBF->addRelocation(Rel.getOffset(), ReferencedSymbol, RType,
                                Addend, ExtractedValue);
  } else if (IsToCode || ForceRelocation) {
    BC->addRelocation(Rel.getOffset(), ReferencedSymbol, RType, Addend,
                      ExtractedValue);
  } else {
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: ignoring relocation from data to data\n");
  }
}

void RewriteInstance::selectFunctionsToProcess() {
  // Extend the list of functions to process or skip from a file.
  auto populateFunctionNames = [](cl::opt<std::string> &FunctionNamesFile,
                                  cl::list<std::string> &FunctionNames) {
    if (FunctionNamesFile.empty())
      return;
    std::ifstream FuncsFile(FunctionNamesFile, std::ios::in);
    std::string FuncName;
    while (std::getline(FuncsFile, FuncName))
      FunctionNames.push_back(FuncName);
  };
  populateFunctionNames(opts::FunctionNamesFile, opts::ForceFunctionNames);
  populateFunctionNames(opts::SkipFunctionNamesFile, opts::SkipFunctionNames);
  populateFunctionNames(opts::FunctionNamesFileNR, opts::ForceFunctionNamesNR);

  // Make a set of functions to process to speed up lookups.
  std::unordered_set<std::string> ForceFunctionsNR(
      opts::ForceFunctionNamesNR.begin(), opts::ForceFunctionNamesNR.end());

  if ((!opts::ForceFunctionNames.empty() ||
       !opts::ForceFunctionNamesNR.empty()) &&
      !opts::SkipFunctionNames.empty()) {
    errs() << "BOLT-ERROR: cannot select functions to process and skip at the "
              "same time. Please use only one type of selection.\n";
    exit(1);
  }

  uint64_t LiteThresholdExecCount = 0;
  if (opts::LiteThresholdPct) {
    if (opts::LiteThresholdPct > 100)
      opts::LiteThresholdPct = 100;

    std::vector<const BinaryFunction *> TopFunctions;
    for (auto &BFI : BC->getBinaryFunctions()) {
      const BinaryFunction &Function = BFI.second;
      if (ProfileReader->mayHaveProfileData(Function))
        TopFunctions.push_back(&Function);
    }
    llvm::sort(
        TopFunctions, [](const BinaryFunction *A, const BinaryFunction *B) {
          return A->getKnownExecutionCount() < B->getKnownExecutionCount();
        });

    size_t Index = TopFunctions.size() * opts::LiteThresholdPct / 100;
    if (Index)
      --Index;
    LiteThresholdExecCount = TopFunctions[Index]->getKnownExecutionCount();
    outs() << "BOLT-INFO: limiting processing to functions with at least "
           << LiteThresholdExecCount << " invocations\n";
  }
  LiteThresholdExecCount = std::max(
      LiteThresholdExecCount, static_cast<uint64_t>(opts::LiteThresholdCount));

  StringSet<> ReorderFunctionsUserSet;
  StringSet<> ReorderFunctionsLTOCommonSet;
  if (opts::ReorderFunctions == ReorderFunctions::RT_USER) {
    for (const std::string &Function :
         ReorderFunctions::readFunctionOrderFile()) {
      ReorderFunctionsUserSet.insert(Function);
      if (std::optional<StringRef> LTOCommonName = getLTOCommonName(Function))
        ReorderFunctionsLTOCommonSet.insert(*LTOCommonName);
    }
  }

  uint64_t NumFunctionsToProcess = 0;
  auto mustSkip = [&](const BinaryFunction &Function) {
    if (opts::MaxFunctions.getNumOccurrences() &&
        NumFunctionsToProcess >= opts::MaxFunctions)
      return true;
    for (std::string &Name : opts::SkipFunctionNames)
      if (Function.hasNameRegex(Name))
        return true;

    return false;
  };

  auto shouldProcess = [&](const BinaryFunction &Function) {
    if (mustSkip(Function))
      return false;

    // If the list is not empty, only process functions from the list.
    if (!opts::ForceFunctionNames.empty() || !ForceFunctionsNR.empty()) {
      // Regex check (-funcs and -funcs-file options).
      for (std::string &Name : opts::ForceFunctionNames)
        if (Function.hasNameRegex(Name))
          return true;

      // Non-regex check (-funcs-no-regex and -funcs-file-no-regex).
      std::optional<StringRef> Match =
          Function.forEachName([&ForceFunctionsNR](StringRef Name) {
            return ForceFunctionsNR.count(Name.str());
          });
      return Match.has_value();
    }

    if (opts::Lite) {
      // Forcibly include functions specified in the -function-order file.
      if (opts::ReorderFunctions == ReorderFunctions::RT_USER) {
        std::optional<StringRef> Match =
            Function.forEachName([&](StringRef Name) {
              return ReorderFunctionsUserSet.contains(Name);
            });
        if (Match.has_value())
          return true;
        for (const StringRef Name : Function.getNames())
          if (std::optional<StringRef> LTOCommonName = getLTOCommonName(Name))
            if (ReorderFunctionsLTOCommonSet.contains(*LTOCommonName))
              return true;
      }

      if (ProfileReader && !ProfileReader->mayHaveProfileData(Function))
        return false;

      if (Function.getKnownExecutionCount() < LiteThresholdExecCount)
        return false;
    }

    return true;
  };

  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;

    // Pseudo functions are explicitly marked by us not to be processed.
    if (Function.isPseudo()) {
      Function.IsIgnored = true;
      Function.HasExternalRefRelocations = true;
      continue;
    }

    // Decide what to do with fragments after parent functions are processed.
    if (Function.isFragment())
      continue;

    if (!shouldProcess(Function)) {
      if (opts::Verbosity >= 1) {
        outs() << "BOLT-INFO: skipping processing " << Function
               << " per user request\n";
      }
      Function.setIgnored();
    } else {
      ++NumFunctionsToProcess;
      if (opts::MaxFunctions.getNumOccurrences() &&
          NumFunctionsToProcess == opts::MaxFunctions)
        outs() << "BOLT-INFO: processing ending on " << Function << '\n';
    }
  }

  if (!BC->HasSplitFunctions)
    return;

  // Fragment overrides:
  // - If the fragment must be skipped, then the parent must be skipped as well.
  // Otherwise, fragment should follow the parent function:
  // - if the parent is skipped, skip fragment,
  // - if the parent is processed, process the fragment(s) as well.
  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    if (!Function.isFragment())
      continue;
    if (mustSkip(Function)) {
      for (BinaryFunction *Parent : Function.ParentFragments) {
        if (opts::Verbosity >= 1) {
          outs() << "BOLT-INFO: skipping processing " << *Parent
                 << " together with fragment function\n";
        }
        Parent->setIgnored();
        --NumFunctionsToProcess;
      }
      Function.setIgnored();
      continue;
    }

    bool IgnoredParent =
        llvm::any_of(Function.ParentFragments, [&](BinaryFunction *Parent) {
          return Parent->isIgnored();
        });
    if (IgnoredParent) {
      if (opts::Verbosity >= 1) {
        outs() << "BOLT-INFO: skipping processing " << Function
               << " together with parent function\n";
      }
      Function.setIgnored();
    } else {
      ++NumFunctionsToProcess;
      if (opts::Verbosity >= 1) {
        outs() << "BOLT-INFO: processing " << Function
               << " as a sibling of non-ignored function\n";
      }
      if (opts::MaxFunctions && NumFunctionsToProcess == opts::MaxFunctions)
        outs() << "BOLT-INFO: processing ending on " << Function << '\n';
    }
  }
}

void RewriteInstance::readDebugInfo() {
  NamedRegionTimer T("readDebugInfo", "read debug info", TimerGroupName,
                     TimerGroupDesc, opts::TimeRewrite);
  if (!opts::UpdateDebugSections)
    return;

  BC->preprocessDebugInfo();
}

void RewriteInstance::preprocessProfileData() {
  if (!ProfileReader)
    return;

  NamedRegionTimer T("preprocessprofile", "pre-process profile data",
                     TimerGroupName, TimerGroupDesc, opts::TimeRewrite);

  outs() << "BOLT-INFO: pre-processing profile using "
         << ProfileReader->getReaderName() << '\n';

  if (BAT->enabledFor(InputFile)) {
    outs() << "BOLT-INFO: profile collection done on a binary already "
              "processed by BOLT\n";
    ProfileReader->setBAT(&*BAT);
  }

  if (Error E = ProfileReader->preprocessProfile(*BC.get()))
    report_error("cannot pre-process profile", std::move(E));

  if (!BC->hasSymbolsWithFileName() && ProfileReader->hasLocalsWithFileName()) {
    errs() << "BOLT-ERROR: input binary does not have local file symbols "
              "but profile data includes function names with embedded file "
              "names. It appears that the input binary was stripped while a "
              "profiled binary was not\n";
    exit(1);
  }
}

void RewriteInstance::initializeMetadataManager() {
  if (opts::LinuxKernelMode)
    MetadataManager.registerRewriter(createLinuxKernelRewriter(*BC));

  MetadataManager.registerRewriter(createPseudoProbeRewriter(*BC));

  MetadataManager.registerRewriter(createSDTRewriter(*BC));
}

void RewriteInstance::processMetadataPreCFG() {
  initializeMetadataManager();

  MetadataManager.runInitializersPreCFG();

  processProfileDataPreCFG();
}

void RewriteInstance::processMetadataPostCFG() {
  MetadataManager.runInitializersPostCFG();
}

void RewriteInstance::processProfileDataPreCFG() {
  if (!ProfileReader)
    return;

  NamedRegionTimer T("processprofile-precfg", "process profile data pre-CFG",
                     TimerGroupName, TimerGroupDesc, opts::TimeRewrite);

  if (Error E = ProfileReader->readProfilePreCFG(*BC.get()))
    report_error("cannot read profile pre-CFG", std::move(E));
}

void RewriteInstance::processProfileData() {
  if (!ProfileReader)
    return;

  NamedRegionTimer T("processprofile", "process profile data", TimerGroupName,
                     TimerGroupDesc, opts::TimeRewrite);

  if (Error E = ProfileReader->readProfile(*BC.get()))
    report_error("cannot read profile", std::move(E));

  if (opts::PrintProfile || opts::PrintAll) {
    for (auto &BFI : BC->getBinaryFunctions()) {
      BinaryFunction &Function = BFI.second;
      if (Function.empty())
        continue;

      Function.print(outs(), "after attaching profile");
    }
  }

  if (!opts::SaveProfile.empty()) {
    YAMLProfileWriter PW(opts::SaveProfile);
    PW.writeProfile(*this);
  }
  if (opts::AggregateOnly &&
      opts::ProfileFormat == opts::ProfileFormatKind::PF_YAML) {
    YAMLProfileWriter PW(opts::OutputFilename);
    PW.writeProfile(*this);
  }

  // Release memory used by profile reader.
  ProfileReader.reset();

  if (opts::AggregateOnly)
    exit(0);
}

void RewriteInstance::disassembleFunctions() {
  NamedRegionTimer T("disassembleFunctions", "disassemble functions",
                     TimerGroupName, TimerGroupDesc, opts::TimeRewrite);
  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;

    ErrorOr<ArrayRef<uint8_t>> FunctionData = Function.getData();
    if (!FunctionData) {
      errs() << "BOLT-ERROR: corresponding section is non-executable or "
             << "empty for function " << Function << '\n';
      exit(1);
    }

    // Treat zero-sized functions as non-simple ones.
    if (Function.getSize() == 0) {
      Function.setSimple(false);
      continue;
    }

    // Offset of the function in the file.
    const auto *FileBegin =
        reinterpret_cast<const uint8_t *>(InputFile->getData().data());
    Function.setFileOffset(FunctionData->begin() - FileBegin);

    if (!shouldDisassemble(Function)) {
      NamedRegionTimer T("scan", "scan functions", "buildfuncs",
                         "Scan Binary Functions", opts::TimeBuild);
      Function.scanExternalRefs();
      Function.setSimple(false);
      continue;
    }

    if (!Function.disassemble()) {
      if (opts::processAllFunctions())
        BC->exitWithBugReport("function cannot be properly disassembled. "
                              "Unable to continue in relocation mode.",
                              Function);
      if (opts::Verbosity >= 1)
        outs() << "BOLT-INFO: could not disassemble function " << Function
               << ". Will ignore.\n";
      // Forcefully ignore the function.
      Function.setIgnored();
      continue;
    }

    if (opts::PrintAll || opts::PrintDisasm)
      Function.print(outs(), "after disassembly");
  }

  BC->processInterproceduralReferences();
  BC->populateJumpTables();

  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;

    if (!shouldDisassemble(Function))
      continue;

    Function.postProcessEntryPoints();
    Function.postProcessJumpTables();
  }

  BC->clearJumpTableTempData();
  BC->adjustCodePadding();

  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;

    if (!shouldDisassemble(Function))
      continue;

    if (!Function.isSimple()) {
      assert((!BC->HasRelocations || Function.getSize() == 0 ||
              Function.hasIndirectTargetToSplitFragment()) &&
             "unexpected non-simple function in relocation mode");
      continue;
    }

    // Fill in CFI information for this function
    if (!Function.trapsOnEntry() && !CFIRdWrt->fillCFIInfoFor(Function)) {
      if (BC->HasRelocations) {
        BC->exitWithBugReport("unable to fill CFI.", Function);
      } else {
        errs() << "BOLT-WARNING: unable to fill CFI for function " << Function
               << ". Skipping.\n";
        Function.setSimple(false);
        continue;
      }
    }

    // Parse LSDA.
    if (Function.getLSDAAddress() != 0 &&
        !BC->getFragmentsToSkip().count(&Function))
      Function.parseLSDA(getLSDAData(), getLSDAAddress());
  }
}

void RewriteInstance::buildFunctionsCFG() {
  NamedRegionTimer T("buildCFG", "buildCFG", "buildfuncs",
                     "Build Binary Functions", opts::TimeBuild);

  // Create annotation indices to allow lock-free execution
  BC->MIB->getOrCreateAnnotationIndex("JTIndexReg");
  BC->MIB->getOrCreateAnnotationIndex("NOP");
  BC->MIB->getOrCreateAnnotationIndex("Size");

  ParallelUtilities::WorkFuncWithAllocTy WorkFun =
      [&](BinaryFunction &BF, MCPlusBuilder::AllocatorIdTy AllocId) {
        if (!BF.buildCFG(AllocId))
          return;

        if (opts::PrintAll) {
          auto L = BC->scopeLock();
          BF.print(outs(), "while building cfg");
        }
      };

  ParallelUtilities::PredicateTy SkipPredicate = [&](const BinaryFunction &BF) {
    return !shouldDisassemble(BF) || !BF.isSimple();
  };

  ParallelUtilities::runOnEachFunctionWithUniqueAllocId(
      *BC, ParallelUtilities::SchedulingPolicy::SP_INST_LINEAR, WorkFun,
      SkipPredicate, "disassembleFunctions-buildCFG",
      /*ForceSequential*/ opts::SequentialDisassembly || opts::PrintAll);

  BC->postProcessSymbolTable();
}

void RewriteInstance::postProcessFunctions() {
  // We mark fragments as non-simple here, not during disassembly,
  // So we can build their CFGs.
  BC->skipMarkedFragments();
  BC->clearFragmentsToSkip();

  BC->TotalScore = 0;
  BC->SumExecutionCount = 0;
  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;

    // Set function as non-simple if it has dynamic relocations
    // in constant island, we don't want this function to be optimized
    // e.g. function splitting is unsupported.
    if (Function.hasDynamicRelocationAtIsland())
      Function.setSimple(false);

    if (Function.empty())
      continue;

    Function.postProcessCFG();

    if (opts::PrintAll || opts::PrintCFG)
      Function.print(outs(), "after building cfg");

    if (opts::DumpDotAll)
      Function.dumpGraphForPass("00_build-cfg");

    if (opts::PrintLoopInfo) {
      Function.calculateLoopInfo();
      Function.printLoopInfo(outs());
    }

    BC->TotalScore += Function.getFunctionScore();
    BC->SumExecutionCount += Function.getKnownExecutionCount();
  }

  if (opts::PrintGlobals) {
    outs() << "BOLT-INFO: Global symbols:\n";
    BC->printGlobalSymbols(outs());
  }
}

void RewriteInstance::runOptimizationPasses() {
  NamedRegionTimer T("runOptimizationPasses", "run optimization passes",
                     TimerGroupName, TimerGroupDesc, opts::TimeRewrite);
  BinaryFunctionPassManager::runAllPasses(*BC);
}

void RewriteInstance::renameAndPreregisterSections() {

  auto Rename = [this](BinarySection *Section) {
    if (!Section)
      return;
    std::string OldName = Section->getName().str();
    const std::string NewName = (getOrgSecPrefix() + Section->getName()).str();
    LLVM_DEBUG(dbgs() << "BOLT-DEBUG: renaming input section "
                      << Section->getName() << " to " << NewName << "\n";);
    BC->renameSection(*Section, NewName);
    BC->registerOrUpdateSection(OldName, ELF::SHT_PROGBITS,
                                Section->getELFFlags());
  };
  if (!opts::Rewrite) {
    if (BC->HasRelocations)
      Rename(getSection(BC->getMainCodeSectionName()));
    Rename(getSection(getEHFrameSectionName()));
    if (EHFrameSection)
      BC->registerSection(".relocated.eh_frame", *EHFrameSection);
    Rename(getSection(".eh_frame_hdr"));
    Rename(getSection(".gcc_except_table"));
  }
  if (opts::JumpTables > JTS_BASIC)
    Rename(getSection(".rodata"));
}

void RewriteInstance::emitAndLink() {
  NamedRegionTimer T("emitAndLink", "emit and link", TimerGroupName,
                     TimerGroupDesc, opts::TimeRewrite);

  SmallString<0> ObjectBuffer;
  raw_svector_ostream OS(ObjectBuffer);

  // Implicitly MCObjectStreamer takes ownership of MCAsmBackend (MAB)
  // and MCCodeEmitter (MCE). ~MCObjectStreamer() will delete these
  // two instances.
  std::unique_ptr<MCStreamer> Streamer = BC->createStreamer(OS);

  if (EHFrameSection) {
    if (opts::Rewrite || opts::StrictMode) {
      // The section is going to be regenerated from scratch.
      // Empty the contents, but keep the section reference.
      EHFrameSection->clearContents();
    } else {
      // Make .eh_frame relocatable.
      relocateEHFrameSection();
    }
  }

  emitBinaryContext(*Streamer, *BC, getOrgSecPrefix());

  Streamer->finish();
  if (Streamer->getContext().hadError()) {
    errs() << "BOLT-ERROR: Emission failed.\n";
    exit(1);
  }

  if (opts::KeepTmp) {
    SmallString<128> OutObjectPath;
    sys::fs::getPotentiallyUniqueTempFileName("output", "o", OutObjectPath);
    std::error_code EC;
    raw_fd_ostream FOS(OutObjectPath, EC);
    check_error(EC, "cannot create output object file");
    FOS << ObjectBuffer;
    outs() << "BOLT-INFO: intermediary output object file saved for debugging "
              "purposes: "
           << OutObjectPath << "\n";
  }

  //////////////////////////////////////////////////////////////////////////////
  // Assign addresses to new sections.
  //////////////////////////////////////////////////////////////////////////////

  // Get output object as ObjectFile.
  std::unique_ptr<MemoryBuffer> ObjectMemBuffer =
      MemoryBuffer::getMemBuffer(ObjectBuffer, "in-memory object file", false);

  auto EFMM = std::make_unique<ExecutableFileMemoryManager>(*BC);

  Linker = std::make_unique<JITLinkLinker>(*BC, std::move(EFMM));
  Linker->loadObject(
      ObjectMemBuffer->getMemBufferRef(),
      [this](auto AssignAddress) { mapAllocatableSections(AssignAddress); });

  MCAsmLayout FinalLayout(
      static_cast<MCObjectStreamer *>(Streamer.get())->getAssembler());

  // Update output addresses based on the new section map and
  // layout. Only do this for the object created by ourselves.
  updateOutputValues(FinalLayout);

  if (opts::UpdateDebugSections)
    DebugInfoRewriter->updateLineTableOffsets(FinalLayout);

  if (RuntimeLibrary *RtLibrary = BC->getRuntimeLibrary())
    RtLibrary->link(*BC, ToolPath, *Linker, [this](auto AssignAddress) {
      // Map newly registered sections.
      this->mapRuntimeLibrary(AssignAddress);
    });

  if (opts::Rewrite) {
    // map the first non-allocatable address to the first non-allocatable
    // offset so that BSS doesn't occupy file space.
    FirstNonAllocatableOffset =
        BC->OutputSegments.back().p_offset + BC->OutputSegments.back().p_filesz;
    BC->OutputAddressToOffsetMap[NextAvailableAddress] =
        FirstNonAllocatableOffset;
  }
  mapNonLoadableSegments();
  // Once the code is emitted, we can rename function sections to actual
  // output sections and de-register sections used for emission.
  for (BinaryFunction *Function : BC->getAllBinaryFunctions()) {
    ErrorOr<BinarySection &> Section = Function->getCodeSection();
    if (Section &&
        (Function->getImageAddress() == 0 || Function->getImageSize() == 0))
      continue;

    // Restore origin section for functions that were emitted or supposed to
    // be emitted to patch sections.
    if (Section)
      BC->deregisterSection(*Section);
    assert(Function->getOriginSectionName() && "expected origin section");
    Function->CodeSectionName = Function->getOriginSectionName()->str();
    for (const FunctionFragment &FF :
         Function->getLayout().getSplitFragments()) {
      if (ErrorOr<BinarySection &> ColdSection =
              Function->getCodeSection(FF.getFragmentNum()))
        BC->deregisterSection(*ColdSection);
    }
    if (Function->getLayout().isSplit())
      Function->setColdCodeSectionName(getBOLTTextSectionName());
  }

  if (opts::PrintCacheMetrics) {
    outs() << "BOLT-INFO: cache metrics after emitting functions:\n";
    CacheMetrics::printAll(BC->getSortedFunctions());
  }
}

void RewriteInstance::updateMetadata() {
  MetadataManager.runFinalizersAfterEmit();

  if (opts::UpdateDebugSections) {
    NamedRegionTimer T("updateDebugInfo", "update debug info", TimerGroupName,
                       TimerGroupDesc, opts::TimeRewrite);
    DebugInfoRewriter->updateDebugInfo();
  }

  if (opts::WriteBoltInfoSection)
    addBoltInfoSection();
}

void RewriteInstance::mapAllocatableSections(
    BOLTLinker::SectionMapper AssignAddress) {
  if (opts::Rewrite) {
    // Assign new address to every section and
    // put new sections in the end of old segments,
    // matching sections with segments by their flags.
    mapLoadableSegments(AssignAddress);
  } else {
    // Reassign old addresses to old sections and segments,
    // and create new segments for new sections.
    remapLoadableSegments(AssignAddress);
  }
  ProgramHeader LastSegment = BC->OutputSegments.back();
  const uint64_t EndOfLastSegmentInMem =
      LastSegment.p_vaddr + LastSegment.p_memsz;
  const uint64_t EndOfLastSegmentInFile =
      LastSegment.p_offset + LastSegment.p_filesz;
  for (BinarySection &Section : BC->allocatableSections()) {
    if (!Section.getOutputAddress()) {
      // because of strip, zero-sized sections such as .bss may end up outside
      // of any segments and stay unmapped. Just put them in end of the last
      // segment.
      if (Section.getInputFileOffset() && Section.getSize() == 0) {
        dbgs() << formatv("BOLT-WARNING: zero-sized allocatable section {0} is "
                          "outside of any segment.\n",
                          Section.getName());

        Section.setOutputAddress(EndOfLastSegmentInMem);
        Section.setOutputFileOffset(EndOfLastSegmentInFile);
        if (Section.hasValidSectionID())
          AssignAddress(Section, Section.getOutputAddress());
        Section.setIsFinalized();
      }
    }
  }
}

void RewriteInstance::remapLoadableSegments(
    BOLTLinker::SectionMapper AssignAddress) {
  BC->deregisterUnusedSections();

  auto RestoreName = [this](BinarySection *Section, StringRef Name) {
    if (!Section)
      return;
    LLVM_DEBUG(dbgs() << formatv("BOLT-DEBUG: original section {0} was "
                                 "not duplicated, restoring name\n",
                                 Name));
    assert(!getSection(Name) && "Unexpected section emitted");
    BC->renameSection(*Section, Name);
  };
  // If no new .eh_frame was written, remove relocated original .eh_frame.
  BinarySection *RelocatedEHFrameSection =
      getSection(".relocated" + getEHFrameSectionName());
  if (RelocatedEHFrameSection && RelocatedEHFrameSection->hasValidSectionID()) {
    BinarySection *NewEHFrameSection = getSection(getEHFrameSectionName());
    if (!NewEHFrameSection || !NewEHFrameSection->isFinalized()) {
      // JITLink will still have to process relocations for the section, hence
      // we need to assign it the address that wouldn't result in relocation
      // processing failure.
      AssignAddress(*RelocatedEHFrameSection, NextAvailableAddress);
      BC->deregisterSection(*RelocatedEHFrameSection);

      // Remove org prefix from sections since they weren't duplicated
      RestoreName(&*EHFrameSection, getEHFrameSectionName());
      RestoreName(getSection(getOrgSecPrefix() + getEHFrameHeaderSectionName()),
                  getEHFrameHeaderSectionName());
      RestoreName(getSection(getOrgSecPrefix() + ".gcc_except_table"),
                  ".gcc_except_table");
    }
  }

  if (BC->HasRelocations) {
    // Map sections for functions with pre-assigned addresses.
    for (BinaryFunction *InjectedFunction : BC->getInjectedBinaryFunctions()) {
      const uint64_t OutputAddress = InjectedFunction->getOutputAddress();
      if (!OutputAddress)
        continue;

      ErrorOr<BinarySection &> FunctionSection =
          InjectedFunction->getCodeSection();
      assert(FunctionSection && "function should have section");
      FunctionSection->setOutputAddress(OutputAddress);
      AssignAddress(*FunctionSection, OutputAddress);
      InjectedFunction->setImageAddress(FunctionSection->getAllocAddress());
      InjectedFunction->setImageSize(FunctionSection->getOutputSize());
    }
  }

  for (const ProgramHeader &Phdr : BC->loadableSegments()) {
    copySegment(Phdr, AssignAddress);
  }

  assert(NextAvailableAddress && "Uninitialized NextAvailableAddress!");
  assert(PHDRTableAddress && PHDRTableOffset &&
         "Uninitialized PHDRTable location!");
  BC->OutputSegments.insert(
      BC->OutputSegments.begin(),
      ProgramHeader(ELF::PT_PHDR, ELF::PF_R, PHDRTableOffset, PHDRTableAddress,
                    PHDRTableAddress,
                    /* file/mem size will be adjusted in writeELFPHDRTable*/ 0,
                    0, 0x8));

  if (!BC->HasRelocations) {
    mapFunctionsNonRelocMode(AssignAddress);
  }

  // if we instrument without rewriting, we want a single segment at
  // PHDRTableAddress that will be created in mapRuntimeLibrary
  if (BC->getRuntimeLibrary())
    mapSectionGroup(BC->getAllNewSections(), AssignAddress);
  else {
    createLoadSegment(
        BC->getNewSectionsByFlags(ELF::SHF_ALLOC | ELF::SHF_EXECINSTR,
                                  /*ROwithRX*/ true),
        AssignAddress, ELF::PF_R | ELF::PF_X, BC->PageAlign, PHDRTableAddress);
    // In case writable sections were emitted, put them in a separate segment.
    createLoadSegment(
        BC->getNewSectionsByFlags(ELF::SHF_ALLOC | ELF::SHF_WRITE),
        AssignAddress, ELF::PF_R | ELF::PF_W, BC->PageAlign);
  }
}

// Processing in non-relocation mode.
void RewriteInstance::mapFunctionsNonRelocMode(
    BOLTLinker::SectionMapper AssignAddress) {
  uint64_t NewTextSectionStartAddress = NextAvailableAddress;

  for (auto &BFI : BC->getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    if (!Function.isEmitted())
      continue;

    bool TooLarge = false;
    ErrorOr<BinarySection &> FuncSection = Function.getCodeSection();
    assert(FuncSection && "cannot find section for function");
    FuncSection->setOutputAddress(Function.getAddress());
    LLVM_DEBUG(dbgs() << formatv(
                   "BOLT-DEBUG: mapping function {0} at {1:x} to {2:x}\n",
                   Function.getOneName(), FuncSection->getAllocAddress(),
                   Function.getAddress()));

    AssignAddress(*FuncSection, Function.getAddress());
    Function.setImageAddress(FuncSection->getAllocAddress());
    Function.setImageSize(FuncSection->getOutputSize());
    if (Function.getImageSize() > Function.getMaxSize()) {
      TooLarge = true;
      FailedAddresses.emplace_back(Function.getAddress());
    }

    // Map jump tables if updating in-place.
    if (opts::JumpTables == JTS_BASIC) {
      for (auto &JTI : Function.JumpTables) {
        JumpTable *JT = JTI.second;
        BinarySection &Section = JT->getOutputSection();
        Section.setOutputAddress(JT->getAddress());
        Section.setOutputFileOffset(getFileOffsetForAddress(JT->getAddress()));
        LLVM_DEBUG(dbgs() << "BOLT-DEBUG: mapping JT " << Section.getName()
                          << " to 0x" << Twine::utohexstr(JT->getAddress())
                          << '\n');
        AssignAddress(Section, JT->getAddress());
      }
    }

    if (!Function.isSplit())
      continue;

    assert(Function.getLayout().isHotColdSplit() &&
           "Cannot allocate more than two fragments per function in "
           "non-relocation mode.");

    FunctionFragment &FF =
        Function.getLayout().getFragment(FragmentNum::cold());
    ErrorOr<BinarySection &> ColdSection =
        Function.getCodeSection(FF.getFragmentNum());
    assert(ColdSection && "cannot find section for cold part");
    // Cold fragments are aligned at 16 bytes.
    NextAvailableAddress = alignTo(NextAvailableAddress, 16);
    if (TooLarge) {
      // The corresponding FDE will refer to address 0.
      FF.setAddress(0);
      FF.setImageAddress(0);
      FF.setImageSize(0);
      FF.setFileOffset(0);
    } else {
      FF.setAddress(NextAvailableAddress);
      FF.setImageAddress(ColdSection->getAllocAddress());
      FF.setImageSize(ColdSection->getOutputSize());
      FF.setFileOffset(getOutputFileOffsetForAddress(NextAvailableAddress));
      ColdSection->setOutputAddress(FF.getAddress());
    }

    LLVM_DEBUG(
        dbgs() << formatv(
            "BOLT: mapping cold fragment {0:x+} to {1:x+} with size {2:x+}\n",
            FF.getImageAddress(), FF.getAddress(), FF.getImageSize()));
    AssignAddress(*ColdSection, FF.getAddress());

    if (TooLarge)
      BC->deregisterSection(*ColdSection);

    NextAvailableAddress += FF.getImageSize();
  }

  // Add the new text section aggregating all existing code sections.
  // This is pseudo-section that serves a purpose of creating a corresponding
  // entry in section header table.
  int64_t NewTextSectionSize =
      NextAvailableAddress - NewTextSectionStartAddress;
  if (NewTextSectionSize) {
    const unsigned Flags = BinarySection::getFlags(/*IsReadOnly=*/true,
                                                   /*IsText=*/true,
                                                   /*IsAllocatable=*/true);
    BinarySection &Section =
      BC->registerOrUpdateSection(getBOLTTextSectionName(),
                                  ELF::SHT_PROGBITS,
                                  Flags,
                                  /*Data=*/nullptr,
                                  NewTextSectionSize,
                                  16);
    Section.setOutputAddress(NewTextSectionStartAddress);
    Section.setOutputFileOffset(
        getOutputFileOffsetForAddress(NewTextSectionStartAddress));
  }
}

void RewriteInstance::createNonLoadSegment(
    const std::vector<BinarySection *> &Sections, const unsigned p_type,
    const unsigned p_flags, const unsigned p_align) {
  assert(p_type != ELF::PT_LOAD &&
         "Called createNonLoadSegment with PT_LOAD type");
  if (p_type == ELF::PT_GNU_STACK) {
    assert(Sections.empty() && "Unexpected sections in GNU_STACK segment");
    BC->OutputSegments.push_back(
        ProgramHeader(ELF::PT_GNU_STACK, p_flags, 0, 0, 0, 0, 0, p_align));
    return;
  }
  if (Sections.empty())
    return;

  bool Mapped = all_of(Sections, std::bind(&BinarySection::getOutputAddress,
                                           std::placeholders::_1)) &&
                all_of(Sections, std::bind(&BinarySection::getOutputFileOffset,
                                           std::placeholders::_1));
  assert(Mapped && "attempt to create non-load segment with unmapped sections");
  (void)Mapped;
  const uint64_t p_vaddr = Sections.front()->getOutputAddress();
  const uint64_t p_memsz = Sections.back()->getOutputAddress() +
                           Sections.back()->getOutputSize() - p_vaddr;
  const uint64_t p_filesz =
      p_memsz - Sections.back()->isVirtual() * Sections.back()->getOutputSize();
  const uint64_t p_offset = Sections.front()->getOutputFileOffset();
  ProgramHeader Result(p_type, p_flags, p_offset, p_vaddr, p_vaddr, p_filesz,
                       p_memsz, p_align);

  if (p_type == ELF::PT_INTERP) {
    assert(BC->OutputSegments.size() && "PHDR segment expected");
    // Put INTERP after PHDR because it must precede loadable segments
    BC->OutputSegments.insert(BC->OutputSegments.begin() + 1, Result);
  } else {
    BC->OutputSegments.push_back(Result);
  }
}

void RewriteInstance::mapSection(BinarySection &Section) {
  assert(!Section.getOutputAddress());
  if (!Section.isTBSS())
    NextAvailableAddress =
        alignTo(NextAvailableAddress, Section.getAlignment());

  Section.setOutputAddress(NextAvailableAddress);
  Section.setOutputFileOffset(
      getOutputFileOffsetForAddress(Section.getOutputAddress()));

  if (!Section.isTBSS())
    NextAvailableAddress += Section.getOutputSize();
  Section.setIsFinalized();
  LLVM_DEBUG({
    std::string What = Section.hasSectionRef() ? "original" : "new";
    dbgs() << formatv(
        "BOLT-DEBUG: mapping {0} section {1} ({2:x}) to {3:x}:{4:x}\n", What,
        Section.getName(), Section.getAllocAddress(),
        Section.getOutputAddress(),
        Section.getOutputAddress() + Section.getOutputSize());
  });
  // Hugify: Additional huge page from right side due to
  // weird ASLR mapping addresses (4KB aligned)
  if (opts::Hugify && Section.getOutputName() == BC->getMainCodeSectionName())
    NextAvailableAddress =
        alignTo(NextAvailableAddress, Section.getAlignment());
}

uint64_t
RewriteInstance::mapSectionGroup(const std::vector<BinarySection *> &Sections,
                                 BOLTLinker::SectionMapper AssignAddress) {
  uint64_t NobitsSize = 0;
  for (BinarySection *Section : Sections) {
    if (Section->getOutputAddress()) {
      LLVM_DEBUG({
        dbgs() << formatv(
            "BOLT-DEBUG: section {0} is already mapped at {1:x}\n",
            Section->getName(), Section->getOutputAddress());
      });
      continue;
    }
    // .eh_frame_hdr address and size are assigned at mapEhFrames
    if (Section->getName() == getEHFrameHeaderSectionName())
      continue;
    if (Section->getName().endswith(getEHFrameSectionName())) {
      mapEhFrameAndHeader(AssignAddress);
      continue;
    }

    mapSection(*Section);
    if (Section->isVirtual() && !Section->isTLS())
      NobitsSize += Section->getOutputSize();
    if (Section->hasValidSectionID())
      AssignAddress(*Section, Section->getOutputAddress());
  }
  return NobitsSize;
}
void RewriteInstance::createLoadSegment(
    const std::vector<BinarySection *> &Sections,
    BOLTLinker::SectionMapper AssignAddress, const unsigned p_flags,
    const unsigned p_align, const std::optional<uint64_t> ForceAddress) {

  if (Sections.empty())
    return;
  if (!ForceAddress) {
    NextAvailableAddress = alignTo(NextAvailableAddress, p_align);
    if (BC->OutputSegments.size() && BC->OutputSegments.back().isLOAD()) {
      const ProgramHeader &PrevSegment = BC->OutputSegments.back();
      NextAvailableAddress = alignTo(NextAvailableAddress, PrevSegment.p_align);
      const uint64_t Align =
          Sections.size() ? Sections.front()->getAlignment() : 8;
      const uint64_t SegmentStartOffset =
          alignTo(PrevSegment.p_offset + PrevSegment.p_memsz, Align);
      NextAvailableAddress += SegmentStartOffset & (p_align - 1);
      BC->OutputAddressToOffsetMap[NextAvailableAddress] = SegmentStartOffset;
    }
  }
  uint64_t p_vaddr = NextAvailableAddress;
  uint64_t p_offset = getOutputFileOffsetForAddress(NextAvailableAddress);
  LLVM_DEBUG({ dbgs() << "BOLT-DEBUG: creating LOAD segment\n"; });
  const uint64_t NobitsSize = mapSectionGroup(Sections, AssignAddress);
  uint64_t p_memsz = NextAvailableAddress - p_vaddr;
  if (!p_memsz)
    return;
  if (ForceAddress) {
    p_memsz += p_vaddr - *ForceAddress;
    p_offset = getOutputFileOffsetForAddress(*ForceAddress);
    p_vaddr = *ForceAddress;
  }
  uint64_t p_filesz = p_memsz - NobitsSize;
  ProgramHeader Result = ProgramHeader(ELF::PT_LOAD, p_flags, p_offset, p_vaddr,
                                       p_vaddr, p_filesz, p_memsz, p_align);
  BC->OutputSegments.push_back(Result);
}

std::vector<BinarySection *>
RewriteInstance::getSectionsForSegment(const ProgramHeader &Phdr,
                                       bool IncludeNew) {
  std::vector<BinarySection *> Result;
  std::vector<BinarySection *> Nobits;
  for (BinarySection &Section : BC->allocatableSections()) {
    if (Phdr.isTLS() > Section.isTLS())
      continue;
    if (Phdr.contains(Section) && !shouldStrip(Section)) {
      if (Section.isVirtual() && !Section.isTLS())
        Nobits.push_back(&Section);
      else
        Result.push_back(&Section);
    }
  }

  auto GetSectionIt = [&Result](StringRef Name) {
    return std::find_if(
        Result.begin(), Result.end(),
        [Name](BinarySection *Sec) { return Sec->getOutputName() == Name; });
  };

  if (IncludeNew && Phdr.isLOAD()) {
    std::vector<BinarySection *> Extra = BC->getNewSectionsByFlags(
        Phdr.getSectionFlags(), /*ROwithRX*/ !HasReadOnlySegment);
    // We need to put .text.cold after .text
    auto PosForNewSections = Result.end();
    if (Phdr.isExec()) {
      PosForNewSections = GetSectionIt(".text");
      if (PosForNewSections != Result.end())
        ++PosForNewSections;
    }
    Result.insert(PosForNewSections, Extra.begin(), Extra.end());
  }
  if (opts::Rewrite && Phdr.isExec()) {
    // Put .plt before .text because:
    // - it makes clear where PLT will be relative to .text, which simplifies
    // stub insertion for AArch64. Without it we can't decide where to insert
    // stubs because we don't know the final binary layout before linking.
    // - unless --hot-functions-at-end is used, PLT comes just before hot text
    // section which increases locality.
    auto PltIt = GetSectionIt(".plt");
    if (PltIt != Result.end()) {
      BinarySection *Plt = *PltIt;
      Result.erase(PltIt);
      auto TextIt = GetSectionIt(".text");
      assert(TextIt != Result.end() && "No .text section!");
      Result.insert(TextIt, Plt);
    }
  }
  Result.insert(Result.end(), Nobits.begin(), Nobits.end());
  return Result;
}
void RewriteInstance::mapNonLoadableSegments() {
  for (const ProgramHeader &Phdr : BC->nonLoadableSegments()) {
    // PHDR is created earlier
    if (Phdr.p_type == ELF::PT_PHDR)
      continue;
    if (!opts::Rewrite && Phdr.p_type == ELF::PT_GNU_EH_FRAME) {
      BinarySection *EHFrameHeader = getSection(getEHFrameHeaderSectionName());
      assert(EHFrameHeader && "Cannot find .eh_frame_hdr for PT_GNU_EH_FRAME!");
      createNonLoadSegment({EHFrameHeader}, Phdr.p_type, Phdr.p_flags,
                           Phdr.p_align);
    } else {
      createNonLoadSegment(getSectionsForSegment(Phdr), Phdr.p_type,
                           Phdr.p_flags, Phdr.p_align);
    }
  }
}

void RewriteInstance::mapLoadableSegments(
    BOLTLinker::SectionMapper AssignAddress) {
  BC->deregisterUnusedSections();

  PHDRTableOffset = 0x40;
  NextAvailableAddress = BaseAddress + PHDRTableOffset;
  PHDRTableAddress = NextAvailableAddress;
  assert(BC->OutputAddressToOffsetMap.empty() && "Unexpected mapping!");

  BC->OutputAddressToOffsetMap[BaseAddress] = 0;
  // add PHDR and reserve space for RE/RW segments for runtime library
  uint64_t Phnum = BC->InputSegments.size() + !HasProgramHeaderSegment +
                   (!!BC->getRuntimeLibrary()) * 2;
  uint64_t PHDRTableSize = Phnum * sizeof(ELF64LE::Phdr);
  BC->MaxPHDRSize = PHDRTableSize;
  BC->OutputSegments.push_back(
      ProgramHeader(ELF::PT_PHDR, ELF::PF_R, PHDRTableOffset, PHDRTableAddress,
                    PHDRTableAddress, PHDRTableSize, PHDRTableSize, 0x8));
  NextAvailableAddress += PHDRTableSize;

  for (const ProgramHeader &Phdr : BC->loadableSegments()) {

    const uint64_t Align = Phdr.isExec() ? BC->PageAlign : Phdr.p_align;
    bool IsFirstLoad = (Phdr.p_offset == 0);
    createLoadSegment(getSectionsForSegment(Phdr), AssignAddress, Phdr.p_flags,
                      Align,
                      /*ForceAddress=*/
                      IsFirstLoad ? std::optional(BaseAddress) : std::nullopt);
  }
}
void RewriteInstance::copySegment(const ProgramHeader &Segment,
                                  BOLTLinker::SectionMapper AssignAddress) {
  BC->OutputSegments.push_back(Segment);
  BC->OutputAddressToOffsetMap[Segment.p_vaddr] = Segment.p_offset;
  for (BinarySection *Section : getSectionsForSegment(Segment, false)) {

    Section->setOutputAddress(Section->getAddress());
    Section->setOutputFileOffset(Section->getInputFileOffset());
    if (Section->hasValidSectionID())
      AssignAddress(*Section, Section->getOutputAddress());
    Section->setIsFinalized();
  }
}

void RewriteInstance::mapRuntimeLibrary(
    BOLTLinker::SectionMapper AssignAddress) {
  if (opts::Rewrite) {
    // if we are rewriting, we can afford two segments with proper flags.
    createLoadSegment(
        BC->getNewSectionsByFlags(ELF::SHF_ALLOC | ELF::SHF_EXECINSTR,
                                  /*ROwithRX*/ true),
        AssignAddress, ELF::PF_R | ELF::PF_X, BC->RegularPageSize);
    createLoadSegment(
        BC->getNewSectionsByFlags(ELF::SHF_ALLOC | ELF::SHF_WRITE),
        AssignAddress, ELF::PF_R | ELF::PF_W, BC->RegularPageSize);
  } else {
    auto NewSegmentContents = BC->getAllNewSections();
    unsigned Flags = ELF::PF_R | ELF::PF_X | ELF::PF_W;
    createLoadSegment(NewSegmentContents, AssignAddress, Flags, BC->PageAlign,
                      PHDRTableAddress);
  }
}

void RewriteInstance::updateOutputValues(const MCAsmLayout &Layout) {
  for (BinaryFunction *Function : BC->getAllBinaryFunctions())
    Function->updateOutputValues(Layout);
}

void RewriteInstance::writeELFPHDRTable() {
  raw_fd_ostream &OS = Out->os();
  assert(PHDRTableOffset && "No PHDR table offset!");
  assert(BC->OutputSegments.size() && "Empty PHDR!");
  assert(BC->OutputSegments[0].p_type == ELF::PT_PHDR && "No PHDR segment!");
  OS.seek(PHDRTableOffset);
  uint64_t PHDRTableSize = BC->OutputSegments.size() * sizeof(ELF64LE::Phdr);
  assert(PHDRTableSize <= BC->MaxPHDRSize && "Exceeded MaxPHDRSize!");
  BC->OutputSegments[0].p_filesz = PHDRTableSize;
  BC->OutputSegments[0].p_memsz = PHDRTableSize;
  for (const auto &Phdr : BC->OutputSegments) {
    ELF64LE::Phdr PhdrToWrite(Phdr);
    OS.write(reinterpret_cast<const char *>(&PhdrToWrite), sizeof(PhdrToWrite));
  }
  return;
}

namespace {

/// Write padding to \p OS such that its current \p Offset becomes aligned
/// at \p Alignment. Return new (aligned) offset.
uint64_t appendPadding(raw_pwrite_stream &OS, uint64_t Offset,
                       uint64_t Alignment) {
  if (!Alignment)
    return Offset;

  const uint64_t PaddingSize =
      offsetToAlignment(Offset, llvm::Align(Alignment));
  for (unsigned I = 0; I < PaddingSize; ++I)
    OS.write((unsigned char)0);
  return Offset + PaddingSize;
}

}

void RewriteInstance::rewriteNoteSections() {
  auto ELF64LEFile = cast<ELF64LEObjectFile>(InputFile);
  const ELFFile<ELF64LE> &Obj = ELF64LEFile->getELFFile();
  raw_fd_ostream &OS = Out->os();

  uint64_t NextAvailableOffset =
      getOutputFileOffsetForAddress(NextAvailableAddress);
  assert(NextAvailableOffset >= FirstNonAllocatableOffset &&
         "next available offset calculation failure");
  OS.seek(NextAvailableOffset);

  // Copy over non-allocatable section contents and update file offsets.
  for (const ELF64LE::Shdr &Section : cantFail(Obj.sections())) {
    if (Section.sh_type == ELF::SHT_NULL)
      continue;
    if (Section.sh_flags & ELF::SHF_ALLOC)
      continue;

    SectionRef SecRef = ELF64LEFile->toSectionRef(&Section);
    BinarySection *BSec = BC->getSectionForSectionRef(SecRef);
    assert(BSec && !BSec->isAllocatable() &&
           "Matching non-allocatable BinarySection should exist.");

    StringRef SectionName =
        cantFail(Obj.getSectionName(Section), "cannot get section name");
    if (shouldStrip(Section, SectionName))
      continue;

    // Insert padding as needed.
    NextAvailableOffset =
        appendPadding(OS, NextAvailableOffset, Section.sh_addralign);

    // New section size.
    uint64_t Size = 0;
    bool DataWritten = false;
    uint8_t *SectionData = nullptr;
    // Copy over section contents unless it's one of the sections we overwrite.
    if (!willOverwriteSection(SectionName)) {
      Size = Section.sh_size;
      StringRef Dataref = InputFile->getData().substr(Section.sh_offset, Size);
      std::string Data;
      if (BSec->getPatcher()) {
        Data = BSec->getPatcher()->patchBinary(Dataref);
        Dataref = StringRef(Data);
      }

      // Section was expanded, so need to treat it as overwrite.
      if (Size != Dataref.size()) {
        BSec = &BC->registerOrUpdateNoteSection(
            SectionName, copyByteArray(Dataref), Dataref.size());
        Size = 0;
      } else {
        OS << Dataref;
        DataWritten = true;

        // Add padding as the section extension might rely on the alignment.
        Size = appendPadding(OS, Size, Section.sh_addralign);
      }
    }

    // Perform section post-processing.
    assert(BSec->getAlignment() <= Section.sh_addralign &&
           "alignment exceeds value in file");

    if (BSec->getAllocAddress() && !DataWritten) {
      assert(!DataWritten && "Writing section twice.");
      (void)DataWritten;
      SectionData = BSec->getOutputData();

      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: " << (Size ? "appending" : "writing")
                        << " contents to section " << SectionName << '\n');
      OS.write(reinterpret_cast<char *>(SectionData), BSec->getOutputSize());
      Size += BSec->getOutputSize();
    }

    BSec->setOutputFileOffset(NextAvailableOffset);
    BSec->flushPendingRelocations(OS, [this](const MCSymbol *S) {
      return getNewValueForSymbol(S->getName());
    });

    // Set/modify section info.
    BinarySection &NewSection = BC->registerOrUpdateNoteSection(
        SectionName, SectionData, Size, Section.sh_addralign,
        !BSec->isWritable(), BSec->getELFType());
    NewSection.setOutputAddress(0);
    NewSection.setOutputFileOffset(NextAvailableOffset);

    NextAvailableOffset += Size;
  }

  // Write new note sections.
  for (BinarySection &Section : BC->nonAllocatableSections()) {
    if (Section.getOutputFileOffset() || !Section.getAllocAddress() ||
        shouldStrip(Section))
      continue;

    assert(!Section.hasPendingRelocations() && "cannot have pending relocs");

    NextAvailableOffset =
        appendPadding(OS, NextAvailableOffset, Section.getAlignment());
    Section.setOutputFileOffset(NextAvailableOffset);

    LLVM_DEBUG(
        dbgs() << "BOLT-DEBUG: writing out new section " << Section.getName()
               << " of size " << Section.getOutputSize() << " at offset 0x"
               << Twine::utohexstr(Section.getOutputFileOffset()) << '\n');

    OS.write(Section.getOutputContents().data(), Section.getOutputSize());
    NextAvailableOffset += Section.getOutputSize();
  }
}

template <typename ELFT>
void RewriteInstance::finalizeSectionStringTable(ELFObjectFile<ELFT> *File) {
  // Pre-populate section header string table.
  for (const BinarySection &Section : BC->sections())
    if (!Section.isAnonymous()) {
      SHStrTab.add(Section.getOutputName());
      LLVM_DEBUG(dbgs() << "BOLT-DEBUG: adding " << Section.getOutputName()
                        << " to shstrtab\n";);
    }

  SHStrTab.finalize();

  const size_t SHStrTabSize = SHStrTab.getSize();
  uint8_t *DataCopy = new uint8_t[SHStrTabSize];
  memset(DataCopy, 0, SHStrTabSize);
  SHStrTab.write(DataCopy);
  BC->registerOrUpdateNoteSection(".shstrtab",
                                  DataCopy,
                                  SHStrTabSize,
                                  /*Alignment=*/1,
                                  /*IsReadOnly=*/true,
                                  ELF::SHT_STRTAB);
}

void RewriteInstance::addBoltInfoSection() {
  std::string DescStr;
  raw_string_ostream DescOS(DescStr);

  DescOS << "BOLT revision: " << BoltRevision << ", "
         << "command line:";
  for (int I = 0; I < Argc; ++I)
    DescOS << " " << Argv[I];
  DescOS.flush();

  // Encode as GNU GOLD VERSION so it is easily printable by 'readelf -n'
  const std::string BoltInfo =
      BinarySection::encodeELFNote("GNU", DescStr, 4 /*NT_GNU_GOLD_VERSION*/);
  BC->registerOrUpdateNoteSection(".note.bolt_info", copyByteArray(BoltInfo),
                                  BoltInfo.size(),
                                  /*Alignment=*/1,
                                  /*IsReadOnly=*/true, ELF::SHT_NOTE);
}

void RewriteInstance::addBATSection() {
  BC->registerOrUpdateNoteSection(BoltAddressTranslation::SECTION_NAME, nullptr,
                                  0,
                                  /*Alignment=*/1,
                                  /*IsReadOnly=*/true, ELF::SHT_NOTE);
}

void RewriteInstance::encodeBATSection() {
  std::string DescStr;
  raw_string_ostream DescOS(DescStr);

  BAT->write(*BC, DescOS);
  DescOS.flush();

  const std::string BoltInfo =
      BinarySection::encodeELFNote("BOLT", DescStr, BinarySection::NT_BOLT_BAT);
  BC->registerOrUpdateNoteSection(BoltAddressTranslation::SECTION_NAME,
                                  copyByteArray(BoltInfo), BoltInfo.size(),
                                  /*Alignment=*/1,
                                  /*IsReadOnly=*/true, ELF::SHT_NOTE);
}

template <typename ELFShdrTy>
bool RewriteInstance::shouldStrip(const ELFShdrTy &Section,
                                  StringRef SectionName) {
  // Strip non-allocatable relocation sections.
  if (!(Section.sh_flags & ELF::SHF_ALLOC) && Section.sh_type == ELF::SHT_RELA)
    return true;

  // Strip debug sections if not updating them.
  if (isDebugSection(SectionName) && !opts::UpdateDebugSections)
    return true;

  // Strip symtab section if needed
  if (opts::RemoveSymtab && Section.sh_type == ELF::SHT_SYMTAB)
    return true;

  return false;
}
bool RewriteInstance::shouldStrip(const BinarySection &Section) {
  // Strip non-allocatable relocation sections.
  if (!(Section.getELFFlags() & ELF::SHF_ALLOC) &&
      Section.getELFType() == ELF::SHT_RELA)
    return true;

  // Strip debug sections if not updating them.
  if (isDebugSection(Section.getName()) && !opts::UpdateDebugSections)
    return true;

  // Strip symtab section if needed
  if (opts::RemoveSymtab && Section.getELFType() == ELF::SHT_SYMTAB)
    return true;

  return false;
}

template <typename ELFT>
std::vector<typename object::ELFObjectFile<ELFT>::Elf_Shdr>
RewriteInstance::getOutputSections(ELFObjectFile<ELFT> *File,
                                   std::vector<uint32_t> &NewSectionIndex) {
  using ELFShdrTy = typename ELFObjectFile<ELFT>::Elf_Shdr;
  const ELFFile<ELFT> &Obj = File->getELFFile();
  typename ELFT::ShdrRange InputSections = cantFail(Obj.sections());

  // Keep track of section header entries together with their name.
  std::vector<std::pair<std::string, ELFShdrTy>> OutputSections;
  auto addSection = [&](const std::string &Name, const ELFShdrTy &Section) {
    ELFShdrTy NewSection = Section;
    NewSection.sh_name = SHStrTab.getOffset(Name);
    OutputSections.emplace_back(Name, std::move(NewSection));
  };

  // NULL Section
  OutputSections.emplace_back("", ELFShdrTy{});

  // TODO: Reduce number of for loops
  for (const BinarySection &Section : BC->allocatableSections()) {

    if (Section.isAnonymous() || !Section.isFinalized()) {
      if (opts::Verbosity)
        outs() << "BOLT-INFO: not writing section header for section "
               << Section.getOutputName() << '\n';
      continue;
    }

    if (opts::Verbosity >= 1)
      outs() << "BOLT-INFO: writing section header for " << Section.getName()
             << '\n';
    ELFShdrTy NewSection;
    NewSection.sh_type = Section.getELFType();
    NewSection.sh_addr = Section.getOutputAddress();
    NewSection.sh_offset = Section.getOutputFileOffset();
    NewSection.sh_size = Section.getOutputSize();
    NewSection.sh_entsize = 0;
    NewSection.sh_flags = Section.getELFFlags();
    NewSection.sh_link = 0;
    NewSection.sh_info = 0;
    NewSection.sh_addralign = Section.getAlignment();
    addSection(std::string(Section.getOutputName()), NewSection);
  }

  // Sort all allocatable sections by their offset.
  llvm::stable_sort(OutputSections,
                    [](const std::pair<std::string, ELFShdrTy> &A,
                       const std::pair<std::string, ELFShdrTy> &B) {
                      return A.second.sh_offset < B.second.sh_offset;
                    });

  // Fix section sizes to prevent overlapping.
  ELFShdrTy *PrevSection = nullptr;
  StringRef PrevSectionName;
  for (auto &SectionKV : OutputSections) {
    ELFShdrTy &Section = SectionKV.second;

    // TBSS section does not take file or memory space. Ignore it for layout
    // purposes.
    if (Section.sh_type == ELF::SHT_NOBITS && (Section.sh_flags & ELF::SHF_TLS))
      continue;

    if (PrevSection &&
        PrevSection->sh_addr + PrevSection->sh_size > Section.sh_addr) {
      if (opts::Verbosity > 1)
        outs() << "BOLT-INFO: adjusting size for section " << PrevSectionName
               << '\n';
      PrevSection->sh_size = Section.sh_addr > PrevSection->sh_addr
                                 ? Section.sh_addr - PrevSection->sh_addr
                                 : 0;
    }

    PrevSection = &Section;
    PrevSectionName = SectionKV.first;
  }

  uint64_t LastFileOffset = 0;

  // Copy over entries for non-allocatable sections performing necessary
  // adjustments.
  for (const ELFShdrTy &Section : InputSections) {
    if (Section.sh_type == ELF::SHT_NULL)
      continue;
    if (Section.sh_flags & ELF::SHF_ALLOC)
      continue;

    StringRef SectionName =
        cantFail(Obj.getSectionName(Section), "cannot get section name");

    if (shouldStrip(Section, SectionName))
      continue;

    ErrorOr<BinarySection &> BSec = BC->getUniqueSectionByName(SectionName);
    assert(BSec && "missing section info for non-allocatable section");

    ELFShdrTy NewSection = Section;
    NewSection.sh_offset = BSec->getOutputFileOffset();
    NewSection.sh_size = BSec->getOutputSize();

    if (NewSection.sh_type == ELF::SHT_SYMTAB)
      NewSection.sh_info = NumLocalSymbols;

    addSection(std::string(SectionName), NewSection);

    LastFileOffset = BSec->getOutputFileOffset();
  }

  // Create entries for new non-allocatable sections.
  for (BinarySection &Section : BC->nonAllocatableSections()) {
    if (Section.getOutputFileOffset() <= LastFileOffset)
      continue;

    if (shouldStrip(Section))
      continue;

    if (opts::Verbosity >= 1)
      outs() << "BOLT-INFO: writing section header for " << Section.getName()
             << '\n';

    ELFShdrTy NewSection;
    NewSection.sh_type = Section.getELFType();
    NewSection.sh_addr = 0;
    NewSection.sh_offset = Section.getOutputFileOffset();
    NewSection.sh_size = Section.getOutputSize();
    NewSection.sh_entsize = 0;
    NewSection.sh_flags = Section.getELFFlags();
    NewSection.sh_link = 0;
    NewSection.sh_info = 0;
    NewSection.sh_addralign = Section.getAlignment();

    addSection(std::string(Section.getName()), NewSection);
  }

  // Assign indices to sections.
  std::unordered_map<std::string, uint64_t> NameToIndex;
  for (uint32_t Index = 1; Index < OutputSections.size(); ++Index) {
    const std::string &SectionName = OutputSections[Index].first;
    NameToIndex[SectionName] = Index;
    if (ErrorOr<BinarySection &> Section =
            BC->getUniqueSectionByName(SectionName))
      Section->setIndex(Index);
  }

  // Update section index mapping
  NewSectionIndex.clear();
  NewSectionIndex.resize(InputSections.size(), 0);
  for (const ELFShdrTy &Section : InputSections) {
    if (Section.sh_type == ELF::SHT_NULL)
      continue;

    size_t OldIndex = std::distance(InputSections.begin(), &Section);
    std::string SectionName =
        std::string(cantFail(Obj.getSectionName(Section)));

    // Some sections are stripped
    if (!NameToIndex.count(SectionName))
      continue;
    if (SectionName == ".rodata" || SectionName == ".text") {
      std::string OrgSec = (getOrgSecPrefix() + SectionName).str();
      // if we emitted new .rodata or .text, point the indices to the old ones
      if (NameToIndex.find(OrgSec) != NameToIndex.end()) {
        NewSectionIndex[OldIndex] = NameToIndex[OrgSec];
        continue;
      }
    }
    NewSectionIndex[OldIndex] = NameToIndex[SectionName];
  }
  for (const ELFShdrTy &Section : InputSections) {
    if (Section.sh_type == ELF::SHT_NULL)
      continue;

    std::string SectionName =
        std::string(cantFail(Obj.getSectionName(Section)));

    if (shouldStrip(Section, SectionName))
      continue;
    const uint64_t NewIndex = NameToIndex[SectionName];
    if (Section.sh_link) {
      auto &LinkSection = InputSections[Section.sh_link];
      std::string LinkSectionName =
          std::string(cantFail(Obj.getSectionName(LinkSection)));
      OutputSections[NewIndex].second.sh_link = NameToIndex[LinkSectionName];
    }
    if (Section.sh_info) {
      if (Section.sh_type == ELF::SHT_REL || Section.sh_type == ELF::SHT_RELA) {
        auto &InfoSection = InputSections[Section.sh_info];
        std::string InfoSectionName =
            std::string(cantFail(Obj.getSectionName(InfoSection)));
        OutputSections[NewIndex].second.sh_info = NameToIndex[InfoSectionName];
      } else if (SectionName == ".symtab")
        OutputSections[NewIndex].second.sh_info = NumLocalSymbols;
      else
        OutputSections[NewIndex].second.sh_info = Section.sh_info;
    }
    OutputSections[NewIndex].second.sh_entsize = Section.sh_entsize;
  }
  std::vector<ELFShdrTy> SectionsOnly(OutputSections.size());
  llvm::copy(llvm::make_second_range(OutputSections), SectionsOnly.begin());

  return SectionsOnly;
}

// Rewrite section header table inserting new entries as needed. The sections
// header table size itself may affect the offsets of other sections,
// so we are placing it at the end of the binary.
//
// As we rewrite entries we need to track how many sections were inserted
// as it changes the sh_link value. We map old indices to new ones for
// existing sections.
template <typename ELFT>
void RewriteInstance::patchELFSectionHeaderTable(ELFObjectFile<ELFT> *File) {
  using ELFShdrTy = typename ELFObjectFile<ELFT>::Elf_Shdr;
  using ELFEhdrTy = typename ELFObjectFile<ELFT>::Elf_Ehdr;
  raw_fd_ostream &OS = Out->os();
  const ELFFile<ELFT> &Obj = File->getELFFile();

  std::vector<uint32_t> NewSectionIndex;
  std::vector<ELFShdrTy> OutputSections =
      getOutputSections(File, NewSectionIndex);
  LLVM_DEBUG(
    dbgs() << "BOLT-DEBUG: old to new section index mapping:\n";
    for (uint64_t I = 0; I < NewSectionIndex.size(); ++I)
      dbgs() << "  " << I << " -> " << NewSectionIndex[I] << '\n';
  );

  // Align starting address for section header table. There's no architecutal
  // need to align this, it is just for pleasant human readability.
  uint64_t SHTOffset = OS.tell();
  SHTOffset = appendPadding(OS, SHTOffset, 16);

  // Write all section header entries while patching section references.
  for (ELFShdrTy &Section : OutputSections) {
    OS.write(reinterpret_cast<const char *>(&Section), sizeof(Section));
  }

  // Fix ELF header.
  ELFEhdrTy NewEhdr = Obj.getHeader();

  if (BC->HasRelocations) {
    if (RuntimeLibrary *RtLibrary = BC->getRuntimeLibrary())
      NewEhdr.e_entry = RtLibrary->getRuntimeStartAddress();
    else
      NewEhdr.e_entry = getNewFunctionAddress(NewEhdr.e_entry);
    assert((NewEhdr.e_entry || !Obj.getHeader().e_entry) &&
           "cannot find new address for entry point");
  }
  NewEhdr.e_phoff = PHDRTableOffset;
  NewEhdr.e_phnum = BC->OutputSegments.size();
  NewEhdr.e_shoff = SHTOffset;
  NewEhdr.e_shnum = OutputSections.size();
  NewEhdr.e_shstrndx = NewSectionIndex[NewEhdr.e_shstrndx];
  OS.pwrite(reinterpret_cast<const char *>(&NewEhdr), sizeof(NewEhdr), 0);
}

template <typename ELFT, typename WriteFuncTy, typename StrTabFuncTy>
void RewriteInstance::updateELFSymbolTable(
    ELFObjectFile<ELFT> *File, bool IsDynSym,
    const typename object::ELFObjectFile<ELFT>::Elf_Shdr &SymTabSection,
    const std::vector<uint32_t> &NewSectionIndex, WriteFuncTy Write,
    StrTabFuncTy AddToStrTab) {
  const ELFFile<ELFT> &Obj = File->getELFFile();
  using ELFSymTy = typename ELFObjectFile<ELFT>::Elf_Sym;

  StringRef StringSection =
      cantFail(Obj.getStringTableForSymtab(SymTabSection));

  unsigned NumHotTextSymsUpdated = 0;
  unsigned NumHotDataSymsUpdated = 0;

  std::map<const BinaryFunction *, uint64_t> IslandSizes;
  auto getConstantIslandSize = [&IslandSizes](const BinaryFunction &BF) {
    auto Itr = IslandSizes.find(&BF);
    if (Itr != IslandSizes.end())
      return Itr->second;
    return IslandSizes[&BF] = BF.estimateConstantIslandSize();
  };

  // Symbols for the new symbol table.
  std::vector<ELFSymTy> Symbols;

  auto getNewSectionIndex = [&](uint32_t OldIndex) {
    // For dynamic symbol table, the section index could be wrong on the input,
    // and its value is ignored by the runtime if it's different from
    // SHN_UNDEF and SHN_ABS.
    // However, we still need to update dynamic symbol table, so return a
    // section index, even though the index is broken.
    if (IsDynSym && OldIndex >= NewSectionIndex.size())
      return OldIndex;

    assert(OldIndex < NewSectionIndex.size() && "section index out of bounds");
    const uint32_t NewIndex = NewSectionIndex[OldIndex];

    // We may have stripped the section that dynsym was referencing due to
    // the linker bug. In that case return the old index avoiding marking
    // the symbol as undefined.
    if (IsDynSym && NewIndex != OldIndex && NewIndex == ELF::SHN_UNDEF)
      return OldIndex;
    return NewIndex;
  };

  // Add extra symbols for the function.
  //
  // Note that addExtraSymbols() could be called multiple times for the same
  // function with different FunctionSymbol matching the main function entry
  // point.
  auto addExtraSymbols = [&](const BinaryFunction &Function,
                             const ELFSymTy &FunctionSymbol) {
    if (Function.isFolded()) {
      BinaryFunction *ICFParent = Function.getFoldedIntoFunction();
      while (ICFParent->isFolded())
        ICFParent = ICFParent->getFoldedIntoFunction();
      ELFSymTy ICFSymbol = FunctionSymbol;
      SmallVector<char, 256> Buf;
      ICFSymbol.st_name =
          AddToStrTab(Twine(cantFail(FunctionSymbol.getName(StringSection)))
                          .concat(".icf.0")
                          .toStringRef(Buf));
      ICFSymbol.st_value = ICFParent->getOutputAddress();
      ICFSymbol.st_size = ICFParent->getOutputSize();
      ICFSymbol.st_shndx = ICFParent->getCodeSection()->getIndex();
      Symbols.emplace_back(ICFSymbol);
    }
    if (Function.isSplit()) {
      for (const FunctionFragment &FF :
           Function.getLayout().getSplitFragments()) {
        if (FF.getAddress()) {
          ELFSymTy NewColdSym = FunctionSymbol;
          const SmallString<256> SymbolName = formatv(
              "{0}.cold.{1}", cantFail(FunctionSymbol.getName(StringSection)),
              FF.getFragmentNum().get() - 1);
          NewColdSym.st_name = AddToStrTab(SymbolName);
          NewColdSym.st_shndx =
              Function.getCodeSection(FF.getFragmentNum())->getIndex();
          NewColdSym.st_value = FF.getAddress();
          NewColdSym.st_size = FF.getImageSize();
          NewColdSym.setBindingAndType(ELF::STB_LOCAL, ELF::STT_FUNC);
          Symbols.emplace_back(NewColdSym);
        }
      }
    }
    if (Function.hasConstantIsland()) {
      uint64_t DataMark = Function.getOutputDataAddress();
      uint64_t CISize = getConstantIslandSize(Function);
      uint64_t CodeMark = DataMark + CISize;
      ELFSymTy DataMarkSym = FunctionSymbol;
      DataMarkSym.st_name = AddToStrTab("$d");
      DataMarkSym.st_value = DataMark;
      DataMarkSym.st_size = 0;
      DataMarkSym.setType(ELF::STT_NOTYPE);
      DataMarkSym.setBinding(ELF::STB_LOCAL);
      ELFSymTy CodeMarkSym = DataMarkSym;
      CodeMarkSym.st_name = AddToStrTab("$x");
      CodeMarkSym.st_value = CodeMark;
      Symbols.emplace_back(DataMarkSym);
      Symbols.emplace_back(CodeMarkSym);
    }
    if (Function.hasConstantIsland() && Function.isSplit()) {
      uint64_t DataMark = Function.getOutputColdDataAddress();
      uint64_t CISize = getConstantIslandSize(Function);
      uint64_t CodeMark = DataMark + CISize;
      ELFSymTy DataMarkSym = FunctionSymbol;
      DataMarkSym.st_name = AddToStrTab("$d");
      DataMarkSym.st_value = DataMark;
      DataMarkSym.st_size = 0;
      DataMarkSym.setType(ELF::STT_NOTYPE);
      DataMarkSym.setBinding(ELF::STB_LOCAL);
      ELFSymTy CodeMarkSym = DataMarkSym;
      CodeMarkSym.st_name = AddToStrTab("$x");
      CodeMarkSym.st_value = CodeMark;
      Symbols.emplace_back(DataMarkSym);
      Symbols.emplace_back(CodeMarkSym);
    }
  };

  // For regular (non-dynamic) symbol table, exclude symbols referring
  // to non-allocatable sections.
  auto shouldStrip = [&](const ELFSymTy &Symbol) {
    if (Symbol.isAbsolute() || !Symbol.isDefined())
      return false;

    // If we cannot link the symbol to a section, leave it as is.
    Expected<const typename ELFT::Shdr *> Section =
        Obj.getSection(Symbol.st_shndx);
    if (!Section)
      return false;

    // Remove the section symbol iif the corresponding section was stripped.
    if (Symbol.getType() == ELF::STT_SECTION) {
      if (!getNewSectionIndex(Symbol.st_shndx))
        return true;
      return false;
    }

    // Symbols in non-allocatable sections are typically remnants of relocations
    // emitted under "-emit-relocs" linker option. Delete those as we delete
    // relocations against non-allocatable sections.
    if (!((*Section)->sh_flags & ELF::SHF_ALLOC))
      return true;

    return false;
  };

  for (const ELFSymTy &Symbol : cantFail(Obj.symbols(&SymTabSection))) {
    // For regular (non-dynamic) symbol table strip unneeded symbols.
    if (!IsDynSym && shouldStrip(Symbol))
      continue;

    const BinaryFunction *Function =
        BC->getBinaryFunctionAtAddress(Symbol.st_value);
    // Ignore false function references, e.g. when the section address matches
    // the address of the function.
    if (Function && Symbol.getType() == ELF::STT_SECTION)
      Function = nullptr;

    // For non-dynamic symtab, make sure the symbol section matches that of
    // the function. It can mismatch e.g. if the symbol is a section marker
    // in which case we treat the symbol separately from the function.
    // For dynamic symbol table, the section index could be wrong on the input,
    // and its value is ignored by the runtime if it's different from
    // SHN_UNDEF and SHN_ABS.
    if (!IsDynSym && Function && Symbol.st_shndx &&
        Symbol.st_shndx !=
            Function->getOriginSection()->getSectionRef().getIndex())
      Function = nullptr;

    // Create a new symbol based on the existing symbol.
    ELFSymTy NewSymbol = Symbol;

    if (Function) {
      // If the symbol matched a function that was not emitted, update the
      // corresponding section index but otherwise leave it unchanged.
      if (Function->isPLTFunction()) {
        NewSymbol.st_value = Function->getOutputAddress();
      } else if (Function->isEmitted()) {
        NewSymbol.st_value = Function->getOutputAddress();
        NewSymbol.st_size = Function->getOutputSize();
        NewSymbol.st_shndx = Function->getCodeSection()->getIndex();
      } else if (Symbol.st_shndx < ELF::SHN_LORESERVE) {
        NewSymbol.st_shndx = getNewSectionIndex(Symbol.st_shndx);
      }

      // Add new symbols to the symbol table if necessary.
      if (!IsDynSym)
        addExtraSymbols(*Function, NewSymbol);
    } else {
      // Check if the function symbol matches address inside a function, i.e.
      // it marks a secondary entry point.
      Function =
          (Symbol.getType() == ELF::STT_FUNC)
              ? BC->getBinaryFunctionContainingAddress(Symbol.st_value,
                                                       /*CheckPastEnd=*/false,
                                                       /*UseMaxSize=*/true)
              : nullptr;

      if (Function && Function->isEmitted()) {
        assert(Function->getLayout().isHotColdSplit() &&
               "Adding symbols based on cold fragment when there are more than "
               "2 fragments");
        const uint64_t OutputAddress =
            Function->translateInputToOutputAddress(Symbol.st_value);

        NewSymbol.st_value = OutputAddress;
        // Force secondary entry points to have zero size.
        NewSymbol.st_size = 0;

        // Find fragment containing entrypoint
        FunctionLayout::fragment_const_iterator FF = llvm::find_if(
            Function->getLayout().fragments(), [&](const FunctionFragment &FF) {
              uint64_t Lo = FF.getAddress();
              uint64_t Hi = Lo + FF.getImageSize();
              return Lo <= OutputAddress && OutputAddress < Hi;
            });

        if (FF == Function->getLayout().fragment_end()) {
          assert(
              OutputAddress >= Function->getCodeSection()->getOutputAddress() &&
              OutputAddress < (Function->getCodeSection()->getOutputAddress() +
                               Function->getCodeSection()->getOutputSize()) &&
              "Cannot locate fragment containg secondary entrypoint");
          FF = Function->getLayout().fragment_begin();
        }

        NewSymbol.st_shndx =
            Function->getCodeSection(FF->getFragmentNum())->getIndex();
      } else {
        // Check if the symbol belongs to moved data object and update it.
        BinaryData *BD = nullptr;
        if (!opts::ReorderData.empty() || opts::Rewrite)
          BD = BC->getBinaryDataAtAddress(Symbol.st_value);

        if (BD && BD->isMoved() && !BD->isJumpTable()) {

          BinarySection &OutputSection = BD->getOutputSection();
          assert(OutputSection.getIndex());
          LLVM_DEBUG(dbgs()
                     << "BOLT-DEBUG: moving " << BD->getName() << " from "
                     << *BC->getSectionNameForAddress(Symbol.st_value) << " ("
                     << Symbol.st_shndx << ") to " << OutputSection.getName()
                     << " (" << OutputSection.getIndex() << ")\n");
          NewSymbol.st_shndx = OutputSection.getIndex();
          NewSymbol.st_value = BD->getOutputAddress();
        } else {
          // Otherwise just update the section for the symbol.
          if (Symbol.st_shndx < ELF::SHN_LORESERVE)
            NewSymbol.st_shndx = getNewSectionIndex(Symbol.st_shndx);
        }

        // Detect local syms in the text section that we didn't update
        // and that were preserved by the linker to support relocations against
        // .text. Remove them from the symtab.
        if (Symbol.getType() == ELF::STT_NOTYPE &&
            Symbol.getBinding() == ELF::STB_LOCAL && Symbol.st_size == 0) {
          if (BC->getBinaryFunctionContainingAddress(Symbol.st_value,
                                                     /*CheckPastEnd=*/false,
                                                     /*UseMaxSize=*/true)) {
            // Can only delete the symbol if not patching. Such symbols should
            // not exist in the dynamic symbol table.
            assert(!IsDynSym && "cannot delete symbol");
            continue;
          }
        }
      }
    }

    // Handle special symbols based on their name.
    Expected<StringRef> SymbolName = Symbol.getName(StringSection);
    assert(SymbolName && "cannot get symbol name");

    auto updateSymbolValue = [&](const StringRef Name,
                                 std::optional<uint64_t> Value = std::nullopt) {
      NewSymbol.st_value = Value ? *Value : getNewValueForSymbol(Name);
      NewSymbol.st_shndx = ELF::SHN_ABS;
      outs() << "BOLT-INFO: setting " << Name << " to 0x"
             << Twine::utohexstr(NewSymbol.st_value) << '\n';
    };

    auto SetToEnd = [&](BinarySection *Section) {
      if (Section->getOutputName() == ".bolt.org.text") {
        // move .text end symbol with text itself.
        Section = &BC->getUniqueSectionByName(".text").get();
      }
      NewSymbol.st_value = Section->getOutputEndAddress();
      NewSymbol.st_shndx = Section->getIndex();
    };

    if (opts::HotText &&
        (*SymbolName == "__hot_start" || *SymbolName == "__hot_end")) {
      updateSymbolValue(*SymbolName);
      ++NumHotTextSymsUpdated;
    }

    if (opts::HotData && (*SymbolName == "__hot_data_start" ||
                          *SymbolName == "__hot_data_end")) {
      updateSymbolValue(*SymbolName);
      ++NumHotDataSymsUpdated;
    }

    if (auto It = BC->EndSymbols.find(*SymbolName); It != BC->EndSymbols.end())
      SetToEnd(It->second);

    if (*SymbolName == "_end")
      updateSymbolValue(*SymbolName, NextAvailableAddress);

    if (IsDynSym)
      Write((&Symbol - cantFail(Obj.symbols(&SymTabSection)).begin()) *
                sizeof(ELFSymTy),
            NewSymbol);
    else
      Symbols.emplace_back(NewSymbol);
  }

  if (IsDynSym) {
    assert(Symbols.empty());
    return;
  }

  // Add symbols of injected functions
  for (BinaryFunction *Function : BC->getInjectedBinaryFunctions()) {
    ELFSymTy NewSymbol;
    BinarySection *OriginSection = Function->getOriginSection();
    NewSymbol.st_shndx =
        OriginSection
            ? getNewSectionIndex(OriginSection->getSectionRef().getIndex())
            : Function->getCodeSection()->getIndex();
    NewSymbol.st_value = Function->getOutputAddress();
    NewSymbol.st_name = AddToStrTab(Function->getOneName());
    NewSymbol.st_size = Function->getOutputSize();
    NewSymbol.st_other = 0;
    NewSymbol.setBindingAndType(ELF::STB_LOCAL, ELF::STT_FUNC);
    Symbols.emplace_back(NewSymbol);

    if (Function->isSplit()) {
      assert(Function->getLayout().isHotColdSplit() &&
             "Adding symbols based on cold fragment when there are more than "
             "2 fragments");
      ELFSymTy NewColdSym = NewSymbol;
      NewColdSym.setType(ELF::STT_NOTYPE);
      SmallVector<char, 256> Buf;
      NewColdSym.st_name = AddToStrTab(
          Twine(Function->getPrintName()).concat(".cold.0").toStringRef(Buf));
      const FunctionFragment &ColdFF =
          Function->getLayout().getFragment(FragmentNum::cold());
      NewColdSym.st_value = ColdFF.getAddress();
      NewColdSym.st_size = ColdFF.getImageSize();
      Symbols.emplace_back(NewColdSym);
    }
  }

  assert((!NumHotTextSymsUpdated || NumHotTextSymsUpdated == 2) &&
         "either none or both __hot_start/__hot_end symbols were expected");
  assert((!NumHotDataSymsUpdated || NumHotDataSymsUpdated == 2) &&
         "either none or both __hot_data_start/__hot_data_end symbols were "
         "expected");

  auto addSymbol = [&](const std::string &Name) {
    ELFSymTy Symbol;
    Symbol.st_value = getNewValueForSymbol(Name);
    Symbol.st_shndx = ELF::SHN_ABS;
    Symbol.st_name = AddToStrTab(Name);
    Symbol.st_size = 0;
    Symbol.st_other = 0;
    Symbol.setBindingAndType(ELF::STB_WEAK, ELF::STT_NOTYPE);

    outs() << "BOLT-INFO: setting " << Name << " to 0x"
           << Twine::utohexstr(Symbol.st_value) << '\n';

    Symbols.emplace_back(Symbol);
  };

  if (opts::HotText && !NumHotTextSymsUpdated) {
    addSymbol("__hot_start");
    addSymbol("__hot_end");
  }

  if (opts::HotData && !NumHotDataSymsUpdated) {
    addSymbol("__hot_data_start");
    addSymbol("__hot_data_end");
  }

  // Put local symbols at the beginning.
  llvm::stable_sort(Symbols, [](const ELFSymTy &A, const ELFSymTy &B) {
    if (A.getBinding() == ELF::STB_LOCAL && B.getBinding() != ELF::STB_LOCAL)
      return true;
    return false;
  });

  for (const ELFSymTy &Symbol : Symbols)
    Write(0, Symbol);
}

template <typename ELFT>
void RewriteInstance::patchELFSymTabs(ELFObjectFile<ELFT> *File) {
  const ELFFile<ELFT> &Obj = File->getELFFile();
  using ELFShdrTy = typename ELFObjectFile<ELFT>::Elf_Shdr;
  using ELFSymTy = typename ELFObjectFile<ELFT>::Elf_Sym;

  // Compute a preview of how section indices will change after rewriting, so
  // we can properly update the symbol table based on new section indices.
  std::vector<uint32_t> NewSectionIndex;
  getOutputSections(File, NewSectionIndex);

  // Set pointer at the end of the output file, so we can pwrite old symbol
  // tables if we need to.
  uint64_t NextAvailableOffset =
      getOutputFileOffsetForAddress(NextAvailableAddress);
  assert(NextAvailableOffset >= FirstNonAllocatableOffset &&
         "next available offset calculation failure");
  Out->os().seek(NextAvailableOffset);

  // Update dynamic symbol table.
  const ELFShdrTy *DynSymSection = nullptr;
  for (const ELFShdrTy &Section : cantFail(Obj.sections())) {
    if (Section.sh_type == ELF::SHT_DYNSYM) {
      DynSymSection = &Section;
      break;
    }
  }
  assert((DynSymSection || BC->IsStaticExecutable) &&
         "dynamic symbol table expected");

  if (DynSymSection) {
    auto Dynsym = BC->getUniqueSectionByName(".dynsym");
    assert(Dynsym);
    const uint64_t NewOffset = Dynsym->getOutputFileOffset();
    assert(NewOffset);
    updateELFSymbolTable(
        File,
        /*IsDynSym=*/true, *DynSymSection, NewSectionIndex,
        [&, NewOffset](size_t Offset, const ELFSymTy &Sym) {
          Out->os().pwrite(reinterpret_cast<const char *>(&Sym),
                           sizeof(ELFSymTy), NewOffset + Offset);
        },
        [](StringRef) -> size_t { return 0; });
  }

  if (opts::RemoveSymtab)
    return;

  // (re)create regular symbol table.
  const ELFShdrTy *SymTabSection = nullptr;
  for (const ELFShdrTy &Section : cantFail(Obj.sections())) {
    if (Section.sh_type == ELF::SHT_SYMTAB) {
      SymTabSection = &Section;
      break;
    }
  }
  if (!SymTabSection) {
    errs() << "BOLT-WARNING: no symbol table found\n";
    return;
  }

  const ELFShdrTy *StrTabSection =
      cantFail(Obj.getSection(SymTabSection->sh_link));
  std::string NewContents;
  std::string NewStrTab = std::string(
      File->getData().substr(StrTabSection->sh_offset, StrTabSection->sh_size));
  StringRef SecName = cantFail(Obj.getSectionName(*SymTabSection));
  StringRef StrSecName = cantFail(Obj.getSectionName(*StrTabSection));

  NumLocalSymbols = 0;
  updateELFSymbolTable(
      File,
      /*IsDynSym=*/false,
      *SymTabSection,
      NewSectionIndex,
      [&](size_t Offset, const ELFSymTy &Sym) {
        if (Sym.getBinding() == ELF::STB_LOCAL)
          ++NumLocalSymbols;
        NewContents.append(reinterpret_cast<const char *>(&Sym),
                           sizeof(ELFSymTy));
      },
      [&](StringRef Str) {
        size_t Idx = NewStrTab.size();
        NewStrTab.append(NameResolver::restore(Str).str());
        NewStrTab.append(1, '\0');
        return Idx;
      });

  BC->registerOrUpdateNoteSection(SecName,
                                  copyByteArray(NewContents),
                                  NewContents.size(),
                                  /*Alignment=*/1,
                                  /*IsReadOnly=*/true,
                                  ELF::SHT_SYMTAB);

  BC->registerOrUpdateNoteSection(StrSecName,
                                  copyByteArray(NewStrTab),
                                  NewStrTab.size(),
                                  /*Alignment=*/1,
                                  /*IsReadOnly=*/true,
                                  ELF::SHT_STRTAB);
}

template <typename ELFT>
void RewriteInstance::patchELFAllocatableRelrSection(
    ELFObjectFile<ELFT> *File) {
  if (!DynamicRelrAddress)
    return;

  raw_fd_ostream &OS = Out->os();
  const uint8_t PSize = BC->AsmInfo->getCodePointerSize();
  const uint64_t MaxDelta = ((CHAR_BIT * DynamicRelrEntrySize) - 1) * PSize;

  auto FixAddend = [&](const BinarySection &Section, const Relocation &Rel) {
    uint64_t Addend = 0;

    if (const uint64_t EndSymValue = Section.getNewEndSymbolValue(Rel.Offset))
      Addend = EndSymValue;

    if (!Addend)
      Addend = getNewFunctionOrDataAddress(Rel.Addend);
    // No fixup needed if symbol address was not changed

    if (!Addend)
      return;

    uint64_t FileOffset = Section.getOutputFileOffset();
    if (!FileOffset)
      FileOffset = Section.getInputFileOffset();

    FileOffset += Rel.Offset;
    OS.pwrite(reinterpret_cast<const char *>(&Addend), PSize, FileOffset);
  };

  // Fill new relative relocation offsets set
  std::set<uint64_t> RelOffsets;
  for (const BinarySection &Section : BC->allocatableSections()) {
    const uint64_t SectionInputAddress = Section.getAddress();
    uint64_t SectionAddress = Section.getOutputAddress();
    if (!SectionAddress)
      SectionAddress = SectionInputAddress;

    for (const Relocation &Rel : Section.dynamicRelocations()) {
      if (!Rel.isRelative())
        continue;

      uint64_t RelOffset =
          getNewFunctionOrDataAddress(SectionInputAddress + Rel.Offset);

      RelOffset = RelOffset == 0 ? SectionAddress + Rel.Offset : RelOffset;
      assert((RelOffset & 1) == 0 && "Wrong relocation offset");
      RelOffsets.emplace(RelOffset);
      FixAddend(Section, Rel);
    }
  }

  ErrorOr<BinarySection &> Section =
      BC->getSectionForAddress(*DynamicRelrAddress);
  assert(Section && "cannot get .relr.dyn section");
  assert(Section->isRelr() && "Expected section to be SHT_RELR type");
  uint64_t RelrDynOffset = Section->getOutputFileOffset();
  assert(RelrDynOffset && "No output offset for .relr.dyn");

  const uint64_t RelrDynEndOffset = RelrDynOffset + Section->getSize();

  auto WriteRelr = [&](uint64_t Value) {
    if (RelrDynOffset + DynamicRelrEntrySize > RelrDynEndOffset) {
      errs() << "BOLT-ERROR: Offset overflow for relr.dyn section\n";
      exit(1);
    }

    OS.pwrite(reinterpret_cast<const char *>(&Value), DynamicRelrEntrySize,
              RelrDynOffset);
    RelrDynOffset += DynamicRelrEntrySize;
  };

  for (auto RelIt = RelOffsets.begin(); RelIt != RelOffsets.end();) {
    WriteRelr(*RelIt);
    uint64_t Base = *RelIt++ + PSize;
    while (1) {
      uint64_t Bitmap = 0;
      for (; RelIt != RelOffsets.end(); ++RelIt) {
        const uint64_t Delta = *RelIt - Base;
        if (Delta >= MaxDelta || Delta % PSize)
          break;

        Bitmap |= (1ULL << (Delta / PSize));
      }

      if (!Bitmap)
        break;

      WriteRelr((Bitmap << 1) | 1);
      Base += MaxDelta;
    }
  }

  // Fill the rest of the section with empty bitmap value
  while (RelrDynOffset != RelrDynEndOffset)
    WriteRelr(1);
}

template <typename ELFT>
void
RewriteInstance::patchELFAllocatableRelaSections(ELFObjectFile<ELFT> *File) {
  using Elf_Rela = typename ELFT::Rela;
  raw_fd_ostream &OS = Out->os();
  const ELFFile<ELFT> &EF = File->getELFFile();

  uint64_t RelDynOffset = 0, RelDynEndOffset = 0;
  uint64_t RelPltOffset = 0, RelPltEndOffset = 0;

  auto setSectionFileOffsets = [&](uint64_t Address, uint64_t &Start,
                                   uint64_t &End) {
    ErrorOr<BinarySection &> Section = BC->getSectionForAddress(Address);
    assert(Section && "cannot get relocation section");
    Start = Section->getOutputFileOffset();
    End = Start + Section->getOutputSize();
  };

  if (!DynamicRelocationsAddress && !PLTRelocationsAddress)
    return;

  if (DynamicRelocationsAddress)
    setSectionFileOffsets(*DynamicRelocationsAddress, RelDynOffset,
                          RelDynEndOffset);

  if (PLTRelocationsAddress)
    setSectionFileOffsets(*PLTRelocationsAddress, RelPltOffset,
                          RelPltEndOffset);

  DynamicRelativeRelocationsCount = 0;

  auto writeRela = [&OS](const Elf_Rela *RelA, uint64_t &Offset) {
    OS.pwrite(reinterpret_cast<const char *>(RelA), sizeof(*RelA), Offset);
    Offset += sizeof(*RelA);
  };

  auto writeRelocations = [&](bool PatchRelative) {
    for (BinarySection &Section : BC->allocatableSections()) {
      const uint64_t SectionInputAddress = Section.getAddress();
      uint64_t SectionAddress = Section.getOutputAddress();
      if (!SectionAddress)
        SectionAddress = SectionInputAddress;

      for (const Relocation &Rel : Section.dynamicRelocations()) {
        const bool IsRelative = Rel.isRelative();
        if (PatchRelative != IsRelative)
          continue;

        if (IsRelative)
          ++DynamicRelativeRelocationsCount;

        Elf_Rela NewRelA;
        MCSymbol *Symbol = Rel.Symbol;
        uint32_t SymbolIdx = 0;
        uint64_t Addend = Rel.Addend;
        uint64_t RelOffset =
            getNewFunctionOrDataAddress(SectionInputAddress + Rel.Offset);

        RelOffset = RelOffset == 0 ? SectionAddress + Rel.Offset : RelOffset;
        if (Rel.Symbol) {
          SymbolIdx = getOutputDynamicSymbolIndex(Symbol);
        } else {
          // Usually this case is used for R_*_(I)RELATIVE relocations

          // Check if it's end-of-section relocation first because they're
          // tricky
          uint64_t Address = Section.getNewEndSymbolValue(Rel.Offset);

          if (!Address)
            Address = getNewFunctionOrDataAddress(Rel.Addend);

          if (Address)
            Addend = Address;
        }

        NewRelA.setSymbolAndType(SymbolIdx, Rel.Type, EF.isMips64EL());
        NewRelA.r_offset = RelOffset;
        NewRelA.r_addend = Addend;

        const bool IsJmpRel = IsJmpRelocation.contains(Rel.Type);
        uint64_t &Offset = IsJmpRel ? RelPltOffset : RelDynOffset;
        const uint64_t &EndOffset =
            IsJmpRel ? RelPltEndOffset : RelDynEndOffset;
        if (!Offset || !EndOffset) {
          errs() << "BOLT-ERROR: Invalid offsets for dynamic relocation\n";
          exit(1);
        }

        if (Offset + sizeof(NewRelA) > EndOffset) {
          errs() << "BOLT-ERROR: Offset overflow for dynamic relocation\n";
          exit(1);
        }

        writeRela(&NewRelA, Offset);
      }
    }
  };

  // Place R_*_RELATIVE relocations in RELA section if RELR is not presented.
  // The dynamic linker expects all R_*_RELATIVE relocations in RELA
  // to be emitted first.
  if (!DynamicRelrAddress)
    writeRelocations(/* PatchRelative */ true);
  writeRelocations(/* PatchRelative */ false);

  auto fillNone = [&](uint64_t &Offset, uint64_t EndOffset) {
    if (!Offset)
      return;

    typename ELFObjectFile<ELFT>::Elf_Rela RelA;
    RelA.setSymbolAndType(0, Relocation::getNone(), EF.isMips64EL());
    RelA.r_offset = 0;
    RelA.r_addend = 0;
    while (Offset < EndOffset)
      writeRela(&RelA, Offset);

    assert(Offset == EndOffset && "Unexpected section overflow");
  };

  // Fill the rest of the sections with R_*_NONE relocations
  fillNone(RelDynOffset, RelDynEndOffset);
  fillNone(RelPltOffset, RelPltEndOffset);
}

template <typename ELFT>
void RewriteInstance::patchELFDynamic(ELFObjectFile<ELFT> *File) {
  if (BC->IsStaticExecutable)
    return;

  const ELFFile<ELFT> &Obj = File->getELFFile();
  raw_fd_ostream &OS = Out->os();

  using Elf_Dyn = typename ELFFile<ELFT>::Elf_Dyn;

  // Locate DYNAMIC by looking through program headers.
  const ProgramHeader *DynamicPhdr = nullptr;
  for (const ProgramHeader &Phdr : BC->InputSegments) {
    if (Phdr.p_type == ELF::PT_DYNAMIC) {
      DynamicPhdr = &Phdr;
      assert(Phdr.p_memsz == Phdr.p_filesz && "dynamic sizes should match");
      break;
    }
  }
  assert(DynamicPhdr && "missing dynamic in ELF binary");

  bool ZNowSet = false;

  auto DynamicSection = BC->getUniqueSectionByName(".dynamic");
  assert(DynamicSection);
  uint64_t NewDynamicOffset = DynamicSection->getOutputFileOffset();

  // Go through all dynamic entries and patch functions addresses with
  // new ones.
  typename ELFT::DynRange DynamicEntries =
      cantFail(Obj.dynamicEntries(), "error accessing dynamic table");
  auto DTB = DynamicEntries.begin();
  for (const Elf_Dyn &Dyn : DynamicEntries) {
    Elf_Dyn NewDE = Dyn;
    bool ShouldPatch = true;
    switch (Dyn.d_tag) {
    default:
      ShouldPatch = false;
      break;
    case ELF::DT_RELACOUNT:
      NewDE.d_un.d_val = DynamicRelativeRelocationsCount;
      break;
    case ELF::DT_INIT:
    case ELF::DT_FINI: {
      if (BC->HasRelocations) {
        if (uint64_t NewAddress = getNewFunctionAddress(Dyn.getPtr())) {
          LLVM_DEBUG(dbgs() << "BOLT-DEBUG: patching dynamic entry of type "
                            << Dyn.getTag() << '\n');
          NewDE.d_un.d_ptr = NewAddress;
        }
      }
      RuntimeLibrary *RtLibrary = BC->getRuntimeLibrary();
      if (RtLibrary && Dyn.getTag() == ELF::DT_FINI) {
        if (uint64_t Addr = RtLibrary->getRuntimeFiniAddress())
          NewDE.d_un.d_ptr = Addr;
      }
      if (RtLibrary && Dyn.getTag() == ELF::DT_INIT && !BC->HasInterpHeader) {
        if (auto Addr = RtLibrary->getRuntimeStartAddress()) {
          LLVM_DEBUG(dbgs() << "BOLT-DEBUG: Set DT_INIT to 0x"
                            << Twine::utohexstr(Addr) << '\n');
          NewDE.d_un.d_ptr = Addr;
        }
      }
      break;
    }
    case ELF::DT_FLAGS:
      if (BC->RequiresZNow) {
        NewDE.d_un.d_val |= ELF::DF_BIND_NOW;
        ZNowSet = true;
      }
      break;
    case ELF::DT_FLAGS_1:
      if (BC->RequiresZNow) {
        NewDE.d_un.d_val |= ELF::DF_1_NOW;
        ZNowSet = true;
      }
      break;
    case ELF::DT_INIT_ARRAY:
    case ELF::DT_FINI_ARRAY:
    case ELF::DT_GNU_HASH:
    case ELF::DT_SYMTAB:
    case ELF::DT_STRTAB:
    case ELF::DT_PLTGOT:
    case ELF::DT_RELA:
    case ELF::DT_RELR:
    case ELF::DT_JMPREL:
    case ELF::DT_VERNEED:
    case ELF::DT_VERSYM: {
      auto Section = BC->getSectionForAddress(Dyn.d_un.d_ptr);
      assert(Section && "Cant'find section for dynamic entry");
      assert(Section->getAddress() == NewDE.d_un.d_ptr &&
             "Invalid address for dynamic entry");
      assert(Section->getOutputAddress() &&
             "No output address for section referred in .dynamic");
      NewDE.d_un.d_ptr = Section->getOutputAddress();
      break;
    }
    }
    if (ShouldPatch)
      OS.pwrite(reinterpret_cast<const char *>(&NewDE), sizeof(NewDE),
                NewDynamicOffset + (&Dyn - DTB) * sizeof(Dyn));
  }

  if (BC->RequiresZNow && !ZNowSet) {
    errs() << "BOLT-ERROR: output binary requires immediate relocation "
              "processing which depends on DT_FLAGS or DT_FLAGS_1 presence in "
              ".dynamic. Please re-link the binary with -znow.\n";
    exit(1);
  }
}

template <typename ELFT>
Error RewriteInstance::readELFDynamic(ELFObjectFile<ELFT> *File) {
  const ELFFile<ELFT> &Obj = File->getELFFile();

  using Elf_Dyn = typename ELFFile<ELFT>::Elf_Dyn;

  // Locate DYNAMIC by looking through program headers.
  const ProgramHeader *DynamicPhdr = 0;
  for (const ProgramHeader &Phdr : BC->InputSegments) {
    if (Phdr.p_type == ELF::PT_DYNAMIC) {
      DynamicPhdr = &Phdr;
      break;
    }
  }

  if (!DynamicPhdr) {
    outs() << "BOLT-INFO: static input executable detected\n";
    // TODO: static PIE executable might have dynamic header
    BC->IsStaticExecutable = true;
    return Error::success();
  }

  if (DynamicPhdr->p_memsz != DynamicPhdr->p_filesz)
    return createStringError(errc::executable_format_error,
                             "dynamic section sizes should match");

  // Go through all dynamic entries to locate entries of interest.
  auto DynamicEntriesOrErr = Obj.dynamicEntries();
  if (!DynamicEntriesOrErr)
    return DynamicEntriesOrErr.takeError();
  typename ELFT::DynRange DynamicEntries = DynamicEntriesOrErr.get();

  for (const Elf_Dyn &Dyn : DynamicEntries) {
    switch (Dyn.d_tag) {
    case ELF::DT_INIT:
      if (!BC->HasInterpHeader) {
        LLVM_DEBUG(dbgs() << "BOLT-DEBUG: Set start function address\n");
        BC->StartFunctionAddress = Dyn.getPtr();
      }
      break;
    case ELF::DT_FINI:
      BC->FiniFunctionAddress = Dyn.getPtr();
      break;
    case ELF::DT_RELA:
      DynamicRelocationsAddress = Dyn.getPtr();
      break;
    case ELF::DT_RELASZ:
      DynamicRelocationsSize = Dyn.getVal();
      break;
    case ELF::DT_JMPREL:
      PLTRelocationsAddress = Dyn.getPtr();
      break;
    case ELF::DT_PLTRELSZ:
      PLTRelocationsSize = Dyn.getVal();
      break;
    case ELF::DT_RELACOUNT:
      DynamicRelativeRelocationsCount = Dyn.getVal();
      break;
    case ELF::DT_RELR:
      DynamicRelrAddress = Dyn.getPtr();
      break;
    case ELF::DT_RELRSZ:
      DynamicRelrSize = Dyn.getVal();
      break;
    case ELF::DT_RELRENT:
      DynamicRelrEntrySize = Dyn.getVal();
      break;
    }
  }

  if (!DynamicRelocationsAddress || !DynamicRelocationsSize) {
    DynamicRelocationsAddress.reset();
    DynamicRelocationsSize = 0;
  }

  if (!PLTRelocationsAddress || !PLTRelocationsSize) {
    PLTRelocationsAddress.reset();
    PLTRelocationsSize = 0;
  }

  if (!DynamicRelrAddress || !DynamicRelrSize) {
    DynamicRelrAddress.reset();
    DynamicRelrSize = 0;
  } else if (!DynamicRelrEntrySize) {
    errs() << "BOLT-ERROR: expected DT_RELRENT to be presented "
           << "in DYNAMIC section\n";
    exit(1);
  } else if (DynamicRelrSize % DynamicRelrEntrySize) {
    errs() << "BOLT-ERROR: expected RELR table size to be divisible "
           << "by RELR entry size\n";
    exit(1);
  }

  return Error::success();
}

uint64_t RewriteInstance::getNewFunctionAddress(uint64_t OldAddress) {
  const BinaryFunction *Function = BC->getBinaryFunctionAtAddress(OldAddress);
  if (!Function)
    return 0;

  return Function->getOutputAddress();
}

uint64_t RewriteInstance::getNewFunctionOrDataAddress(uint64_t OldAddress) {
  if (uint64_t Function = getNewFunctionAddress(OldAddress))
    return Function;

  const BinaryData *BD = BC->getBinaryDataAtAddress(OldAddress);
  if (BD && BD->isMoved())
    return BD->getOutputAddress();

  if (auto Section = BC->getSectionForAddress(OldAddress)) {
    assert(Section->getOutputAddress());
    return Section->getOutputAddress() + OldAddress - Section->getAddress();
  }
  return 0;
}

void RewriteInstance::rewriteFile() {
  std::error_code EC;
  Out = std::make_unique<ToolOutputFile>(opts::OutputFilename, EC,
                                         sys::fs::OF_None);
  check_error(EC, "cannot create output executable file");

  raw_fd_ostream &OS = Out->os();

  if (!opts::Rewrite) {
    // Copy allocatable part of the input.
    OS << InputFile->getData().substr(0, FirstNonAllocatableOffset);
  }
  // We obtain an asm-specific writer so that we can emit nops in an
  // architecture-specific way at the end of the function.
  std::unique_ptr<MCAsmBackend> MAB(
      BC->TheTarget->createMCAsmBackend(*BC->STI, *BC->MRI, MCTargetOptions()));
  auto Streamer = BC->createStreamer(OS);
  // Make sure output stream has enough reserved space, otherwise
  // pwrite() will fail.
  uint64_t Offset =
      OS.seek(getOutputFileOffsetForAddress(NextAvailableAddress));
  (void)Offset;
  assert(Offset == getOutputFileOffsetForAddress(NextAvailableAddress) &&
         "error resizing output file");

  // Overwrite functions with fixed output address. This is mostly used by
  // non-relocation mode, with one exception: injected functions are covered
  // here in both modes.
  uint64_t CountOverwrittenFunctions = 0;
  uint64_t OverwrittenScore = 0;
  for (BinaryFunction *Function : BC->getAllBinaryFunctions()) {
    if (Function->getImageAddress() == 0 || Function->getImageSize() == 0)
      continue;

    if (Function->getImageSize() > Function->getMaxSize()) {
      if (opts::Verbosity >= 1)
        errs() << "BOLT-WARNING: new function size (0x"
               << Twine::utohexstr(Function->getImageSize())
               << ") is larger than maximum allowed size (0x"
               << Twine::utohexstr(Function->getMaxSize()) << ") for function "
               << *Function << '\n';

      // Remove jump table sections that this function owns in non-reloc mode
      // because we don't want to write them anymore.
      if (!BC->HasRelocations && opts::JumpTables == JTS_BASIC) {
        for (auto &JTI : Function->JumpTables) {
          JumpTable *JT = JTI.second;
          BinarySection &Section = JT->getOutputSection();
          BC->deregisterSection(Section);
        }
      }
      continue;
    }

    const auto HasAddress = [](const FunctionFragment &FF) {
      return FF.empty() ||
             (FF.getImageAddress() != 0 && FF.getImageSize() != 0);
    };
    const bool SplitFragmentsHaveAddress =
        llvm::all_of(Function->getLayout().getSplitFragments(), HasAddress);
    if (Function->isSplit() && !SplitFragmentsHaveAddress) {
      const auto HasNoAddress = [](const FunctionFragment &FF) {
        return FF.getImageAddress() == 0 && FF.getImageSize() == 0;
      };
      assert(llvm::all_of(Function->getLayout().getSplitFragments(),
                          HasNoAddress) &&
             "Some split fragments have an address while others do not");
      (void)HasNoAddress;
      continue;
    }

    OverwrittenScore += Function->getFunctionScore();
    ++CountOverwrittenFunctions;

    // Overwrite function in the output file.
    if (opts::Verbosity >= 2)
      outs() << "BOLT: rewriting function \"" << *Function << "\"\n";

    OS.pwrite(reinterpret_cast<char *>(Function->getImageAddress()),
              Function->getImageSize(), Function->getFileOffset());

    // Write nops at the end of the function.
    if (Function->getMaxSize() != std::numeric_limits<uint64_t>::max()) {
      uint64_t Pos = OS.tell();
      OS.seek(Function->getFileOffset() + Function->getImageSize());
      BC->MAB->writeNopData(
          OS, Function->getMaxSize() - Function->getImageSize(), &*BC->STI);

      OS.seek(Pos);
    }

    if (!Function->isSplit())
      continue;

    // Write cold part
    if (opts::Verbosity >= 2) {
      outs() << formatv("BOLT: rewriting function \"{0}\" (split parts)\n",
                        *Function);
    }

    for (const FunctionFragment &FF :
         Function->getLayout().getSplitFragments()) {
      OS.pwrite(reinterpret_cast<char *>(FF.getImageAddress()),
                FF.getImageSize(), FF.getFileOffset());
    }
  }

  // Print function statistics for non-relocation mode.
  if (!BC->HasRelocations) {
    outs() << "BOLT: " << CountOverwrittenFunctions << " out of "
           << BC->getBinaryFunctions().size()
           << " functions were overwritten.\n";
    if (BC->TotalScore != 0) {
      double Coverage = OverwrittenScore / (double)BC->TotalScore * 100.0;
      outs() << format("BOLT-INFO: rewritten functions cover %.2lf", Coverage)
             << "% of the execution count of simple functions of "
                "this binary\n";
    }
  }

  if (BC->HasRelocations && opts::TrapOldCode && !opts::Rewrite) {
    uint64_t SavedPos = OS.tell();
    // Overwrite function body to make sure we never execute these instructions.
    for (auto &BFI : BC->getBinaryFunctions()) {
      BinaryFunction &BF = BFI.second;
      if (!BF.getFileOffset() || !BF.isEmitted())
        continue;
      OS.seek(BF.getFileOffset());
      for (unsigned I = 0; I < BF.getMaxSize(); ++I)
        OS.write((unsigned char)BC->MIB->getTrapFillValue());
    }
    OS.seek(SavedPos);
  }

  auto ShouldWrite = [](BinarySection &Section) {
    return Section.isFinalized() && Section.getOutputData() &&
           !Section.isVirtual() &&
           (Section.getData() != Section.getOutputData() ||
            Section.getAddress() != Section.getOutputAddress() ||
            opts::Rewrite);
  };
  // Write all allocatable sections - reloc-mode text is written here as well
  for (BinarySection &Section : BC->allocatableSections()) {
    if (!ShouldWrite(Section))
      continue;

    if (opts::Verbosity >= 1)
      outs() << "BOLT: writing new section " << Section.getName()
             << "\n data at 0x" << Twine::utohexstr(Section.getAllocAddress())
             << "\n of size 0x" << Twine::utohexstr(Section.getOutputSize())
             << "\n at offset 0x"
             << Twine::utohexstr(Section.getOutputFileOffset()) << '\n';
    OS.pwrite(reinterpret_cast<const char *>(Section.getOutputData()),
              Section.getOutputSize(), Section.getOutputFileOffset());
  }

  for (BinarySection &Section : BC->allocatableSections())
    Section.flushPendingRelocations(OS, [this](const MCSymbol *S) {
      return getNewValueForSymbol(S->getName());
    });

  // If .eh_frame is present create .eh_frame_hdr.
  if (EHFrameSection)
    writeEHFrameHeader();

  // Add BOLT Addresses Translation maps to allow profile collection to
  // happen in the output binary
  if (opts::EnableBAT)
    addBATSection();

  // Write program header table.
  writeELFPHDRTable();

  // Finalize memory image of section string table.
  finalizeSectionStringTable();

  // Update symbol tables.
  patchELFSymTabs();

  patchBuildID();

  if (opts::EnableBAT)
    encodeBATSection();

  // Copy non-allocatable sections once allocatable part is finished.
  rewriteNoteSections();

  if (BC->HasRelocations) {
    patchELFAllocatableRelaSections();
    patchELFAllocatableRelrSection();
  }

  // Patch dynamic section/segment.
  patchELFDynamic();

  // Update ELF book-keeping info.
  patchELFSectionHeaderTable();

  if (opts::PrintSections) {
    outs() << "BOLT-INFO: Sections after processing:\n";
    BC->printSections(outs());
  }

  Out->keep();
  EC = sys::fs::setPermissions(opts::OutputFilename, sys::fs::perms::all_all);
  check_error(EC, "cannot set permissions of output file");
}

// this function is needed to estimate .eh_frame_hdr size when assigning
// addresses to leave enough space for entries, but the actual content of
// .eh_frame_hdr can only we written later after linking
void RewriteInstance::mapEhFrameAndHeader(
    BOLTLinker::SectionMapper AssignAddress) {

  auto getEHFrameHeaderEntriesCount = [this](BinarySection *EHFrameSection) {
    DWARFDebugFrame NewEHFrame(BC->TheTriple->getArch(), true,
                               EHFrameSection->getOutputAddress());
    Error E = NewEHFrame.parse(DWARFDataExtractor(
        EHFrameSection->getOutputContents(), BC->AsmInfo->isLittleEndian(),
        BC->AsmInfo->getCodePointerSize()));
    check_error(std::move(E), "failed to parse EH frame");
    const uint64_t NumEntries =
        NewEHFrame.entries().end() - NewEHFrame.entries().begin();
    return NumEntries;
  };

  BinarySection *NewEHFrameSection = getSection(getEHFrameSectionName());
  assert(NewEHFrameSection && NewEHFrameSection->hasValidSectionID());

  mapSection(*NewEHFrameSection);
  AssignAddress(*NewEHFrameSection, NewEHFrameSection->getOutputAddress());
  uint64_t NumEntries = getEHFrameHeaderEntriesCount(NewEHFrameSection);

  if (BinarySection *RelocatedEHFrameSection =
          getSection(".relocated" + getEHFrameSectionName())) {
    NumEntries += getEHFrameHeaderEntriesCount(RelocatedEHFrameSection);
    mapSection(*RelocatedEHFrameSection);
    AssignAddress(*RelocatedEHFrameSection,
                  RelocatedEHFrameSection->getOutputAddress());
  }

  const uint64_t Size = NumEntries * 8 + 12;

  const unsigned Flags = BinarySection::getFlags(/*IsReadOnly=*/true,
                                                 /*IsText=*/false,
                                                 /*IsAllocatable=*/true);
  BinarySection &EHFrameHdr = BC->registerOrUpdateSection(
      getEHFrameHeaderSectionName(), ELF::SHT_PROGBITS, Flags, nullptr, Size,
      /*Alignment=*/1);

  mapSection(EHFrameHdr);
}

void RewriteInstance::writeEHFrameHeader() {
  BinarySection *NewEHFrameSection = getSection(getEHFrameSectionName());

  // No need to update the header if no new .eh_frame was created.
  if (!NewEHFrameSection || !NewEHFrameSection->hasValidSectionID()) {
    return;
  }

  DWARFDebugFrame NewEHFrame(BC->TheTriple->getArch(), true,
                             NewEHFrameSection->getOutputAddress());

  Error E = NewEHFrame.parse(DWARFDataExtractor(
      NewEHFrameSection->getOutputContents(), BC->AsmInfo->isLittleEndian(),
      BC->AsmInfo->getCodePointerSize()));
  check_error(std::move(E), "failed to parse EH frame");
  LLVM_DEBUG(dbgs() << "BOLT: writing a new .eh_frame_hdr\n");

  uint64_t RelocatedEHFrameAddress = 0;
  StringRef RelocatedEHFrameContents;
  BinarySection *RelocatedEHFrameSection =
      getSection(".relocated" + getEHFrameSectionName());
  if (RelocatedEHFrameSection) {
    RelocatedEHFrameAddress = RelocatedEHFrameSection->getOutputAddress();
    RelocatedEHFrameContents = RelocatedEHFrameSection->getOutputContents();
  }
  DWARFDebugFrame RelocatedEHFrame(BC->TheTriple->getArch(), true,
                                   RelocatedEHFrameAddress);
  Error Er = RelocatedEHFrame.parse(DWARFDataExtractor(
      RelocatedEHFrameContents, BC->AsmInfo->isLittleEndian(),
      BC->AsmInfo->getCodePointerSize()));
  check_error(std::move(Er), "failed to parse EH frame");

  auto EHFrameHdr = BC->getUniqueSectionByName(getEHFrameHeaderSectionName());
  assert(EHFrameHdr);
  std::vector<char> NewEHFrameHdr = CFIRdWrt->generateEHFrameHeader(
      RelocatedEHFrame, NewEHFrame, EHFrameHdr->getOutputAddress(),
      FailedAddresses);

  assert(NewEHFrameHdr.size() <= EHFrameHdr->getOutputSize());
  Out->os().pwrite(NewEHFrameHdr.data(), NewEHFrameHdr.size(),
                   EHFrameHdr->getOutputFileOffset());

  // Merge new .eh_frame with the relocated original so that gdb can locate all
  // FDEs.
  if (RelocatedEHFrameSection) {
    const uint64_t NewEHFrameSectionSize =
        RelocatedEHFrameSection->getOutputAddress() +
        RelocatedEHFrameSection->getOutputSize() -
        NewEHFrameSection->getOutputAddress();
    NewEHFrameSection->updateContents(nullptr, NewEHFrameSectionSize);
    BC->deregisterSection(*RelocatedEHFrameSection);
  }

  LLVM_DEBUG(dbgs() << "BOLT-DEBUG: size of new .eh_frame is "
                    << EHFrameSection->getOutputSize() << '\n');
}

uint64_t RewriteInstance::getNewValueForSymbol(const StringRef Name) {
  auto Value = Linker->lookupSymbol(Name);
  if (Value)
    return *Value;

  // Return the original value if we haven't emitted the symbol.
  BinaryData *BD = BC->getBinaryDataByName(Name);
  if (!BD)
    return 0;

  return BD->getAddress();
}

uint64_t
RewriteInstance::getFileOffsetForAddress(const uint64_t Address) const {
  // Find an existing segment that matches the address.
  const auto SegmentInfoI = BC->SegmentMapInfo.upper_bound(Address);
  if (SegmentInfoI == BC->SegmentMapInfo.begin())
    return 0;

  const ProgramHeader &SegmentInfo = std::prev(SegmentInfoI)->second;
  if (Address < SegmentInfo.p_vaddr ||
      Address >= SegmentInfo.p_vaddr + SegmentInfo.p_filesz)
    return 0;

  return SegmentInfo.p_offset + Address - SegmentInfo.p_vaddr;
}

uint64_t
RewriteInstance::getOutputFileOffsetForAddress(const uint64_t Address) const {
  auto LB = BC->OutputAddressToOffsetMap.lower_bound(Address);
  if (LB == BC->OutputAddressToOffsetMap.end() || LB->first > Address) {
    assert(LB != BC->OutputAddressToOffsetMap.begin());
    --LB;
  }
  const uint64_t SegmentStartAddress = LB->first;
  const uint64_t SegmentStartOffset = LB->second;
  const uint64_t OffsetInSegment = Address - SegmentStartAddress;
  return SegmentStartOffset + OffsetInSegment;
}

bool RewriteInstance::willOverwriteSection(StringRef SectionName) {
  if (llvm::is_contained(SectionsToOverwrite, SectionName))
    return true;
  if (llvm::is_contained(DebugSectionsToOverwrite, SectionName))
    return true;

  ErrorOr<BinarySection &> Section = BC->getUniqueSectionByName(SectionName);
  return Section && Section->isAllocatable() && Section->isFinalized();
}

bool RewriteInstance::isDebugSection(StringRef SectionName) {
  if (SectionName.startswith(".debug_") || SectionName.startswith(".zdebug_") ||
      SectionName == ".gdb_index" || SectionName == ".stab" ||
      SectionName == ".stabstr")
    return true;

  return false;
}

bool RewriteInstance::isKSymtabSection(StringRef SectionName) {
  if (SectionName.startswith("__ksymtab"))
    return true;

  return false;
}
