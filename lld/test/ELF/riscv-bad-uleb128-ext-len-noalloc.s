# REQUIRES: riscv
## Relax R_RISCV_SET_ULEB128 and R_RISCV_SUB_ULEB128.

# RUN: rm -rf %t && split-file %s %t && cd %t

## RELAX
# RUN: yaml2obj a.yml -o a.o
# RUN: not ld.lld -T lds a.o -o 32c 2>&1 | FileCheck %s

# CHECK: ULEB128 difference relocation pair overflow: 2 bytes needed but only 1 bytes allocated; references '.Lend_start' - '_start' 

#--- a.yml

--- !ELF
FileHeader:
  Class:           ELFCLASS32
  Data:            ELFDATA2LSB
  Type:            ET_REL
  Machine:         EM_RISCV
  Flags:           [ EF_RISCV_RVC, EF_RISCV_FLOAT_ABI_SOFT ]
  SectionHeaderStringTable: .strtab
Sections:
  - Name:            .text
    Type:            SHT_PROGBITS
    Flags:           [ SHF_ALLOC, SHF_EXECINSTR ]
    AddressAlign:    0x2
    Content:         '01000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010001000100010097000000E78000008280'
  - Name:            .debug_info
    Type:            SHT_PROGBITS
    AddressAlign:    0x1
    Content:         '0000000000'
  - Name:            .rela.text
    Type:            SHT_RELA
    Flags:           [ SHF_INFO_LINK ]
    Link:            .symtab
    AddressAlign:    0x4
    Info:            .text
    Relocations:
      - Offset:          0x1FE
        Symbol:          a
        Type:            R_RISCV_CALL_PLT
      - Offset:          0x1FE
        Type:            R_RISCV_RELAX
  - Name:            .rela.debug_info
    Type:            SHT_RELA
    Flags:           [ SHF_INFO_LINK ]
    Link:            .symtab
    AddressAlign:    0x4
    Info:            .debug_info
    Relocations:
      - Symbol:          .Lend_start
        Type:            R_RISCV_SET_ULEB128
      - Symbol:          _start
        Type:            R_RISCV_SUB_ULEB128
  - Type:            SectionHeaderTable
    Sections:
      - Name:            .strtab
      - Name:            .text
      - Name:            .rela.text
      - Name:            .debug_info
      - Name:            .rela.debug_info
      - Name:            .symtab
Symbols:
  - Name:            '$x.0'
    Section:         .text
  - Name:            a
    Section:         .text
    Value:           0x206
  - Name:            .Lend_start
    Section:         .text
    Value:           0x206
  - Name:            _start
    Section:         .text
    Binding:         STB_GLOBAL
    Size:            0x206
...


#--- lds
SECTIONS {
  .text 0x10000 : { *(.text) }
  .data 0x20000 : { *(.data) }
}
