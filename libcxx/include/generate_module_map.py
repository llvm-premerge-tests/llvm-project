# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

import argparse
import json
import operator
import os
import pathlib
import posixpath
import re
import subprocess
import typing


# The private libc++ detail headers are interdependent such that putting all
# of the e.g. __algorithm/*.h headers in a single module will result in module
# cycles with other libc++ modules. This script figures out how to group the
# private headers in the minimum number of modules per directory using the
# following system.
# 1. Run the preprocessor to get the includes of every libc++ header.
# 2. Build an include tree from the preprocessor output.
# 3. For each private header, collect the list of headers that transitively
#    include it, and it transitively includes.
# 4. Convert the private headers and their transitive includes into modules.
# 5  Merge modules that are in the same directory and don't create a module
#    cycle. Modules can be merged if the includers of each module are not in
#    the includees of the other module. In other words, if they're at the same
#    level of the module tree.
# This is quite similar to a disjoint-set/union-find forest, and could possibly
# be reimplemented as such.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libcxx-include-directory", required=True)
    parser.add_argument("--libcxx-include-target-directory")
    parser.add_argument("--intermediates-directory", required=True)
    parser.add_argument("--cxx-compiler", required=True)
    parser.add_argument("--target")
    parser.add_argument("--isysroot")
    args = parser.parse_args()

    with open(
        os.path.join(args.intermediates_directory, "internal_module_attributes.json")
    ) as module_attributes_file:
        module_attributes = json.load(module_attributes_file)
    validate_exports(
        args.libcxx_include_directory,
        args.libcxx_include_target_directory,
        args.intermediates_directory,
        module_attributes,
    )
    libcxx_includes_path = os.path.join(
        args.intermediates_directory, "libcxx_includes.cpp"
    )
    with open(libcxx_includes_path, "w") as libcxx_includes_file:
        write_libcxx_includes(args.libcxx_include_directory, libcxx_includes_file)
    include_tree = build_include_tree(
        args.cxx_compiler,
        args.target,
        args.isysroot,
        args.libcxx_include_directory,
        args.libcxx_include_target_directory,
        libcxx_includes_path,
        module_attributes,
    )
    private_headers_by_directory = collect_transitive_includes(include_tree)
    modules_by_directory = build_modules(private_headers_by_directory)
    write_module_map(
        modules_by_directory,
        args.libcxx_include_directory,
        args.intermediates_directory,
        module_attributes,
    )
    sanity_check_module_map(args.libcxx_include_directory, include_tree)


def validate_exports(
    libcxx_include_directory: str,
    libcxx_include_target_directory: str,
    intermediates_directory: str,
    module_attributes: typing.Dict[str, dict],
) -> None:
    # Behavior is undefined if a module exports something that none of its headers include.
    def export_is_valid_in_header(export: str, header: str) -> bool:
        def find_include_in_header(
            include_regular_expression: re.Pattern, header_file: typing.TextIO
        ) -> bool:
            for line in header_file:
                if include_regular_expression.search(line):
                    return True
            return False

        # Exports can be either a placeholder for a private header or a real module
        # name. It's assumed that module names can be mapped to header names by
        # stripping std_ and converting a trailing _h to .h.
        include = export
        if include.startswith("std_"):
            include = include[len("std_") :]
            if include.endswith("_h"):
                include = include[: -len("_h")] + ".h"
        include_regular_expression = re.compile(
            r"# *include <" + re.escape(include) + r">"
        )

        try:
            with open(os.path.join(libcxx_include_directory, header)) as header_file:
                return find_include_in_header(include_regular_expression, header_file)
        except FileNotFoundError:
            if libcxx_include_target_directory != libcxx_include_directory:
                with open(
                    os.path.join(libcxx_include_target_directory, header)
                ) as header_file:
                    return find_include_in_header(
                        include_regular_expression, header_file
                    )
            else:
                raise

    invalid_exports = {}

    # Go through the module map lines in reverse order. This makes some simplifying
    # assumptions.
    # 1. `export` comes after all the `header`s in the module.
    # 2. There's only 1 `header` per module, or at least the last listed header in the
    #    module is the one that includes the export.
    with open(os.path.join(intermediates_directory, "module.modulemap")) as module_map:
        module_map_lines = list(reversed(list(module_map)))

    # Exports can be of three forms.
    # 1. A real module like `export std_cstddef`.
    # 2. A placeholder for a private file like `export "__fwd/span.h"`.
    # 3. A wildcard `export *`, which doesn't need to be validated.
    specific_export_regular_expression = re.compile(r'export "?([\w./]+)')
    header_regular_expression = re.compile(r'header "(.*)"')
    for line_index, module_map_line in enumerate(module_map_lines):
        export_match = specific_export_regular_expression.search(module_map_line)
        if export_match:
            found_header_match = False
            for remaining_line in module_map_lines[line_index:]:
                header_match = header_regular_expression.search(remaining_line)
                if header_match:
                    found_header_match = True
                    if not export_is_valid_in_header(export_match[1], header_match[1]):
                        invalid_exports.setdefault(header_match[1], []).append(
                            export_match[1]
                        )
                    break
            assert found_header_match, f"Failed to find header for {export_match[1]}"

    # Go through the module attributes. (Some of the entries are directories, but
    # those don't have exports.)
    for header, header_module_attributes in module_attributes.items():
        for export in header_module_attributes.get("exports", []):
            if (export != "*") and not export_is_valid_in_header(export, header):
                invalid_exports.setdefault(header, []).append(export)

    if invalid_exports:
        for header, exports in invalid_exports.items():
            print("Module for <", header, "> has invalid exports", sep="")
            for export in exports:
                print("    ", export)
            print()
        exit(1)


def write_libcxx_includes(
    libcxx_include_directory: str, libcxx_includes_file: typing.TextIO
) -> None:
    for directory, subdirectorynames, filenames in os.walk(libcxx_include_directory):
        if directory == libcxx_include_directory:
            include_root = ""
            # The ext, and __support headers are not included in the module. Not all of
            # the __pstl headers compile, so exclude that directory too, even though some
            # of the headers will be pulled in transitively and modularized.
            subdirectorynames.remove("ext")
            subdirectorynames.remove("__pstl")
            subdirectorynames.remove("__support")
        else:
            include_root = pathlib.PurePath(
                os.path.relpath(directory, libcxx_include_directory)
            ).as_posix()

        subdirectorynames.sort()
        for filename in sorted(filenames):
            # The __pstl and cxxabi headers are not included in the module.
            if (directory == libcxx_include_directory) and (
                filename.startswith("__pstl_") or ("cxxabi" in filename)
            ):
                continue

            extension = os.path.splitext(filename)[1]
            if (
                (not extension)
                or (extension == ".h")
                or (extension == ".hh")
                or (extension == ".hp")
                or (extension == ".hpp")
                or (extension == ".hxx")
                or (extension == ".h++")
                or (extension == ".ipp")
            ):
                # Only extension-less and .h headers should be present, but allow for any C++ header file.
                libcxx_includes_file.write("#include <")
                libcxx_includes_file.write(posixpath.join(include_root, filename))
                libcxx_includes_file.write(">\n")


def build_include_tree(
    cxx_compiler: str,
    target: typing.Optional[str],
    isysroot: typing.Optional[str],
    libcxx_include_directory: str,
    libcxx_include_target_directory: typing.Optional[str],
    libcxx_includes_path: str,
    module_attributes: typing.Dict[str, dict],
) -> typing.Iterable["Header"]:
    headers = {}  # header objects keyed by path
    # includes vary by standard, so all standards need to be run, and the includes unioned.
    for standard in ["c++98", "c++11", "c++14", "c++17", "c++20", "c++2b"]:
        cxx_command = [cxx_compiler]
        if target:
            cxx_command += ["-target", target]
        if isysroot:
            cxx_command += ["-isysroot", isysroot]
        cxx_command += ["-E", "-H", "-fshow-skipped-includes"]
        cxx_command.append("-std=" + standard)
        cxx_command.append("-D_LIBCPP_ENABLE_EXPERIMENTAL")
        if standard != "c++98":
            cxx_command.append("-D_LIBCPP_DEBUG_RANDOMIZE_UNSPECIFIED_STABILITY")
        cxx_command.append("-nostdinc++")
        cxx_command += ["-cxx-isystem", libcxx_include_directory]
        if libcxx_include_target_directory:
            cxx_command += ["-cxx-isystem", libcxx_include_target_directory]
        cxx_command.append("-w")
        cxx_command.append(libcxx_includes_path)

        # Don't require the command to succeed, some of the headers have #error directives if features
        # are disabled like threads, locales, wide characters, etc. Those headers just don't show any
        # includes which is fine.
        includes = subprocess.run(
            cxx_command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        ).stderr

        # Consider 'a' that includes 'b' and 'c', 'b' that includes 'c', and 'c' that includes
        # nothing. All three headers have header guards. For an input file that includes all three,
        # -H -fshow-skipped-includes will look like this.
        # . /a
        # .. /b
        # ... /c
        # .. /c
        # . /b
        # . /c
        # The last three lines are skipped includes - a's include of c is skipped because c has already
        # been seen via b, and same with the input file's includes of b and c that have already been seen
        # via a and b. Skipped includes do not re-list any of their includes, which is why there isn't a
        # ".. /c" under ". /b". The first time a header is seen, it will show all of its direct includes
        # and potentially some of its transitive includes. Headers without header guards are not skipped,
        # and can have different includes depending on the context.
        header_stack = []  # list of tuples (level, path)
        for line in includes.splitlines():
            if not line.startswith("."):
                continue

            dots, include = line.split(" ", maxsplit=1)
            level = len(dots)
            current_header = headers.get(include)
            if not current_header:
                current_header = Header(
                    include, libcxx_include_directory, module_attributes
                )
                headers[include] = current_header

            while header_stack and (level <= header_stack[-1][0]):
                del header_stack[-1]

            if header_stack:
                parent_header = header_stack[-1][1]
                parent_header.included_headers.add(current_header)
                current_header.including_headers.add(parent_header)

            header_stack.append((level, current_header))
    return headers.values()


def collect_transitive_includes(
    include_tree: typing.Iterable["Header"],
) -> typing.Dict[str, typing.List["Header"]]:
    def collect_transitive_includes_for_header(
        header: "Header",
        include_attribute: str,
        transitive_include_attribute: str,
        should_detect_cycles: bool,
    ) -> typing.Set[typing.Tuple["Header"]]:
        transitive_includes = getattr(header, transitive_include_attribute)

        # includes_to_follow is a list of sets of headers. It starts out as the
        # includes from the header being analyzed. One of the includes is taken
        # from the last set, and its includes are added to the end of the list.
        # Another include is selected from the new last set, and so on until the
        # include path hits the end, or a cycle is detected. After that, the
        # include path removes its last include, and a sibling include from the
        # last set is selected. When there are no more siblings, the include path
        # goes up a level until eventually there are no more includes to process.
        includes = getattr(header, include_attribute)
        includes_to_follow = [includes.copy()] if includes else []
        include_path = [header]

        seen_headers = {header}
        cycles = set()
        while includes_to_follow:
            selected_header = includes_to_follow[-1].pop()
            transitive_includes.add(selected_header)

            include_path.append(selected_header)
            selected_header_includes = getattr(selected_header, include_attribute)

            if should_detect_cycles:
                cycles |= detect_cycles(include_path, selected_header_includes)

            selected_header_includes_to_follow = selected_header_includes - seen_headers
            if selected_header_includes_to_follow:
                includes_to_follow.append(selected_header_includes_to_follow)
            else:
                # Hit the end of the include path, peel back a level.
                del include_path[-1]

                # Peel back the completed levels.
                while includes_to_follow and not includes_to_follow[-1]:
                    del includes_to_follow[-1]
                    del include_path[-1]

            seen_headers.add(selected_header)

        return cycles

    private_headers_by_directory = {}
    cycles = set()
    for header in include_tree:
        if not header.is_private_libcxx_header:
            continue

        # The same cycles should be returned as the graph is traversed up and down. Just collect them
        # from the included headers since that's the more intuitive direction ("a includes b includes c
        # includes a" rather than "c is included by b is included by a is included by c").
        collect_transitive_includes_for_header(
            header,
            "including_headers",
            "transitive_including_headers",
            should_detect_cycles=False,
        )
        cycles |= collect_transitive_includes_for_header(
            header,
            "included_headers",
            "transitive_included_headers",
            should_detect_cycles=True,
        )
        private_headers_by_directory.setdefault(header.module_directory, []).append(
            header
        )

    if cycles:
        print(
            "Include cycles detected across libc++ modules, these must be broken before modules can be made for the libc++ private headers\n"
        )
        for cycle in sorted(cycles):
            for header in cycle:
                print(header.path)
            print()
        exit(1)

    return private_headers_by_directory


def detect_cycles(
    include_path: typing.List["Header"], next_includes: typing.Set["Header"]
) -> typing.Set[typing.Tuple["Header"]]:
    cycles = set()
    for index, include_header in enumerate(include_path):
        if include_header in next_includes:
            # Include cycles are only a problem if they cross module boundaries.
            # If the cycle just consists of non-libc++ headers, then presumably
            # those are all in the same module and it's fine.
            #
            # If only one libc++ header is involved, then it's unclear if the
            # cycle needs to be broken in libc++ or externally. Don't report
            # that cycle either, if the non-libc++ headers are in a
            # [no_undeclared_includes] module, then that will break the cycle.
            # Otherwise, the module cycle should be caught in the modules_include
            # unit test and a strategy can be devised from there.
            #
            # A cycle involving only private headers in the same directory isn't
            # technically a problem if those headers could be grouped into the
            # same module. However, figuring out if the headers can be grouped is
            # tricky at this point, and there aren't currently any such cycles.
            # For now, don't allow that situation.
            libcxx_header_count = 0
            cycle = include_path[index:]
            for cycle_header in cycle:
                if cycle_header.is_libcxx_header:
                    libcxx_header_count += 1

                if libcxx_header_count >= 2:
                    # The include cycle could potentially be entered at any point and
                    # thus get double reported, i.e. as a -> b -> a and b -> a -> b.
                    # Rotate the cycle so that the smallest item is at the end points.
                    # For the above cycle, assume that include_path = cycle = [b, a]
                    # and b is in next_includes. Report (a, b, a) as the cycle by
                    # starting at a and going to the end, then going from the start
                    # back to and including a.
                    starting_point = cycle.index(min(cycle))
                    cycles.add(
                        (*cycle[starting_point:], *cycle[: (starting_point + 1)])
                    )
                    break
    return cycles


def build_modules(
    private_headers_by_directory: typing.Dict[str, typing.List["Header"]]
) -> typing.Dict[str, "Module"]:
    # Build the initial set of modules.
    modules_by_directory, all_modules = group_headers_into_modules(
        private_headers_by_directory
    )

    # The modules essentially constitute a module tree parallel to the include tree*. Enumerate
    # the modules and merge them where it doesn't create a cycle. Module a and b can be merged
    # if none of b's including modules are included by a, and if none of a's including modules
    # are included by b. Multiple passes are required, consider a -> c -> b. On the first pass,
    # a and b can't be merged because that would make a cycle ab -> c -> ab. However a and c
    # can be merged: ac -> b. On the second pass, ac can merge with b. Keep looping until no
    # more merges can be made.
    #
    # * "essentially" because only the private libc++ headers are put into modules, the remaining
    # headers stay as headers and are optimistically treated like modules. The libc++ public headers
    # are each in their own standalone module, except for the experimental headers which are all
    # together in a single module. It's assumed that the experimental module sits strictly on top
    # of all of the other libc++ modules. That is, nothing includes experimental headers except other
    # experimental headers, they'll never show up in any module's included headers, and so they won't
    # effectively matter for grouping. The non-libc++ headers are optimistically assumed to be in
    # individual modules. If that isn't the case, then it's possible that they can cause module cycles.
    # The modules_include unit test will catch those, and if there are any then a strategy can be
    # found to solve the cycle.
    did_merge = True
    while did_merge:
        did_merge, modules_by_directory = merge_modules(
            modules_by_directory, all_modules
        )

    # The module lists in modules_for_directory are sorted so that the headers roughly come out in
    # alphabetical order. If headers are grouped into modules acf, be, d, then the modules will be
    # in that order, except reversed for ease of list enumeration. Reverse the lists so that they're
    # in the right order before returning them. The headers in each module are unsorted, e.g. the
    # first module might actually be afc, sort those so that they're in the right order too.
    for directory, modules_for_directory in modules_by_directory.items():
        for module in modules_for_directory:
            module.headers.sort(key=operator.attrgetter("submodule_name"))
        modules_for_directory.reverse()
    return modules_by_directory


def group_headers_into_modules(
    private_headers_by_directory: typing.Dict[str, typing.List["Header"]]
) -> typing.Tuple[typing.Dict[str, "Module"], typing.List["Module"]]:
    # Headers are safe to group if their including and included headers match, modulo each other.
    # In other words, if a and b's transitive includes match except that a includes b and b is
    # included by a, then they can be grouped. It's tempting to try to more aggressively group
    # headers that look like they're at the same level in the include tree, however that can
    # create module cycles. e.g. a1 -> b1, b2 -> a2. None of a2's including headers are included
    # by a1 and vice versa, and same with b2 and b1. However, if the a's and b's are both grouped
    # into modules, then a1a2 -> b1b2 -> a1a2 and it's a module cycle. Start simple with the
    # initial modules and only do the failsafe grouping.
    def has_differing_transitive_includes(
        module: "Module", header: "Header", include_attribute: str
    ) -> bool:
        differing_transitive_includes = getattr(module, include_attribute) ^ getattr(
            header, include_attribute
        )
        differing_transitive_includes.discard(header)
        differing_transitive_includes.difference_update(module.headers)
        return bool(differing_transitive_includes)

    modules = []
    modules_by_directory = {}
    modules_by_header = {}
    for directory, headers in private_headers_by_directory.items():
        # Visit the headers in alphabetical order. Reverse the headers and then enumerate
        # them in reverse so that the consumed headers can be removed while enumerating.
        sorted_headers = sorted(
            headers, key=operator.attrgetter("submodule_name"), reverse=True
        )
        modules_for_directory = modules_by_directory.setdefault(directory, [])
        while sorted_headers:
            header = sorted_headers.pop()
            module = Module(header)
            modules_by_header[header] = module
            index = -1
            for candidate_header in reversed(sorted_headers):
                if (
                    not has_differing_transitive_includes(
                        module, candidate_header, "transitive_including_headers"
                    )
                ) and (
                    not has_differing_transitive_includes(
                        module, candidate_header, "transitive_included_headers"
                    )
                ):
                    module.headers.append(candidate_header)
                    module.transitive_including_headers.discard(candidate_header)
                    module.transitive_included_headers.discard(candidate_header)

                    modules_by_header[candidate_header] = module
                    del sorted_headers[index]
                else:
                    index -= 1
            modules.append(module)
            # Set up the modules for directory in reverse order so that they can
            # be enumerated and trimmed in reverse order below in similar fashion
            # to the headers here.
            modules_for_directory.insert(0, module)

    # Replace the private headers in the modules' including/included headers with their new modules.
    def replace_headers_with_modules(
        module: "Module", headers_attribute: str, modules_attribute: str
    ) -> None:
        transitive_headers = getattr(module, headers_attribute)
        transitive_modules = getattr(module, modules_attribute)
        headers_with_modules = set()
        for header in transitive_headers:
            module_for_header = modules_by_header.get(header)
            if module_for_header:
                headers_with_modules.add(header)
                transitive_modules.add(module_for_header)
        transitive_headers -= headers_with_modules

    for module in modules:
        replace_headers_with_modules(
            module, "transitive_including_headers", "transitive_including_modules"
        )
        replace_headers_with_modules(
            module, "transitive_included_headers", "transitive_included_modules"
        )

    return modules_by_directory, modules


def merge_modules(
    modules_by_directory: typing.Dict[str, "Module"], all_modules: typing.List["Module"]
) -> typing.Tuple[bool, typing.Dict[str, "Module"]]:
    def merge_transitive_module_includes(
        surviving_module: "Module",
        module_being_merged: "Module",
        transitive_modules_attribute: str,
    ) -> None:
        transitive_modules = getattr(surviving_module, transitive_modules_attribute)
        transitive_modules |= getattr(module_being_merged, transitive_modules_attribute)
        # If the modules being merged included each other, remove those references.
        transitive_modules.discard(surviving_module)
        transitive_modules.discard(module_being_merged)

    def update_transitive_module_includes(
        module: "Module",
        surviving_module: "Module",
        module_being_merged: "Module",
        transitive_modules_attribute: str,
    ) -> None:
        transitive_modules = getattr(module, transitive_modules_attribute)
        if (surviving_module in transitive_modules) and (
            module_being_merged not in transitive_modules
        ):
            transitive_modules |= getattr(
                module_being_merged, transitive_modules_attribute
            )
        if module_being_merged in transitive_modules:
            if surviving_module not in transitive_modules:
                transitive_modules.add(surviving_module)
                transitive_modules |= getattr(
                    surviving_module, transitive_modules_attribute
                )
            transitive_modules.remove(module_being_merged)

    # Use the same double-reversed strategy as was used to enumerate the headers.
    did_merge = False
    merged_modules_by_directory = {}
    for directory, modules_for_directory in modules_by_directory.items():
        merged_modules_for_directory = merged_modules_by_directory.setdefault(
            directory, []
        )
        while modules_for_directory:
            module = modules_for_directory.pop()
            index = -1
            for candidate_module in reversed(modules_for_directory):
                # Modules can be merged as long as it won't create a module cycle. Module a and b can be merged
                # if none of b's including modules are included by a, and if none of a's including modules are
                # included by b. The headers that don't have known modules are treated as if they're in their
                # own standalone module, see the comment in build_modules.
                if (
                    candidate_module.transitive_including_headers.isdisjoint(
                        module.transitive_included_headers
                    )
                    and module.transitive_including_headers.isdisjoint(
                        candidate_module.transitive_included_headers
                    )
                    and candidate_module.transitive_including_modules.isdisjoint(
                        module.transitive_included_modules
                    )
                    and module.transitive_including_modules.isdisjoint(
                        candidate_module.transitive_included_modules
                    )
                ):
                    # Update the rest of the modules with the merge. This needs to happen after every
                    # merge so that subsequent merge attempts can correctly identify cycles. For
                    # a1 -> b1, b2 -> a2, a1 and a2 will be merged into a1a2. b1 and b2 need to be
                    # updated to a1a2 -> b1, b2 -> a1a2 so that they don't get merged and create a cycle.
                    # Another scenario is a -> b, c -> d, e -> a. If d and e merge, c needs to not only
                    # replace d with de, but it also has to add all of e's transitive includes, i.e. a,
                    # so that c doesn't later merge with b and cause a cycle a -> bc -> de -> a.
                    all_modules.remove(candidate_module)
                    for module_to_update in all_modules:
                        update_transitive_module_includes(
                            module_to_update,
                            module,
                            candidate_module,
                            "transitive_including_modules",
                        )
                        update_transitive_module_includes(
                            module_to_update,
                            module,
                            candidate_module,
                            "transitive_included_modules",
                        )

                    # Merge the candidate module.
                    module.headers += candidate_module.headers
                    module.transitive_including_headers |= (
                        candidate_module.transitive_including_headers
                    )
                    module.transitive_included_headers |= (
                        candidate_module.transitive_included_headers
                    )
                    merge_transitive_module_includes(
                        module, candidate_module, "transitive_including_modules"
                    )
                    merge_transitive_module_includes(
                        module, candidate_module, "transitive_included_modules"
                    )

                    did_merge = True
                    del modules_for_directory[index]
                else:
                    index -= 1
            merged_modules_for_directory.insert(0, module)

    return did_merge, merged_modules_by_directory


def write_module_map(
    modules_by_directory: typing.Dict[str, typing.List["Module"]],
    libcxx_include_directory: str,
    intermediates_directory: str,
    module_attributes: typing.Dict[str, dict],
) -> None:
    # Figure out all of the module names to resolve the exports from the module attributes.
    sorted_directories_and_modules = sorted(
        modules_by_directory.items(), key=operator.itemgetter(0)
    )
    libcxx_include_directory_as_posix = pathlib.PurePath(
        libcxx_include_directory
    ).as_posix()
    module_names_by_relative_include_path = {}
    for directory, modules in sorted_directories_and_modules:
        module_name_base = "std_private"
        if directory != libcxx_include_directory:
            directory_name = posixpath.relpath(
                directory, libcxx_include_directory_as_posix
            )
            # Some directories are suffixed with '_dir' to not collide with header names. e.g. <locale>'s
            # private headers are in __locale_dir because there's already a __locale file. Drop the '_dir'
            # from the module name.
            if directory_name.endswith("_dir"):
                directory_name = directory_name[: -len("_dir")]
            module_name_base += "_" + directory_name.lstrip("_").replace(
                posixpath.sep, "_"
            )

        module_number = 1
        use_module_number = len(modules) > 1

        for module in modules:
            module_name = module_name_base
            if use_module_number:
                module_name += "_" + str(module_number)
            module.name = module_name

            for header in module.headers:
                module_names_by_relative_include_path[
                    header.libcxx_include_relative_path
                ] = (module_name + "." + header.submodule_name)

            module_number += 1

    with open(
        os.path.join(libcxx_include_directory, "module.modulemap"), "w"
    ) as destination_module_map:
        # Append the dynamically generated private modules to the statically defined public modules in the
        # intermediates directory. Public modules in the static module map export private modules using the
        # header name in quotes as a placeholder. Replace that with the actual private module name.
        export_regular_expression = re.compile(r'export ("(.*)")')
        with open(
            os.path.join(intermediates_directory, "module.modulemap")
        ) as source_module_map:
            for line in source_module_map:
                line_to_write = line
                match = export_regular_expression.search(line)
                if match:
                    line_to_write = line[: match.start(1)]
                    line_to_write += module_names_by_relative_include_path[match[2]]
                    line_to_write += line[match.end(1) :]
                destination_module_map.write(line_to_write)

        for directory, modules in sorted_directories_and_modules:
            directory_module_attributes = module_attributes.get(
                posixpath.relpath(directory, libcxx_include_directory_as_posix), {}
            )
            for module in modules:
                destination_module_map.write("module ")
                destination_module_map.write(module.name)
                destination_module_map.write(" [system] {")

                wrote_directory_requires = False
                for requires in directory_module_attributes.get("requires", []):
                    if requires:
                        destination_module_map.write("\n  ")
                        destination_module_map.write(requires)
                        wrote_directory_requires = True
                if wrote_directory_requires:
                    destination_module_map.write("\n")

                module_submodule_name_width = 0
                for header in module.headers:
                    submodule_name_length = len(header.submodule_name)
                    if submodule_name_length > module_submodule_name_width:
                        module_submodule_name_width = submodule_name_length

                for header in module.headers:
                    header_module_attributes = module_attributes.get(
                        header.libcxx_include_relative_path, {}
                    )
                    needs_multiline = "exports" in header_module_attributes
                    if not needs_multiline:
                        # configure_file might have configured all of the requires to "".
                        for requires in header_module_attributes.get("requires", []):
                            if requires:
                                needs_multiline = True
                                break

                    destination_module_map.write("\n  module ")
                    destination_module_map.write(
                        header.submodule_name.ljust(module_submodule_name_width)
                    )

                    if needs_multiline:
                        destination_module_map.write(" {")
                        for requires in header_module_attributes.get("requires", []):
                            if requires:
                                destination_module_map.write("\n    ")
                                destination_module_map.write(requires)
                        destination_module_map.write("\n    ")
                    else:
                        destination_module_map.write(" { ")

                    if header_module_attributes.get("textual"):
                        destination_module_map.write("textual ")
                    destination_module_map.write('header "')
                    destination_module_map.write(header.libcxx_include_relative_path)

                    if needs_multiline:
                        destination_module_map.write('"')
                        for export in header_module_attributes.get("exports", []):
                            # export can be a relative header path in the case of the
                            # private headers, a module name in the case of the statically
                            # declared modules, or "*".
                            exported_module = module_names_by_relative_include_path.get(
                                export, export
                            )
                            destination_module_map.write("\n    export ")
                            destination_module_map.write(exported_module)
                        destination_module_map.write("\n  }")
                    else:
                        destination_module_map.write('" }')
                destination_module_map.write("\n}\n")


def sanity_check_module_map(
    libcxx_include_directory: str, include_tree: typing.Iterable["Header"]
):
    with open(
        os.path.join(libcxx_include_directory, "module.modulemap")
    ) as module_map_file:
        module_map_contents = module_map_file.read()
    missing_public_headers = []
    missing_private_headers = []
    public_headers_in_multiple_modules = []
    private_headers_in_multiple_modules = []
    for header in include_tree:
        if not header.is_libcxx_header:
            continue

        # Super basic sanity check: make sure that every header selected by write_libcxx_includes
        # ends up in exactly one module in the module map.
        header_statement = 'header "' + header.libcxx_include_relative_path + '"'
        first_header_match = module_map_contents.find(header_statement)
        if first_header_match == -1:
            if header.is_private_libcxx_header:
                missing_private_headers.append(header)
            else:
                missing_public_headers.append(header)
        elif module_map_contents.find(header_statement, first_header_match + 1) != -1:
            if header.is_private_libcxx_header:
                private_headers_in_multiple_modules.append(header)
            else:
                public_headers_in_multiple_modules.append(header)

    if missing_public_headers:
        print("module.modulemap.in is missing public headers:")
        for header in missing_public_headers:
            print("  ", header.libcxx_include_relative_path)
        print()
    if missing_private_headers:
        print("generate_module_map.py failed to generate modules for private headers:")
        for header in missing_private_headers:
            print("  ", header.libcxx_include_relative_path)
        print()
    if public_headers_in_multiple_modules:
        print("public headers in module.modulemap.in are in multiple modules:")
        for header in public_headers_in_multiple_modules:
            print("  ", header.libcxx_include_relative_path)
        print()
    if private_headers_in_multiple_modules:
        print(
            "private headers generated by generate_module_map.py are in multiple modules:"
        )
        for header in private_headers_in_multiple_modules:
            print("  ", header.libcxx_include_relative_path)
        print()
    if (
        missing_public_headers
        or missing_private_headers
        or public_headers_in_multiple_modules
        or private_headers_in_multiple_modules
    ):
        exit(1)


class Header:
    def __init__(
        self,
        path: str,
        libcxx_include_directory: str,
        module_attributes: typing.Dict[str, dict],
    ) -> None:
        self.path = path
        try:
            self.libcxx_include_relative_path = pathlib.PurePath(
                os.path.relpath(path, libcxx_include_directory)
            ).as_posix()
        except ValueError:
            self.libcxx_include_relative_path = ".."
        if posixpath.commonpath([self.libcxx_include_relative_path, ".."]):
            self.is_libcxx_header = False
            self.libcxx_include_relative_path = ""
            self.is_private_libcxx_header = False
            self.module_directory = pathlib.PurePath(os.path.dirname(path)).as_posix()
            self.submodule_name = ""
        else:
            self.is_libcxx_header = True
            if self.libcxx_include_relative_path.startswith("__"):
                self.is_private_libcxx_header = True

                header_module_attributes = module_attributes.get(
                    self.libcxx_include_relative_path, {}
                )

                module_directory = header_module_attributes.get("module_directory", "")
                if not module_directory:
                    module_directory_parent = posixpath.dirname(
                        self.libcxx_include_relative_path
                    )
                    while module_directory_parent:
                        module_directory = module_directory_parent
                        module_directory_parent = posixpath.dirname(module_directory)
                if module_directory:
                    self.module_directory = posixpath.join(
                        pathlib.PurePath(libcxx_include_directory).as_posix(),
                        module_directory,
                    )
                else:
                    self.module_directory = pathlib.PurePath(
                        libcxx_include_directory
                    ).as_posix()

                submodule_name = header_module_attributes.get("submodule_name")
                if not submodule_name:
                    path_as_posix = pathlib.PurePath(path).as_posix()
                    if (
                        posixpath.commonpath([path_as_posix, self.module_directory])
                        == self.module_directory
                    ):
                        submodule_name = posixpath.relpath(
                            path_as_posix, self.module_directory
                        ).replace(posixpath.sep, "_")
                    else:
                        submodule_name = os.path.basename(path)
                    if submodule_name.endswith(".h"):
                        submodule_name = submodule_name[: -len(".h")]
                self.submodule_name = submodule_name
            else:
                self.is_private_libcxx_header = False
                self.module_directory = pathlib.PurePath(
                    os.path.dirname(path)
                ).as_posix()
                self.submodule_name = ""

        self.including_headers = set()
        self.included_headers = set()
        self.transitive_including_headers = set()
        self.transitive_included_headers = set()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.path}>"

    def __hash__(self) -> int:
        return hash(self.path)

    def __eq__(self, other) -> bool:
        return self.path == other.path

    def __ne__(self, other) -> bool:
        return self.path != other.path

    def __lt__(self, other) -> bool:
        return self.path < other.path

    def __le__(self, other) -> bool:
        return self.path <= other.path

    def __gt__(self, other) -> bool:
        return self.path > other.path

    def __ge__(self, other) -> bool:
        return self.path >= other.path


class Module:
    def __init__(self, header: Header) -> None:
        self.name = ""
        self.headers = [header]
        self.transitive_including_headers = header.transitive_including_headers.copy()
        self.transitive_included_headers = header.transitive_included_headers.copy()
        self.transitive_including_modules = set()
        self.transitive_included_modules = set()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} headers = {self.headers}>"


main()
