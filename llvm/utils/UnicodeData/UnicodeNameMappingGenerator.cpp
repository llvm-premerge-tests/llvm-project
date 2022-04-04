#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include <algorithm>
#include <array>
#include <deque>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// List of generated names
// Should be kept in sync with Unicode
// "Name Derivation Rule Prefix String"
static bool generated(char32_t c) {
  const std::pair<char32_t, char32_t> ranges[] = {
      {0xAC00, 0xD7A3},   {0x3400, 0x4DBF},   {0x4E00, 0x9FFC},
      {0x20000, 0x2A6DD}, {0x2A700, 0x2B734}, {0x2B740, 0x2B81D},
      {0x2B820, 0x2CEA1}, {0x2CEB0, 0x2EBE0}, {0x17000, 0x187F7},
      {0x18D00, 0x18D08}, {0x18B00, 0x18CD5}, {0x1B170, 0x1B2FB},
      {0x0F900, 0xFA6D},  {0x0FA70, 0xFAD9},  {0x2F800, 0x2FA1D}};
  for (const auto &r : ranges) {
    if (c >= r.first && c <= r.second)
      return true;
  }
  return false;
}

const std::string LETTERS = "\t ABCDEFGHIJKLMNOPQRSTUVWXYZ_-0123456789";

// Collect names UnicodeData.txt and AliasNames.txt
// There may be multiple names per code points
static std::unordered_multimap<char32_t, std::string>
load_data(const std::string &NamesFile, const std::string &AliasesFile) {
  std::unordered_multimap<char32_t, std::string> CollectedCharacters;
  auto from_file = [&](const std::string &file, bool isAliasFile = false) {
    std::ifstream InputFile(file);
    for (std::string Line; getline(InputFile, Line);) {
      if (Line.empty() || !isxdigit(Line[0]))
        continue;
      auto FirstSemiPos = Line.find(';');
      if (FirstSemiPos == std::string::npos)
        continue;
      auto SecondSemiPos = Line.find(';', FirstSemiPos + 1);
      if (FirstSemiPos == std::string::npos)
        continue;
      unsigned long long CodePoint;
      if (llvm::getAsUnsignedInteger(
              llvm::StringRef(Line.c_str(), FirstSemiPos), 16, CodePoint)) {
        continue;
      }
      // Ignore characters whose name is generated from codepoints
      if (generated(CodePoint))
        continue;

      auto name =
          Line.substr(FirstSemiPos + 1, SecondSemiPos - FirstSemiPos - 1);

      // Some aliases are ignored for compatibility with C++
      if (isAliasFile) {
        auto kind = Line.substr(SecondSemiPos + 1);
        if (kind != "control" && kind != "correction" && kind != "alternate")
          continue;
      }

      auto InserUnique = [&](char32_t CP, std::string name) {
        auto it = CollectedCharacters.find(CP);
        while (it != std::end(CollectedCharacters) && it->first == CP) {
          if (it->second == name)
            return;
          ++it;
        }
        CollectedCharacters.insert({CP, name});
      };
      InserUnique(CodePoint, name);
    }
  };

  from_file(NamesFile);
  from_file(AliasesFile, true);
  return CollectedCharacters;
}

class trie {
  struct node;

public:
  // When inserting named codepoint
  // We create a node per character in the name.
  // SPARKLE becomes S <- P <- A <- R <- K <- L <- E
  // Once all  characters are inserted, the tree is compacted
  void insert(llvm::StringRef name, char32_t v) {
    node *n = root.get();
    for (auto ch : name) {
      std::string label(1, ch);
      auto it = std::find_if(n->children.begin(), n->children.end(),
                             [&](const auto &c) { return c->name == label; });
      if (it == n->children.end()) {
        it = n->children.insert(it, std::make_unique<node>(label, n));
      }
      n = it->get();
    }
    n->value = v;
  }
  void compact() { compact(root.get()); }

  // This creates 2 arrays of bytes from the tree:
  // A serialized dictionary of node labels,
  // And the nodes themselves.
  // The name of each label is found by indexing into the dictionary.
  // The longest names are inserted first into the dictionary,
  // in the hope it will contain shorter labels as substring,
  // thereby reducing duplication.
  // We could theorically be more clever by trying to minimizing the size
  // of the dictionary.
  std::pair<std::string, std::vector<uint8_t>> Serialize() {
    std::set<std::string> names = this->getNameFragments();
    std::vector<std::string> sorted(names.begin(), names.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto &a, const auto &b) { return a.size() > b.size(); });
    std::string dict = LETTERS;
    dict.reserve(50000);
    for (const auto &n : sorted) {
      if (n.size() <= 1)
        continue;
      if (dict.find(n) != std::string::npos)
        continue;
      dict += n;
    }

    if (dict.size() >= std::numeric_limits<uint16_t>::max()) {
      fprintf(stderr, "Dictionary too big  to be serialized");
      exit(1);
    }

    auto bytes = DumpIndex(dict);
    return {dict, bytes};
  }

  std::set<std::string> getNameFragments() {
    std::set<std::string> set;
    collect_keys(root.get(), set);
    return set;
  }

  // Maps a valid char in an Unicode character name
  // To a 6 bits index.
  static uint8_t letter(char c) {
    auto pos = LETTERS.find(c);
    assert(pos != std::string::npos &&
           "Invalid letter in Unicode character name");
    return pos;
  }

  // clang-format off
  // +================+============+======================+=============+========+===+==============+===============+
  // | 0          | 1             | 2-7 (6)              | 8-23        | 24-44  |    | 46           | 47            |
  // +================+============+======================+=============+========+===+==============+===============+
  // | Has Value |  Has Long Name | Letter OR Name Size  | Dict Index  | Value  |    | Has Sibling  | Has Children  |
  // +----------------+------------+----------------------+-------------+--------+---+--------------+---------------+
  // clang-format on

  std::vector<uint8_t> DumpIndex(const std::string &dict) {
    struct ChildrenOffset {
      node *firstChild;
      std::size_t offset;
      bool hasValue;
    };

    // Keep track of the start of each node
    // position in the serialized data.
    std::unordered_map<node *, int32_t> offsets;

    // Keep track of where to write the index
    // of the first children
    std::vector<ChildrenOffset> children_offsets;
    std::unordered_map<node *, bool> sibling_nodes;
    std::deque<node *> nodes;
    std::vector<uint8_t> bytes;
    bytes.reserve(250'000);

    auto add_children = [&sibling_nodes, &nodes](const auto &container) {
      for (std::size_t idx = 0; idx < container.size(); idx++) {
        const std::unique_ptr<node> &c = container[idx];
        nodes.push_back(c.get());
        if (idx != container.size() - 1)
          sibling_nodes[c.get()] = true;
      }
    };
    add_children(root->children);

    while (!nodes.empty()) {
      const std::size_t offset = bytes.size();
      node *const n = nodes.front();
      nodes.pop_front();

      assert(n->name.size() != 0);
      offsets[n] = offset;

      uint8_t b = (!!n->value) ? 0x80 : 0;
      // Single letter node are indexed in 6 bits
      if (n->name.size() == 1) {
        b |= letter(n->name[0]);
        bytes.push_back(b);
      } else {
        // Otherwise we use a 16 bits index
        b = b | uint8_t(n->name.size()) | 0x40;
        bytes.push_back(b);
        auto pos = dict.find(n->name);
        assert(pos != std::string::npos);
        uint8_t l = pos;
        uint8_t h = ((pos >> 8) & 0xFF);
        bytes.push_back(h);
        bytes.push_back(l);
      }

      const bool has_sibling = sibling_nodes.count(n) != 0;
      const bool has_children = n->children.size() != 0;

      if (!!n->value) {
        uint32_t v = (*(n->value) << 3);
        uint8_t h = ((v >> 16) & 0xFF);
        uint8_t m = ((v >> 8) & 0xFF);
        uint8_t l = (v & 0xFF) | uint8_t(has_sibling ? 0x01 : 0) |
                    uint8_t(has_children ? 0x02 : 0);

        bytes.push_back(h);
        bytes.push_back(m);
        bytes.push_back(l);

        if (has_children) {
          children_offsets.push_back(
              ChildrenOffset{n->children[0].get(), bytes.size(), true});
          // index of the first children
          bytes.push_back(0x00);
          bytes.push_back(0x00);
          bytes.push_back(0x00);
        }
      } else {
        // When there is no value (that's most intermediate nodes)
        // Dispense of the 3 values bytes, and only store
        // 1 byte to track whether the node has sibling and chidren
        // + 2 bytes for the index of the first children if necessary.
        // That index also uses bytes 0-6 of the previous byte.
        uint8_t s =
            uint8_t(has_sibling ? 0x80 : 0) | uint8_t(has_children ? 0x40 : 0);
        bytes.push_back(s);
        if (has_children) {
          children_offsets.emplace_back(
              ChildrenOffset{n->children[0].get(), bytes.size() - 1, false});
          bytes.push_back(0x00);
          bytes.push_back(0x00);
        }
      }
      add_children(n->children);
    }

    // Once all the nodes are in the inndex
    // Fill the bytes we left to indicate the position
    // of the children
    for (auto &&parent : children_offsets) {
      const auto it = offsets.find(parent.firstChild);
      assert(it != offsets.end());
      std::size_t pos = it->second;
      if (parent.hasValue) {
        bytes[parent.offset] = ((pos >> 16) & 0xFF);
      } else {
        bytes[parent.offset] =
            bytes[parent.offset] | uint8_t((pos >> 16) & 0xFF);
      }
      bytes[parent.offset + 1] = ((pos >> 8) & 0xFF);
      bytes[parent.offset + 2] = pos & 0xFF;
    }

    // Add some padding so that the deserialization code
    // doesn't try to read past the enf of the array.
    bytes.push_back(0);
    bytes.push_back(0);
    bytes.push_back(0);
    bytes.push_back(0);
    bytes.push_back(0);
    bytes.push_back(0);

    return bytes;
  }

private:
  void collect_keys(node *n, std::set<std::string> &v) {
    v.insert(n->name);
    for (auto &&child : n->children) {
      collect_keys(child.get(), v);
    }
  }

  // Merge sequences of 1-character nodes
  // This greatly reduce the total number of nodes,
  // and therefore the size of the index.
  // When the tree gets serialized, we only have 5 bytes to store the
  // size of a name. Overlong names (>32 characters) are therefore
  // kep into separate nodes
  void compact(node *n) {
    for (auto &&child : n->children) {
      compact(child.get());
    }
    if (n->parent && n->parent->children.size() == 1 && !n->parent->value &&
        (n->parent->name.size() + n->name.size() <= 32)) {
      n->parent->value = n->value;
      n->parent->name += n->name;
      n->parent->children = std::move(n->children);
      for (auto &c : n->parent->children) {
        c->parent = n->parent;
      }
    }
  }
  struct node {
    node(std::string name, node *parent = nullptr)
        : name(name), parent(parent) {}

    std::vector<std::unique_ptr<node>> children;
    std::string name;
    node *parent = nullptr;
    llvm::Optional<char32_t> value;
  };

  std::unique_ptr<node> root = std::make_unique<node>("");
};

int main(int argc, char **argv) {
  printf("Unicode name -> codepoint mapping generator\n"
         "Usage: %s UnicodeData.txt NameAliases.txt output\n\n",
         argv[0]);
  printf("NameAliases.txt can be found at "
         "https://unicode.org/Public/14.0.0/ucd/NameAliases.txt\n"
         "UnicodeData.txt can be found at "
         "https://unicode.org/Public/14.0.0/ucd/UnicodeData.txt\n\n");

  if (argc != 4)
    return EXIT_FAILURE;

  auto out = fopen(argv[3], "w");
  if (!out) {
    printf("Error creating output file.\n");
    return EXIT_FAILURE;
  }

  trie t;
  uint32_t count = 0;
  auto entries = load_data(argv[1], argv[2]);
  for (std::pair<char32_t, std::string> entry : entries) {
    const auto &codepoint = entry.first;
    const auto &name = entry.second;
    // A name is at least 2 characters long.
    // Fixme: Is this actually true?
    if (name.size() < 2)
      continue;
    // Ignore names which are not valid.
    if (!std::all_of(name.begin(), name.end(),
                     [](char C) { return llvm::is_contained(LETTERS, C); })) {
      continue;
    }
    printf("%06x: %s\n", codepoint, name.c_str());
    t.insert(name, codepoint);
    count++;
  }
  t.compact();

  std::pair<std::string, std::vector<uint8_t>> data = t.Serialize();
  const auto &dict = data.first;
  const auto &tree = data.second;

  fprintf(out,
          "//===-------------- Support/UnicodeNameToCodepointGenerated.cpp "
          "-----------===//\n"
          "//\n"
          "//This file was generated using %s. Do not edit manually."
          "//\n"
          "//"
          "===-----------------------------------------------------------------"
          "-----===//\n"
          "#include <cstdint>\n"
          "#include \"llvm/Support/Compiler.h\"\n",
          argv[0]);

  fprintf(out, "namespace llvm { namespace sys { namespace unicode { \n");
  fprintf(out, "const char* UnicodeNameToCodepointDict = \"%s\";\n",
          dict.c_str());

  fprintf(out, "uint8_t UnicodeNameToCodepointIndex_[%lu] = {\n",
          tree.size() + 1);

  for (auto b : tree) {
    fprintf(out, "0x%02x,", b);
  }

  fprintf(out, "0};");
  fprintf(out, "uint8_t* UnicodeNameToCodepointIndex = "
               "UnicodeNameToCodepointIndex_; \n");
  fprintf(out, "std::size_t UnicodeNameToCodepointIndexSize = %lu; \n",
          tree.size() + 1);
  fprintf(out, "\n}}}\n");
  fclose(out);
  printf("Generated %s: %u Files.\nIndex: %f kB, Dictionary: %f kB.\nDone\n\n",
         argv[3], count, tree.size() / 1024.0, dict.size() / 1024.0);
}
