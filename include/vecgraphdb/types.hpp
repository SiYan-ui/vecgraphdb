#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <stdexcept>

namespace vecgraphdb {

// File format version for backward compatibility
// Version history:
//   v1: initial format with meta.bin only
//   v2: added MANIFEST.json and binary headers for all files
constexpr std::uint32_t kFormatVersion = 2;

// Magic numbers for file identification
constexpr std::uint32_t kMetaMagic = 0x56474442;      // 'VGDB'
constexpr std::uint32_t kVectorMagic = 0x56474456;    // 'VGDV'
constexpr std::uint32_t kEdgesMagic = 0x56474445;     // 'VGDE'
constexpr std::uint32_t kIndexMagic = 0x56474448;     // 'VGDH'

// Space types
enum class SpaceType : std::uint32_t {
  Cosine = 1,
};

inline std::string space_type_to_string(SpaceType type) {
  switch (type) {
    case SpaceType::Cosine: return "cosine";
    default: return "unknown";
  }
}

inline SpaceType space_type_from_string(const std::string& s) {
  if (s == "cosine") return SpaceType::Cosine;
  throw std::invalid_argument("unknown space type: " + s);
  return SpaceType::Cosine; // Never reached, suppresses warning
}

// Binary file header structure (40 bytes total)
// Layout: magic(4) + version(4) + dim(4) + count(8) + reserved(20)
struct FileHeader {
  std::uint32_t magic;
  std::uint32_t version;
  std::uint32_t dim;
  std::uint64_t count;
  std::uint8_t reserved[20];
  
  static constexpr size_t kSize = 40;
  
  FileHeader() : magic(0), version(0), dim(0), count(0) {
    std::memset(reserved, 0, sizeof(reserved));
  }
  
  FileHeader(std::uint32_t m, std::uint32_t v, std::uint32_t d, std::uint64_t c)
    : magic(m), version(v), dim(d), count(c) {
    std::memset(reserved, 0, sizeof(reserved));
  }
};

static_assert(sizeof(FileHeader) == FileHeader::kSize, "FileHeader size mismatch");

// Result structures
struct SimilarResult {
  std::string id;
  float score; // cosine similarity in [-1,1]
};

struct CorrResult {
  std::string id;
  float corr; // correlation coefficient in (-1,1)
};

// Database statistics
struct DbStats {
  std::size_t count = 0;
  std::size_t dim = 0;
  std::uint64_t vectors_bytes = 0;
  std::uint64_t ids_bytes = 0;
  std::uint64_t edges_bytes = 0;
  std::uint64_t index_bytes = 0;
};

} // namespace vecgraphdb
