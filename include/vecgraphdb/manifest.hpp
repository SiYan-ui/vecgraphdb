#pragma once

#include "vecgraphdb/types.hpp"
#include <filesystem>
#include <fstream>
#include <string>
#include <stdexcept>
#include <cstring>

namespace vecgraphdb {

namespace fs = std::filesystem;

// Manifest file structure
struct Manifest {
  std::uint32_t version = kFormatVersion;
  std::uint32_t dim = 0;
  SpaceType space = SpaceType::Cosine;
  std::string created; // ISO 8601 timestamp
  
  // Index parameters
  std::size_t hnsw_M = 16;
  std::size_t hnsw_ef_construction = 200;
  std::size_t hnsw_ef_search = 100;
  
  static constexpr const char* kFilename = "MANIFEST.json";
  
  std::string to_json() const {
    return "{\"version\":" + std::to_string(version) +
           ",\"dim\":" + std::to_string(dim) +
           ",\"space\":\"" + space_type_to_string(space) + "\"" +
           ",\"created\":\"" + created + "\"" +
           ",\"hnsw_M\":" + std::to_string(hnsw_M) +
           ",\"hnsw_ef_construction\":" + std::to_string(hnsw_ef_construction) +
           ",\"hnsw_ef_search\":" + std::to_string(hnsw_ef_search) +
           "}";
  }
  
  static Manifest from_json(const std::string& json, const fs::path& path);
  
  void save(const fs::path& dir) const {
    fs::path p = dir / kFilename;
    std::ofstream out(p.string().c_str(), std::ios::trunc);
    if (!out.good()) throw std::runtime_error("failed to write MANIFEST.json");
    out << to_json();
    if (!out) throw std::runtime_error("failed to write MANIFEST.json");
  }
  
  static Manifest load_or_create(const fs::path& dir, std::uint32_t dim, 
                                  SpaceType space,
                                  std::size_t hnsw_M = 16,
                                  std::size_t hnsw_ef_construction = 200,
                                  std::size_t hnsw_ef_search = 100) {
    fs::path p = dir / kFilename;
    if (fs::exists(p)) {
      std::ifstream in(p.string().c_str());
      if (!in.good()) throw std::runtime_error("failed to read MANIFEST.json");
      std::string json((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
      return Manifest::from_json(json, p);
    }
    
    // Create new manifest
    Manifest m;
    m.version = kFormatVersion;
    m.dim = dim;
    m.space = space;
    m.hnsw_M = hnsw_M;
    m.hnsw_ef_construction = hnsw_ef_construction;
    m.hnsw_ef_search = hnsw_ef_search;
    
    // Set creation timestamp
    auto now = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    m.created = buf;
    
    m.save(dir);
    return m;
  }
};

// Simple JSON parser for manifest (minimal implementation)
inline Manifest Manifest::from_json(const std::string& json, const fs::path& path) {
  Manifest m;
  
  // Very simple parsing - look for known keys
  auto find_value = [&](const std::string& key) -> std::string {
    std::string search = "\"" + key + "\":";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos += search.length();
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    
    if (json[pos] == '"') {
      // String value
      auto start = pos + 1;
      auto end = json.find('"', start);
      return json.substr(start, end - start);
    } else {
      // Numeric value
      auto start = pos;
      auto end = json.find_first_of(",}", start);
      return json.substr(start, end - start);
    }
  };
  
  std::string ver_str = find_value("version");
  if (!ver_str.empty()) m.version = std::stoul(ver_str);
  
  std::string dim_str = find_value("dim");
  if (!dim_str.empty()) m.dim = std::stoul(dim_str);
  
  std::string space_str = find_value("space");
  if (!space_str.empty()) m.space = space_type_from_string(space_str);
  
  std::string created_str = find_value("created");
  if (!created_str.empty()) m.created = created_str;
  
  std::string m_str = find_value("hnsw_M");
  if (!m_str.empty()) m.hnsw_M = std::stoul(m_str);
  
  std::string ef_const_str = find_value("hnsw_ef_construction");
  if (!ef_const_str.empty()) m.hnsw_ef_construction = std::stoul(ef_const_str);
  
  std::string ef_search_str = find_value("hnsw_ef_search");
  if (!ef_search_str.empty()) m.hnsw_ef_search = std::stoul(ef_search_str);
  
  return m;
}

// Binary I/O helpers
inline void write_u32(std::ofstream& out, std::uint32_t v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

inline void write_u64(std::ofstream& out, std::uint64_t v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

inline void write_f32(std::ofstream& out, float v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

inline void write_header(std::ofstream& out, const FileHeader& h) {
  write_u32(out, h.magic);
  write_u32(out, h.version);
  write_u32(out, h.dim);
  write_u64(out, h.count);
  out.write(reinterpret_cast<const char*>(h.reserved), sizeof(h.reserved));
}

inline std::uint32_t read_u32(std::ifstream& in) {
  std::uint32_t v;
  in.read(reinterpret_cast<char*>(&v), sizeof(v));
  if (!in) throw std::runtime_error("unexpected EOF while reading u32");
  return v;
}

inline std::uint64_t read_u64(std::ifstream& in) {
  std::uint64_t v;
  in.read(reinterpret_cast<char*>(&v), sizeof(v));
  if (!in) throw std::runtime_error("unexpected EOF while reading u64");
  return v;
}

inline float read_f32(std::ifstream& in) {
  float v;
  in.read(reinterpret_cast<char*>(&v), sizeof(v));
  if (!in) throw std::runtime_error("unexpected EOF while reading f32");
  return v;
}

inline FileHeader read_header(std::ifstream& in) {
  FileHeader h;
  h.magic = read_u32(in);
  h.version = read_u32(in);
  h.dim = read_u32(in);
  h.count = read_u64(in);
  in.read(reinterpret_cast<char*>(h.reserved), sizeof(h.reserved));
  return h;
}

inline void validate_header(const FileHeader& h, std::uint32_t expected_magic,
                            std::uint32_t expected_dim, const std::string& filename) {
  if (h.magic != expected_magic) {
    throw std::runtime_error(filename + " magic mismatch: expected " + 
                             std::to_string(expected_magic) + 
                             ", got " + std::to_string(h.magic));
  }
  if (h.dim != expected_dim) {
    throw std::runtime_error(filename + " dimension mismatch: expected " +
                             std::to_string(expected_dim) +
                             ", got " + std::to_string(h.dim));
  }
}

} // namespace vecgraphdb
