#include "vecgraphdb/vecgraphdb.hpp"
#include "vecgraphdb/manifest.hpp"
#include "vecgraphdb/types.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <cstring>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <sys/mman.h>
  #include <unistd.h>
  #include <fcntl.h>
#endif

#include "hnswlib/hnswlib.h"

namespace fs = std::filesystem;

namespace vecgraphdb {

// File names
static constexpr const char* kIdsFile = "ids.txt";
static constexpr const char* kVecFile = "vectors.f32";
static constexpr const char* kIndexFile = "index.hnsw";
static constexpr const char* kEdgesFile = "edges.bin";

// Legacy meta file (kept for backward compatibility)
static constexpr const char* kMetaFile = "meta.bin";

struct VecGraphDB::Impl {
  std::unique_ptr<hnswlib::SpaceInterface<float>> space;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
  std::size_t dim = 0;
  std::size_t max_elements = 0;
  std::size_t hnsw_M = 16;
  std::size_t hnsw_ef_construction = 200;
  std::size_t hnsw_ef_search = 100;
  
  // Mmap support for vector reads
#ifdef _WIN32
  HANDLE hFile = INVALID_HANDLE_VALUE;
  HANDLE hMapping = INVALID_HANDLE_VALUE;
  const float* mapped_data = nullptr;
  std::size_t mapped_size = 0;
#else
  int fd = -1;
  const float* mapped_data = MAP_FAILED;
  std::size_t mapped_size = 0;
#endif
  
  ~Impl() {
    // Unmap vectors on destruction
#ifdef _WIN32
    if (mapped_data != nullptr) {
      UnmapViewOfFile(mapped_data);
    }
    if (hMapping != INVALID_HANDLE_VALUE) {
      CloseHandle(hMapping);
    }
    if (hFile != INVALID_HANDLE_VALUE) {
      CloseHandle(hFile);
    }
#else
    if (mapped_data != MAP_FAILED) {
      munmap(const_cast<float*>(mapped_data), mapped_size);
    }
    if (fd >= 0) {
      close(fd);
    }
#endif
  }
};

// Read vector from mmap or file (implementation after Impl is complete)
static std::vector<float> read_vector(const VecGraphDB::Impl* impl, const fs::path& vec_file, 
                                       std::size_t dim, std::uint32_t idx) {
#ifdef _WIN32
  if (impl && impl->mapped_data != nullptr) {
    const float* ptr = impl->mapped_data + idx * dim;
    return std::vector<float>(ptr, ptr + dim);
  }
#else
  if (impl && impl->mapped_data != MAP_FAILED) {
    const float* ptr = impl->mapped_data + idx * dim;
    return std::vector<float>(ptr, ptr + dim);
  }
#endif
  
  // Fallback to file I/O
  std::ifstream in(vec_file.string().c_str(), std::ios::binary);
  if (!in.good()) throw std::runtime_error("failed to open vectors.f32");
  std::uint64_t offset = static_cast<std::uint64_t>(idx) * sizeof(float) * dim;
  in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  std::vector<float> v(dim);
  in.read(reinterpret_cast<char*>(v.data()), sizeof(float) * dim);
  if (!in) throw std::runtime_error("failed to read vector at index");
  return v;
}

VecGraphDB::VecGraphDB(std::string db_path, std::size_t dim, std::string space,
                       std::size_t hnsw_M, std::size_t hnsw_ef_construction,
                       std::size_t hnsw_ef_search)
  : db_path_(std::move(db_path)), dim_(dim), impl_(std::make_unique<Impl>()) {
  if (dim_ == 0) throw std::invalid_argument("dim must be > 0");
  if (space != "cosine") throw std::invalid_argument("only space=cosine is supported in MVP");
  if (hnsw_M == 0) throw std::invalid_argument("hnsw_M must be > 0");
  if (hnsw_ef_construction == 0) throw std::invalid_argument("hnsw_ef_construction must be > 0");
  if (hnsw_ef_search == 0) throw std::invalid_argument("hnsw_ef_search must be > 0");
  impl_->dim = dim_;
  impl_->hnsw_M = hnsw_M;
  impl_->hnsw_ef_construction = hnsw_ef_construction;
  impl_->hnsw_ef_search = hnsw_ef_search;
  load_or_init();
}

VecGraphDB::~VecGraphDB() = default;

static fs::path data_dir(const std::string& p) {
  return fs::path(p) / "data";
}

void VecGraphDB::ensure_dirs() const {
  fs::create_directories(data_dir(db_path_));
}

// Load legacy meta.bin (version 1 format) for backward compatibility
static bool load_legacy_meta(const fs::path& dir, std::uint32_t& dim_out) {
  fs::path p = dir / kMetaFile;
  if (!fs::exists(p)) return false;
  
  std::ifstream in(p.string().c_str(), std::ios::binary);
  if (!in.good()) return false;
  
  try {
    std::uint32_t magic = read_u32(in);
    std::uint32_t ver = read_u32(in);
    std::uint32_t file_dim = read_u32(in);
    std::uint32_t space_val = read_u32(in);
    
    if (magic != kMetaMagic) return false;
    if (ver != 1) return false;
    if (space_val != static_cast<std::uint32_t>(SpaceType::Cosine)) return false;
    
    dim_out = file_dim;
    return true;
  } catch (...) {
    return false;
  }
}

// Setup mmap for reading vectors
static void setup_mmap(Impl* impl, const fs::path& vec_file) {
  if (!fs::exists(vec_file)) return;
  
  std::size_t file_size = static_cast<std::size_t>(fs::file_size(vec_file));
  if (file_size == 0) return;
  
#ifdef _WIN32
  impl->hFile = CreateFileW(
    vec_file.c_str(),
    GENERIC_READ,
    FILE_SHARE_READ,
    nullptr,
    OPEN_EXISTING,
    FILE_ATTRIBUTE_NORMAL,
    nullptr
  );
  
  if (impl->hFile == INVALID_HANDLE_VALUE) {
    return; // Fall back to regular I/O
  }
  
  impl->hMapping = CreateFileMappingW(impl->hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (impl->hMapping == INVALID_HANDLE_VALUE) {
    CloseHandle(impl->hFile);
    impl->hFile = INVALID_HANDLE_VALUE;
    return;
  }
  
  impl->mapped_data = static_cast<const float*>(MapViewOfFile(impl->hMapping, FILE_MAP_READ, 0, 0, 0));
  if (impl->mapped_data == nullptr) {
    CloseHandle(impl->hMapping);
    CloseHandle(impl->hFile);
    impl->hMapping = INVALID_HANDLE_VALUE;
    impl->hFile = INVALID_HANDLE_VALUE;
    return;
  }
  
  impl->mapped_size = file_size;
#else
  impl->fd = open(vec_file.c_str(), O_RDONLY);
  if (impl->fd < 0) return;
  
  impl->mapped_data = static_cast<const float*>(mmap(nullptr, file_size, PROT_READ, MAP_SHARED, impl->fd, 0));
  if (impl->mapped_data == MAP_FAILED) {
    close(impl->fd);
    impl->fd = -1;
    return;
  }
  
  impl->mapped_size = file_size;
#endif
}

// Read vector from mmap or file (defined after Impl is complete)
static std::vector<float> read_vector(const VecGraphDB::Impl* impl, const fs::path& vec_file, 
                                       std::size_t dim, std::uint32_t idx);

void VecGraphDB::load_or_init() {
  ensure_dirs();
  fs::path dir = data_dir(db_path_);
  
  // Try to load manifest (v2) or legacy meta (v1)
  Manifest manifest;
  fs::path manifest_path = dir / Manifest::kFilename;
  
  if (fs::exists(manifest_path)) {
    // Load v2 manifest
    manifest = Manifest::load_or_create(dir, static_cast<std::uint32_t>(dim_), 
                                         SpaceType::Cosine,
                                         impl_->hnsw_M, impl_->hnsw_ef_construction,
                                         impl_->hnsw_ef_search);
    
    // Validate dimensions match
    if (manifest.dim != dim_) {
      throw std::runtime_error("MANIFEST.json dimension mismatch: expected " +
                               std::to_string(dim_) + ", got " + std::to_string(manifest.dim));
    }
  } else {
    // Check for legacy meta.bin
    std::uint32_t legacy_dim = 0;
    if (load_legacy_meta(dir, legacy_dim)) {
      if (legacy_dim != dim_) {
        throw std::runtime_error("meta.bin dimension mismatch");
      }
      // Upgrade: create v2 manifest
      manifest = Manifest::load_or_create(dir, dim_, SpaceType::Cosine,
                                           impl_->hnsw_M, impl_->hnsw_ef_construction,
                                           impl_->hnsw_ef_search);
    } else {
      // No existing metadata, create new
      manifest = Manifest::load_or_create(dir, dim_, SpaceType::Cosine,
                                           impl_->hnsw_M, impl_->hnsw_ef_construction,
                                           impl_->hnsw_ef_search);
    }
  }
  
  // Load ids
  {
    LockGuard lock(mutex_);
    ids_.clear();
    id_to_internal_.clear();
    
    fs::path ids_file = dir / kIdsFile;
    std::ifstream in(ids_file.string().c_str());
    if (in.good()) {
      std::string line;
      while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::uint32_t idx = static_cast<std::uint32_t>(ids_.size());
        if (id_to_internal_.find(line) != id_to_internal_.end()) {
          throw std::runtime_error("ids.txt contains duplicate id: " + line);
        }
        id_to_internal_.emplace(line, idx);
        ids_.push_back(line);
      }
    }
  }
  
  // Validate vectors.f32 length vs ids count (with header support)
  {
    fs::path vec_path = dir / kVecFile;
    std::uint64_t expected_data_size = static_cast<std::uint64_t>(ids_.size()) *
                                        sizeof(float) * static_cast<std::uint64_t>(dim_);
    
    if (!ids_.empty()) {
      if (!fs::exists(vec_path)) {
        throw std::runtime_error("vectors.f32 missing but ids.txt not empty");
      }
      
      std::uint64_t actual_size = fs::file_size(vec_path);
      
      // Check if file has v2 header
      std::ifstream check_in(vec_path.string().c_str(), std::ios::binary);
      FileHeader header = read_header(check_in);
      
      if (header.magic == kVectorMagic) {
        // Has header, validate and adjust expected size
        validate_header(header, kVectorMagic, static_cast<std::uint32_t>(dim_), "vectors.f32");
        expected_data_size += FileHeader::kSize; // Header is part of file size
      }
      
      if (actual_size != expected_data_size) {
        std::string msg = "vectors.f32 size mismatch (corrupt/truncated). Expected: ";
        msg += std::to_string(expected_data_size);
        msg += ", Got: ";
        msg += std::to_string(actual_size);
        throw std::runtime_error(msg);
      }
    }
  }
  
  // Load edges (with header support)
  {
    LockGuard lock(mutex_);
    adj_.assign(ids_.size(), {});
    
    fs::path edges_file = dir / kEdgesFile;
    std::ifstream in(edges_file.string().c_str(), std::ios::binary);
    if (in.good()) {
      // Try to read header first
      FileHeader header = read_header(in);
      
      if (header.magic == kEdgesMagic) {
        // V2 format with header
        validate_header(header, kEdgesMagic, static_cast<std::uint32_t>(dim_), "edges.bin");
        
        std::uint32_t N = static_cast<std::uint32_t>(header.count);
        if (N != ids_.size()) {
          throw std::runtime_error("edges.bin node count mismatch with ids.txt");
        }
        
        adj_.assign(N, {});
        for (std::uint32_t i = 0; i < N; ++i) {
          std::uint32_t deg = read_u32(in);
          adj_[i].reserve(deg);
          for (std::uint32_t d = 0; d < deg; ++d) {
            std::uint32_t j = read_u32(in);
            float c = read_f32(in);
            if (j >= N) throw std::runtime_error("edges.bin contains out-of-range neighbor id");
            adj_[i].push_back({j, c});
          }
        }
      } else {
        // V1 format without header - rewind and read old format
        in.seekg(0, std::ios::beg);
        std::uint32_t N = read_u32(in);
        if (N != ids_.size()) {
          throw std::runtime_error("edges.bin node count mismatch with ids.txt");
        }
        
        adj_.assign(N, {});
        for (std::uint32_t i = 0; i < N; ++i) {
          std::uint32_t deg = read_u32(in);
          adj_[i].reserve(deg);
          for (std::uint32_t d = 0; d < deg; ++d) {
            std::uint32_t j = read_u32(in);
            float c = read_f32(in);
            if (j >= N) throw std::runtime_error("edges.bin contains out-of-range neighbor id");
            adj_[i].push_back({j, c});
          }
        }
      }
    }
  }
  
  // Initialize HNSW
  impl_->space = std::make_unique<hnswlib::InnerProductSpace>(static_cast<int>(dim_));
  
  const std::size_t N = ids_.size();
  impl_->max_elements = std::max<std::size_t>(N + 1024, 1024);
  impl_->index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
      impl_->space.get(), impl_->max_elements,
      static_cast<int>(impl_->hnsw_M), static_cast<int>(impl_->hnsw_ef_construction));
  
  // Load index
  fs::path index_path = dir / kIndexFile;
  if (fs::exists(index_path)) {
    try {
      impl_->index->loadIndex(index_path.string(), impl_->space.get(), impl_->max_elements);
    } catch (...) {
      // Index load failed, will rebuild below
    }
  }
  impl_->index->setEf(static_cast<int>(impl_->hnsw_ef_search));
  
  // Rebuild index from vectors if needed
  if (N > 0 && impl_->index->cur_element_count == 0) {
    fs::path vec_path = dir / kVecFile;
    for (std::uint32_t i = 0; i < N; ++i) {
      std::vector<float> vec = read_vector(impl_.get(), vec_path, dim_, i);
      impl_->index->addPoint(vec.data(), i);
    }
  }
  
  // Setup mmap for future reads
  setup_mmap(impl_.get(), dir / kVecFile);
}

static void validate_vec(std::size_t dim, const std::vector<float>& v, const char* name) {
  if (v.size() != dim) {
    throw std::invalid_argument(std::string(name) + " dimension mismatch");
  }
}

static std::vector<float> l2_normalize_copy(const std::vector<float>& v) {
  double ss = 0.0;
  for (float x : v) ss += static_cast<double>(x) * x;
  double n = std::sqrt(ss);
  if (n <= 0.0) return v;
  std::vector<float> out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) out[i] = static_cast<float>(v[i] / n);
  return out;
}

std::uint32_t VecGraphDB::add_vector_internal(const std::string& id, const std::vector<float>& vec) {
  validate_vec(dim_, vec, "vec");
  
  // Use lock for modifications
  LockGuard lock(mutex_);
  
  if (id_to_internal_.find(id) != id_to_internal_.end()) {
    throw std::invalid_argument("duplicate id: " + id);
  }
  
  // Store normalized vectors so inner product == cosine similarity
  std::vector<float> normed = l2_normalize_copy(vec);
  
  // Append vector to disk
  std::ofstream vout((data_dir(db_path_) / kVecFile).string().c_str(), std::ios::binary | std::ios::app);
  if (!vout.good()) throw std::runtime_error("failed to open vectors.f32 for append");
  vout.write(reinterpret_cast<const char*>(normed.data()), sizeof(float) * dim_);
  
  // Append id
  std::ofstream iout((data_dir(db_path_) / kIdsFile).string().c_str(), std::ios::app);
  if (!iout.good()) throw std::runtime_error("failed to open ids.txt for append");
  iout << id << "\n";
  
  std::uint32_t internal = static_cast<std::uint32_t>(ids_.size());
  ids_.push_back(id);
  id_to_internal_.emplace(id, internal);
  adj_.push_back({});
  
  // Expand HNSW if needed
  if (impl_->index->cur_element_count + 1 >= impl_->max_elements) {
    lock.unlock(); // Release lock before resize (which may take time)
    reserve(impl_->max_elements * 2);
    lock.lock();
  }
  
  impl_->index->addPoint(normed.data(), internal);
  return internal;
}

void VecGraphDB::add_vector(const std::string& id, const std::vector<float>& vec) {
  (void)add_vector_internal(id, vec);
}

void VecGraphDB::add_vectors_batch(const std::vector<std::string>& ids,
                                   const std::vector<std::vector<float>>& vecs) {
  if (ids.size() != vecs.size()) {
    throw std::invalid_argument("ids and vecs must have same length");
  }
  if (ids.empty()) return;
  
  std::unordered_set<std::string> seen;
  seen.reserve(ids.size());
  
  // Pre-check for duplicates (under read lock)
  {
    LockGuard lock(mutex_);
    for (const auto& id : ids) {
      if (id_to_internal_.find(id) != id_to_internal_.end()) {
        throw std::invalid_argument("duplicate id: " + id);
      }
      if (!seen.insert(id).second) {
        throw std::invalid_argument("duplicate id in batch: " + id);
      }
    }
  }
  
  // Ensure HNSW capacity
  {
    LockGuard lock(mutex_);
    std::size_t needed = ids_.size() + ids.size();
    if (needed >= impl_->max_elements) {
      std::size_t target = impl_->max_elements;
      while (target <= needed) target *= 2;
      lock.unlock();
      reserve(target);
      lock.lock();
    }
  }
  
  // Open files for append
  std::ofstream vout((data_dir(db_path_) / kVecFile).string().c_str(), std::ios::binary | std::ios::app);
  if (!vout.good()) throw std::runtime_error("failed to open vectors.f32 for append");
  std::ofstream iout((data_dir(db_path_) / kIdsFile).string().c_str(), std::ios::app);
  if (!iout.good()) throw std::runtime_error("failed to open ids.txt for append");
  
  // Add all vectors under a single write lock
  LockGuard lock(mutex_);
  for (std::size_t i = 0; i < ids.size(); ++i) {
    const auto& id = ids[i];
    const auto& vec = vecs[i];
    validate_vec(dim_, vec, "vec");
    
    std::vector<float> normed = l2_normalize_copy(vec);
    vout.write(reinterpret_cast<const char*>(normed.data()), sizeof(float) * dim_);
    iout << id << "\n";
    
    std::uint32_t internal = static_cast<std::uint32_t>(ids_.size());
    ids_.push_back(id);
    id_to_internal_.emplace(id, internal);
    adj_.push_back({});
    
    impl_->index->addPoint(normed.data(), internal);
  }
}

float VecGraphDB::pearson_corr(const std::vector<float>& x, const std::vector<float>& y) {
  if (x.size() != y.size() || x.empty()) return 0.0f;
  double mean_x = 0, mean_y = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    mean_x += x[i];
    mean_y += y[i];
  }
  mean_x /= x.size();
  mean_y /= y.size();
  
  double num = 0;
  double den_x = 0;
  double den_y = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    double dx = x[i] - mean_x;
    double dy = y[i] - mean_y;
    num += dx * dy;
    den_x += dx * dx;
    den_y += dy * dy;
  }
  double den = std::sqrt(den_x * den_y);
  if (den == 0) return 0.0f;
  double r = num / den;
  if (r >= 1.0) r = 0.999999;
  if (r <= -1.0) r = -0.999999;
  return static_cast<float>(r);
}

void VecGraphDB::add_pair(const std::string& id_a, const std::vector<float>& a,
                          const std::string& id_b, const std::vector<float>& b,
                          std::optional<float> corr) {
  validate_vec(dim_, a, "a");
  validate_vec(dim_, b, "b");
  
  float c = corr.has_value() ? *corr : pearson_corr(a, b);
  if (std::isnan(c) || std::isinf(c)) {
    throw std::invalid_argument("corr must be finite");
  }
  if (!(c > -1.0f && c < 1.0f)) {
    c = std::max(-0.999999f, std::min(0.999999f, c));
  }
  
  std::uint32_t ia = add_vector_internal(id_a, a);
  std::uint32_t ib = add_vector_internal(id_b, b);
  
  LockGuard lock(mutex_);
  adj_[ia].push_back({ib, c});
  adj_[ib].push_back({ia, c});
}

std::vector<SimilarResult> VecGraphDB::topk_similar(const std::vector<float>& query, std::size_t k) const {
  validate_vec(dim_, query, "query");
  
  LockGuard lock(mutex_);
  
  if (ids_.empty() || k == 0) return {};
  
  std::vector<float> qn = l2_normalize_copy(query);
  auto res = impl_->index->searchKnn(qn.data(), static_cast<int>(k));
  
  std::vector<SimilarResult> out;
  out.reserve(res.size());
  while (!res.empty()) {
    auto [dist, label] = res.top();
    res.pop();
    float score = 1.0f - static_cast<float>(dist);
    out.push_back({ids_.at(static_cast<std::size_t>(label)), score});
  }
  std::reverse(out.begin(), out.end());
  return out;
}

static std::vector<CorrResult> topk_corr_from_anchor(const std::vector<std::string>& ids,
                                                     const std::vector<std::vector<std::pair<std::uint32_t, float>>>& adj,
                                                     std::uint32_t anchor,
                                                     std::size_t k) {
  auto neigh = adj.at(anchor);
  std::sort(neigh.begin(), neigh.end(), [](auto& a, auto& b) { return a.second > b.second; });
  
  std::vector<CorrResult> out;
  out.reserve(std::min(k, neigh.size()));
  for (std::size_t i = 0; i < neigh.size() && out.size() < k; ++i) {
    out.push_back({ids.at(neigh[i].first), neigh[i].second});
  }
  return out;
}

std::vector<CorrResult> VecGraphDB::topk_correlated(const std::vector<float>& query, std::size_t k) const {
  validate_vec(dim_, query, "query");
  
  LockGuard lock(mutex_);
  
  if (ids_.empty() || k == 0) return {};
  
  std::vector<float> qn = l2_normalize_copy(query);
  auto nn = impl_->index->searchKnn(qn.data(), 1);
  if (nn.empty()) return {};
  std::uint32_t anchor = static_cast<std::uint32_t>(nn.top().second);
  
  return topk_corr_from_anchor(ids_, adj_, anchor, k);
}

std::vector<CorrResult> VecGraphDB::topk_correlated_by_id(const std::string& id, std::size_t k) const {
  if (k == 0) return {};
  
  LockGuard lock(mutex_);
  
  auto it = id_to_internal_.find(id);
  if (it == id_to_internal_.end()) {
    throw std::invalid_argument("unknown id: " + id);
  }
  return topk_corr_from_anchor(ids_, adj_, it->second, k);
}

void VecGraphDB::flush() {
  ensure_dirs();
  
  // Acquire read lock during flush (allows concurrent reads)
  LockGuard lock(mutex_);
  
  // Save HNSW index
  impl_->index->saveIndex((data_dir(db_path_) / kIndexFile).string());
  
  // Save edges with v2 header - write to tmp then rename
  fs::path dir = data_dir(db_path_);
  fs::path tmp = dir / "edges.bin.tmp";
  fs::path dst = dir / kEdgesFile;
  std::ofstream out(tmp.string().c_str(), std::ios::binary | std::ios::trunc);
  if (!out.good()) throw std::runtime_error("failed to write edges.bin");
  
  // Write v2 header
  FileHeader header(kEdgesMagic, kFormatVersion, static_cast<std::uint32_t>(dim_), 
                    static_cast<std::uint64_t>(adj_.size()));
  write_header(out, header);
  
  // Write adjacency list
  for (std::uint32_t i = 0; i < adj_.size(); ++i) {
    write_u32(out, static_cast<std::uint32_t>(adj_[i].size()));
    for (auto& [j, c] : adj_[i]) {
      write_u32(out, j);
      write_f32(out, c);
    }
  }
  out.close();
  fs::rename(tmp, dst);
}

void VecGraphDB::reserve(std::size_t max_elements) {
  LockGuard lock(mutex_);
  if (max_elements <= impl_->max_elements) return;
  impl_->max_elements = max_elements;
  impl_->index->resizeIndex(impl_->max_elements);
}

void VecGraphDB::set_hnsw_ef_search(std::size_t ef_search) {
  if (ef_search == 0) throw std::invalid_argument("hnsw_ef_search must be > 0");
  
  LockGuard lock(mutex_);
  impl_->hnsw_ef_search = ef_search;
  impl_->index->setEf(static_cast<int>(ef_search));
}

DbStats VecGraphDB::get_stats() const {
  DbStats s;
  
  LockGuard lock(mutex_);
  s.count = ids_.size();
  s.dim = dim_;
  
  fs::path dir = data_dir(db_path_);
  fs::path vec_file = dir / kVecFile;
  fs::path ids_file = dir / kIdsFile;
  fs::path edges_file = dir / kEdgesFile;
  fs::path index_file = dir / kIndexFile;
  
  if (fs::exists(vec_file)) s.vectors_bytes = fs::file_size(vec_file);
  if (fs::exists(ids_file)) s.ids_bytes = fs::file_size(ids_file);
  if (fs::exists(edges_file)) s.edges_bytes = fs::file_size(edges_file);
  if (fs::exists(index_file)) s.index_bytes = fs::file_size(index_file);
  return s;
}

} // namespace vecgraphdb
