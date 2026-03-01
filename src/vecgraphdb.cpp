#include "vecgraphdb/vecgraphdb.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

#include "hnswlib/hnswlib.h"

namespace fs = std::filesystem;

namespace vecgraphdb {

static constexpr const char* kIdsFile = "ids.txt";
static constexpr const char* kVecFile = "vectors.f32";
static constexpr const char* kIndexFile = "index.hnsw";
static constexpr const char* kEdgesFile = "edges.bin";

struct VecGraphDB::Impl {
  std::unique_ptr<hnswlib::SpaceInterface<float>> space;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
  std::size_t dim = 0;
  std::size_t max_elements = 0;
};

static void write_u32(std::ofstream& out, std::uint32_t v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}
static void write_f32(std::ofstream& out, float v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}
static std::uint32_t read_u32(std::ifstream& in) {
  std::uint32_t v;
  in.read(reinterpret_cast<char*>(&v), sizeof(v));
  return v;
}
static float read_f32(std::ifstream& in) {
  float v;
  in.read(reinterpret_cast<char*>(&v), sizeof(v));
  return v;
}

VecGraphDB::VecGraphDB(std::string db_path, std::size_t dim, std::string space)
  : db_path_(std::move(db_path)), dim_(dim), impl_(new Impl()) {
  if (dim_ == 0) throw std::invalid_argument("dim must be > 0");
  if (space != "cosine") throw std::invalid_argument("only space=cosine is supported in MVP");
  impl_->dim = dim_;
  load_or_init();
}

static fs::path data_dir(const std::string& p) {
  return fs::path(p) / "data";
}

void VecGraphDB::ensure_dirs() const {
  fs::create_directories(data_dir(db_path_));
}

void VecGraphDB::load_or_init() {
  ensure_dirs();

  // Load ids
  ids_.clear();
  id_to_internal_.clear();
  {
    std::ifstream in(data_dir(db_path_) / kIdsFile);
    if (in.good()) {
      std::string line;
      while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::uint32_t idx = static_cast<std::uint32_t>(ids_.size());
        // If data already has duplicates, keep the first occurrence as canonical.
        if (id_to_internal_.find(line) == id_to_internal_.end()) {
          id_to_internal_.emplace(line, idx);
        }
        ids_.push_back(line);
      }
    }
  }

  adj_.assign(ids_.size(), {});

  // Load edges
  {
    std::ifstream in(data_dir(db_path_) / kEdgesFile, std::ios::binary);
    if (in.good()) {
      // Format: u32 N, then for each i: u32 deg, then deg*(u32 j + f32 corr)
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
          adj_[i].push_back({j, c});
        }
      }
    }
  }

  // init HNSW
  impl_->space = std::make_unique<hnswlib::InnerProductSpace>(static_cast<int>(dim_));

  const std::size_t N = ids_.size();
  impl_->max_elements = std::max<std::size_t>(N + 1024, 1024);
  impl_->index = std::make_unique<hnswlib::HierarchicalNSW<float>>(impl_->space.get(), impl_->max_elements);

  fs::path index_path = data_dir(db_path_) / kIndexFile;
  if (fs::exists(index_path)) {
    impl_->index->loadIndex(index_path.string(), impl_->space.get(), impl_->max_elements);
  }

  // If index was empty but we have vectors, rebuild index from vector file.
  // We consider index empty if current element count is 0.
  if (N > 0 && impl_->index->cur_element_count == 0) {
    std::ifstream vin(data_dir(db_path_) / kVecFile, std::ios::binary);
    if (!vin.good()) throw std::runtime_error("vectors.f32 missing but ids.txt not empty");
    for (std::uint32_t i = 0; i < N; ++i) {
      std::vector<float> vec(dim_);
      vin.read(reinterpret_cast<char*>(vec.data()), sizeof(float) * dim_);
      if (!vin) throw std::runtime_error("vectors.f32 truncated");
      impl_->index->addPoint(vec.data(), i);
    }
  }
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

  if (id_to_internal_.find(id) != id_to_internal_.end()) {
    throw std::invalid_argument("duplicate id: " + id);
  }

  // Store normalized vectors so inner product == cosine similarity
  std::vector<float> normed = l2_normalize_copy(vec);

  // append vector to disk
  std::ofstream vout(data_dir(db_path_) / kVecFile, std::ios::binary | std::ios::app);
  if (!vout.good()) throw std::runtime_error("failed to open vectors.f32 for append");
  vout.write(reinterpret_cast<const char*>(normed.data()), sizeof(float) * dim_);

  // append id
  std::ofstream iout(data_dir(db_path_) / kIdsFile, std::ios::app);
  if (!iout.good()) throw std::runtime_error("failed to open ids.txt for append");
  iout << id << "\n";

  std::uint32_t internal = static_cast<std::uint32_t>(ids_.size());
  ids_.push_back(id);
  id_to_internal_.emplace(id, internal);
  adj_.push_back({});

  // expand HNSW if needed
  if (impl_->index->cur_element_count + 1 >= impl_->max_elements) {
    impl_->max_elements *= 2;
    impl_->index->resizeIndex(impl_->max_elements);
  }

  impl_->index->addPoint(normed.data(), internal);
  return internal;
}

void VecGraphDB::add_vector(const std::string& id, const std::vector<float>& vec) {
  (void)add_vector_internal(id, vec);
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
  if (!(c > -1.0f && c < 1.0f)) {
    // clamp strictly to (-1,1)
    c = std::max(-0.999999f, std::min(0.999999f, c));
  }

  std::uint32_t ia = add_vector_internal(id_a, a);
  std::uint32_t ib = add_vector_internal(id_b, b);

  adj_[ia].push_back({ib, c});
  adj_[ib].push_back({ia, c});
}

static std::vector<float> load_vector_at(const fs::path& vec_file, std::size_t dim, std::uint32_t idx) {
  std::ifstream in(vec_file, std::ios::binary);
  if (!in.good()) throw std::runtime_error("failed to open vectors.f32");
  std::uint64_t offset = static_cast<std::uint64_t>(idx) * sizeof(float) * dim;
  in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  std::vector<float> v(dim);
  in.read(reinterpret_cast<char*>(v.data()), sizeof(float) * dim);
  if (!in) throw std::runtime_error("failed to read vector at index");
  return v;
}

std::vector<SimilarResult> VecGraphDB::topk_similar(const std::vector<float>& query, std::size_t k) const {
  validate_vec(dim_, query, "query");
  if (ids_.empty() || k == 0) return {};

  std::vector<float> qn = l2_normalize_copy(query);
  auto res = impl_->index->searchKnn(qn.data(), static_cast<int>(k));

  std::vector<SimilarResult> out;
  out.reserve(res.size());
  while (!res.empty()) {
    auto [dist, label] = res.top();
    res.pop();
    // InnerProductSpace distance = 1 - dot(x,y) for normalized vectors.
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
  for (std::size_t i = 0; i < neigh.size() && out.size() < k; ++i) {
    out.push_back({ids.at(neigh[i].first), neigh[i].second});
  }
  return out;
}

std::vector<CorrResult> VecGraphDB::topk_correlated(const std::vector<float>& query, std::size_t k) const {
  validate_vec(dim_, query, "query");
  if (ids_.empty() || k == 0) return {};

  // anchor = nearest neighbor
  std::vector<float> qn = l2_normalize_copy(query);
  auto nn = impl_->index->searchKnn(qn.data(), 1);
  if (nn.empty()) return {};
  std::uint32_t anchor = static_cast<std::uint32_t>(nn.top().second);

  return topk_corr_from_anchor(ids_, adj_, anchor, k);
}

std::vector<CorrResult> VecGraphDB::topk_correlated_by_id(const std::string& id, std::size_t k) const {
  if (k == 0) return {};
  auto it = id_to_internal_.find(id);
  if (it == id_to_internal_.end()) {
    throw std::invalid_argument("unknown id: " + id);
  }
  return topk_corr_from_anchor(ids_, adj_, it->second, k);
}

void VecGraphDB::flush() {
  ensure_dirs();

  // Save HNSW index
  impl_->index->saveIndex((data_dir(db_path_) / kIndexFile).string());

  // Save edges (full snapshot)
  std::ofstream out(data_dir(db_path_) / kEdgesFile, std::ios::binary | std::ios::trunc);
  if (!out.good()) throw std::runtime_error("failed to write edges.bin");

  write_u32(out, static_cast<std::uint32_t>(adj_.size()));
  for (std::uint32_t i = 0; i < adj_.size(); ++i) {
    write_u32(out, static_cast<std::uint32_t>(adj_[i].size()));
    for (auto& [j, c] : adj_[i]) {
      write_u32(out, j);
      write_f32(out, c);
    }
  }
}

} // namespace vecgraphdb
