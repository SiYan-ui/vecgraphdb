#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <optional>
#include <unordered_map>

namespace vecgraphdb {

struct SimilarResult {
  std::string id;
  float score; // cosine similarity in [-1,1]
};

struct CorrResult {
  std::string id;
  float corr; // correlation coefficient in (-1,1)
};

class VecGraphDB {
public:
  // db_path: directory where data files live.
  // dim: vector dimension.
  // space: currently only "cosine" is supported.
  VecGraphDB(std::string db_path, std::size_t dim, std::string space = "cosine");

  std::size_t dim() const noexcept { return dim_; }
  const std::string& path() const noexcept { return db_path_; }

  // Adds a single vector (no edge created).
  void add_vector(const std::string& id, const std::vector<float>& vec);

  // Adds two vectors and creates an undirected edge between them.
  // If corr is not provided, it will be computed using Pearson correlation.
  void add_pair(const std::string& id_a, const std::vector<float>& a,
                const std::string& id_b, const std::vector<float>& b,
                std::optional<float> corr = std::nullopt);

  // Top-k most similar vectors to query (cosine similarity).
  std::vector<SimilarResult> topk_similar(const std::vector<float>& query, std::size_t k) const;

  // Returns top-k highest-correlation neighbors using a vector query.
  // Anchor selection:
  //  - If you want explicit anchoring by an existing id, use topk_correlated_by_id.
  //  - Otherwise: finds nearest stored vector as anchor.
  // Returns neighbors sorted by corr desc.
  std::vector<CorrResult> topk_correlated(const std::vector<float>& query, std::size_t k) const;

  // Returns top-k highest-correlation neighbors for an existing stored vector id.
  // Throws if id does not exist.
  std::vector<CorrResult> topk_correlated_by_id(const std::string& id, std::size_t k) const;

  // Persist current state (vectors/ids are append-only already; this mainly snapshots HNSW).
  void flush();

private:
  std::string db_path_;
  std::size_t dim_;

  // internal id (0..N-1) <-> external string id
  std::vector<std::string> ids_;
  // external id -> internal id (MVP policy: ids are unique; duplicates rejected)
  std::unordered_map<std::string, std::uint32_t> id_to_internal_;

  // adjacency list: for each internal node, list of (neighbor_internal_id, corr)
  std::vector<std::vector<std::pair<std::uint32_t, float>>> adj_;

  // HNSW index (pimpl)
  struct Impl;
  Impl* impl_;

  std::uint32_t add_vector_internal(const std::string& id, const std::vector<float>& vec);
  static float pearson_corr(const std::vector<float>& x, const std::vector<float>& y);

  void load_or_init();
  void ensure_dirs() const;
};

} // namespace vecgraphdb
