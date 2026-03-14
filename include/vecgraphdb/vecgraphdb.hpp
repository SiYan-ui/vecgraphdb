#pragma once

#include "vecgraphdb/types.hpp"
#include "vecgraphdb/thread_utils.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <unordered_map>

namespace vecgraphdb {

class VecGraphDB {
public:
  // db_path: directory where data files live.
  // dim: vector dimension.
  // space: currently only "cosine" is supported.
  // hnsw_M: HNSW M parameter (graph degree)
  // hnsw_ef_construction: construction ef
  // hnsw_ef_search: search ef (can be tuned via set_hnsw_ef_search)
  VecGraphDB(std::string db_path, std::size_t dim, std::string space = "cosine",
             std::size_t hnsw_M = 16, std::size_t hnsw_ef_construction = 200,
             std::size_t hnsw_ef_search = 100);
  ~VecGraphDB();

  VecGraphDB(const VecGraphDB&) = delete;
  VecGraphDB& operator=(const VecGraphDB&) = delete;
  VecGraphDB(VecGraphDB&&) noexcept = default;
  VecGraphDB& operator=(VecGraphDB&&) noexcept = default;

  std::size_t dim() const noexcept { return dim_; }
  const std::string& path() const noexcept { return db_path_; }

  // Adds a single vector (no edge created).
  void add_vector(const std::string& id, const std::vector<float>& vec);

  // Adds a batch of vectors (no edges created).
  // ids.size() must equal vecs.size().
  void add_vectors_batch(const std::vector<std::string>& ids,
                         const std::vector<std::vector<float>>& vecs);

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

  // Reserve HNSW capacity to avoid repeated resizes.
  void reserve(std::size_t max_elements);

  // Set HNSW search ef (higher = more accurate, slower).
  void set_hnsw_ef_search(std::size_t ef_search);

  DbStats get_stats() const;

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
  std::unique_ptr<Impl> impl_;

  // Thread safety: reader-writer lock (or simple mutex on older compilers)
  mutable ReaderWriterLock mutex_;

  std::uint32_t add_vector_internal(const std::string& id, const std::vector<float>& vec);
  static float pearson_corr(const std::vector<float>& x, const std::vector<float>& y);

  void load_or_init();
  void ensure_dirs() const;
};

} // namespace vecgraphdb
