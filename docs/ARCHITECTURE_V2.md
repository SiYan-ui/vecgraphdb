# VecGraphDB Architecture v2

> Modular C++ core + Python bindings for a local, shareable **vector + graph** database. v2 emphasizes clear module boundaries, future‑proof persistence, and minimal memory usage while preserving fast ANN search.

## 1) Goals & Non‑Goals

**Goals**
- Single‑directory DB that can be shared/moved (no server).
- Vector store + ANN index + correlation graph edges.
- Fast query, small memory footprint (vectors on disk, mmap‑friendly).
- C++ core with stable Python bindings.
- Future‑proof persistence with versioning and migrations.

**Non‑Goals (v2)**
- Distributed or multi‑node.
- Fully transactional semantics.
- In‑place updates/deletes (supported later via tombstones + compaction).

---

## 2) Top‑Level Module Breakdown

```
core/          # Facade API, lifecycle, orchestration
storage/       # Durable data layout: manifest, vectors, ids, edges
index/         # ANN implementations + wrappers (HNSW today)
graph/         # In‑memory graph + serialization
io/            # Low‑level file I/O, mmap, endian helpers
utils/         # Small utilities (status, logging, error types)
bindings/      # Python bindings (pybind11)
include/       # Public C++ headers (stable API)
```

### Module Responsibilities & Interfaces

#### core/
- **`DB` facade**: orchestrates storage, index, graph, and query flows.
- **Lifecycle**: `open()`, `close()`, `flush()`, `compact()`.
- **Concurrency policy**: shared locks for reads, exclusive for writes.
- **Interfaces**
  - `core/db.h` – public class `VecGraphDB` (C++ API).
  - `core/context.h` – configuration: dim, space, index params.
  - `core/query.h` – query types & result structures.

#### storage/
- **File layout + versioning**: manifest, file headers, schema versions.
- **Vector store**: append‑only float32 (future: fp16/int8), optional mmap read.
- **ID store**: append‑only string list + in‑memory lookup map.
- **Graph store**: snapshot adjacency list (future: edge WAL + compaction).
- **Interfaces**
  - `storage/manifest.h` – read/write `MANIFEST.json`.
  - `storage/vector_store.h` – append + random read.
  - `storage/id_store.h` – append + lookup.
  - `storage/edge_store.h` – load/save adjacency snapshot.

#### index/
- **Index abstraction**: `IAnnIndex` with `build`, `add`, `search`.
- **HNSW wrapper**: isolates hnswlib dependency.
- **Interfaces**
  - `index/ann_index.h` – base interface.
  - `index/hnsw_index.h` – HNSW implementation.

#### graph/
- **In‑memory adjacency** with optional top‑K pruning.
- **Correlation utilities**: Pearson, clamp, etc.
- **Interfaces**
  - `graph/adjacency.h` – adjacency list container.
  - `graph/correlation.h` – correlation helpers.

#### io/
- **Binary headers**, endian helpers, mmap wrapper.
- **Interfaces**
  - `io/binary_io.h`, `io/mmap.h`.

#### bindings/
- **pybind11** wrappers, thin conversion layer.
- **Interfaces**
  - `bindings/py_vecgraphdb.cpp`.

---

## 3) Data Layout & Persistence (v2)

### Directory Layout (DB Root)
```
<db>/
  data/
    MANIFEST.json
    ids.txt
    vectors.f32
    edges.bin
    index.hnsw
```

### Manifest (versioned)
`data/MANIFEST.json`
```json
{
  "version": 2,
  "dim": 768,
  "space": "cosine",
  "created": "2026-03-01T...Z",
  "index": { "type": "hnsw", "M": 16, "ef_construction": 200, "ef_search": 50 }
}
```

### Binary Headers (future‑proof)
Each binary file has a fixed header:
```
magic[4] | version(u32) | dim(u32) | count(u64) | reserved[32]
```
- `VGDV` (vectors), `VGDE` (edges), `VGDH` (index wrapper).
- Allows forward‑compatible checks and graceful rebuilds.

### Ingestion & Indexing Flow
1. Validate + normalize vectors in batch.
2. Append vectors + ids in storage.
3. Add to ANN index.
4. Update adjacency lists for graph edges.

### Query Flow
- `topk_similar`: normalize query → ANN search.
- `topk_correlated`: ANN anchor → adjacency lookup → top‑K corr.

---

## 4) Public API Surface (C++ / Python)

### C++ API (stable)
```
VecGraphDB(db_path, dim, space="cosine")
add_vector(id, vec)
add_pair(id_a, a, id_b, b, corr=None)
add_vectors_batch(ids, matrix)
add_pairs_batch(pairs, matrix_a, matrix_b, corr=None)

Top‑K queries:
  topk_similar(query, k)
  topk_correlated(query, k)
  topk_correlated_by_id(id, k)

Maintenance:
  flush()
  compact()
  stats()
```

### Python Bindings
- `numpy.float32` inputs, list of ids.
- Mirrors C++ API exactly.

---

## 5) Build & Dependency Layout

**CMake targets**
- `vecgraphdb_core` (core, storage, graph, index, io, utils)
- `vecgraphdb_python` (bindings)

**Third‑party**
- `hnswlib` isolated under `index/`.
- No direct dependency leakage into `core/` or `storage/`.

---

## 6) Proposed Repository Structure
```
vecgraphdb/
  include/
    vecgraphdb/vecgraphdb.hpp
    vecgraphdb/types.hpp
  core/
    db.cpp
    db.h
    context.h
    query.h
  storage/
    manifest.cpp
    manifest.h
    vector_store.cpp
    vector_store.h
    id_store.cpp
    id_store.h
    edge_store.cpp
    edge_store.h
  index/
    ann_index.h
    hnsw_index.cpp
    hnsw_index.h
  graph/
    adjacency.cpp
    adjacency.h
    correlation.cpp
    correlation.h
  io/
    binary_io.h
    mmap.h
  utils/
    errors.h
    logging.h
  bindings/
    py_vecgraphdb.cpp
  tests/
    core_test.cpp
    storage_test.cpp
    index_test.cpp
    graph_test.cpp
  docs/
    ARCHITECTURE.md
    ARCHITECTURE_V2.md
```

---

## 7) Migration Strategy (2–4 PRs, Low Risk)

### PR1 — **Introduce Modular Skeleton** (no behavior change)
- Move `src/vecgraphdb.cpp` into `core/db.cpp`.
- Split small headers: `types.hpp` (result structs), `errors.h`.
- Keep storage/index logic inline but behind new namespaces.
- Add CMake targets `vecgraphdb_core` and update includes.
- **Tests:** existing tests should still pass.

### PR2 — **Extract Storage + Graph**
- Create `storage/` for `ids`, `vectors`, `edges`.
- Create `graph/adjacency` + `correlation` helpers.
- Add `MANIFEST.json` read/write (version 2).
- Keep file formats backward‑compatible; on missing manifest, assume legacy.
- **Tests:** add storage + graph unit tests.

### PR3 — **Extract ANN Index**
- Wrap hnswlib under `index/hnsw_index` and `IAnnIndex` interface.
- Make `core/db` depend on interface only.
- Add rebuild logic if index snapshot incompatible.
- **Tests:** index wrapper tests + integration test.

### PR4 — **Performance & Compatibility Enhancements**
- Optional: mmap vector reads.
- Optional: batch ingest APIs.
- Optional: background flush.
- **Tests:** performance smoke tests, version migration test (legacy db → v2).

---

## 8) Rewrite vs Refactor Recommendations

**Refactor (keep logic, reorganize)**
- `VecGraphDB` core flows (add, query, flush).
- HNSW usage (wrap, don’t rewrite).
- Correlation logic (Pearson, clamp).
- ID management (move into `storage/id_store`).

**Rewrite (clean, limited scope)**
- Persistence layer: introduce manifest + headers + version checks.
- Edge serialization: isolate to `storage/edge_store` with explicit format.
- Binary IO helpers (to avoid ad‑hoc read/write helpers).

---

## 9) Backward Compatibility & Versioning

- **Legacy layout** (no manifest) is treated as v1.
- On `open()`:
  1. Read manifest if present.
  2. Validate `dim/space`.
  3. If index snapshot incompatible, rebuild from vectors.
- Future schema changes increment `version` and add migration utilities.

---

## 10) Why This Modularization Works

- **Low‑risk**: keeps core behavior intact while relocating code.
- **Isolates dependencies**: ANN index swapped without touching storage or API.
- **Future‑proof**: versioned manifest + headers enable evolution.
- **Performance‑friendly**: enables mmap and batch ingestion cleanly.

---

## 11) Implementation Notes (Pragmatic)

- Keep `VecGraphDB` public header stable.
- Do not change data formats in PR1–PR2; add compatibility only.
- Add tests early for storage and index boundaries.
- Avoid premature optimization; focus on clean module interfaces.

---

## Appendix: Minimal Interface Sketches

```cpp
// index/ann_index.h
class IAnnIndex {
public:
  virtual ~IAnnIndex() = default;
  virtual void reserve(size_t max_elements) = 0;
  virtual void add(const float* vec, uint32_t id) = 0;
  virtual std::vector<std::pair<uint32_t, float>> search(const float* q, size_t k) const = 0;
  virtual void save(const std::string& path) const = 0;
  virtual void load(const std::string& path, size_t max_elements) = 0;
};
```

```cpp
// storage/vector_store.h
class VectorStore {
public:
  VectorStore(const fs::path& dir, size_t dim);
  uint32_t append(const std::vector<float>& vec);
  std::vector<float> read(uint32_t idx) const;
  size_t count() const;
};
```

```cpp
// graph/adjacency.h
class Adjacency {
public:
  void add_edge(uint32_t a, uint32_t b, float corr);
  std::vector<CorrResult> topk(uint32_t anchor, size_t k) const;
};
```
```
