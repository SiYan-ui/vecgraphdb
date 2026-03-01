# VecGraphDB Architecture

> Lightweight **vector + graph** local database focused on fast ANN search, small memory footprint, and simple persistence with C++ + Python APIs.

## Goals & Constraints
- **Vector search**: fast top‑k similarity (cosine) using HNSW.
- **Graph**: store correlation edges between vector pairs; query by anchor.
- **Local & shareable**: single directory per DB; no server required.
- **Small memory**: in‑memory index + minimal metadata; vectors on disk.
- **Persistence**: append‑only data files + periodic snapshot for index/edges.
- **APIs**: C++ core + Python bindings with consistent semantics.

## Current Implementation (MVP)
- **Vector store**: `data/vectors.f32` append‑only float32 matrix (L2‑normalized).
- **IDs**: `data/ids.txt` append‑only text, one id per line.
- **Graph edges**: `data/edges.bin` adjacency list (snapshot on `flush()`).
- **ANN index**: `data/index.hnsw` HNSWlib snapshot (inner product on normalized vectors).
- **ID policy**: duplicate ids rejected (throws). Existing data with dupes keeps first occurrence as canonical.

## Data Model
- **Internal id**: `uint32_t` index into vector store (0..N‑1).
- **External id**: user‑provided string (`ids.txt` line order = internal id).
- **Vector**: float32[dim], stored normalized so `dot(x,y) == cosine`.
- **Edge**: undirected (a,b,corr) with corr ∈ (‑1,1). Stored in adjacency lists.

## File Layout (Directory = DB Root)
```
<db>/
  data/
    ids.txt          # append‑only string ids
    vectors.f32      # append‑only float32 matrix, row-major
    edges.bin        # snapshot adjacency list
    index.hnsw       # HNSWlib snapshot
```

### Recommended File Format Versioning
To support evolution without breaking existing DBs, add a **manifest** and **headers**:

**Manifest**: `data/MANIFEST.json`
```json
{ "version": 1, "dim": 768, "space": "cosine", "created": "2026-03-01T...Z" }
```

**Binary headers** (for `vectors.f32`, `edges.bin`, `index.hnsw` wrapper):
- `magic` (4 bytes) e.g. `VGDV`, `VGDE`, `VGDH`
- `version` (u32)
- `dim` (u32)
- `count` (u64 or u32)
- `reserved` (future use)

This keeps backwards compatibility while enabling future options (quantization, mmap, compression, delete tombstones).

## Core Components

### 1) Vector Store
- **Append‑only**: write normalized vectors on insert.
- **Load**: by reading at offset `idx * dim * sizeof(float)`.
- **Future**: optional `mmap` for read‑heavy workloads to avoid repeated file open/seek.

### 2) ID Mapping
- In‑memory `std::vector<std::string>` for id list + `unordered_map` for lookup.
- **Future**: optional on‑disk hash (FST) or prefix tree for huge corpora.

### 3) Graph Store
- `adj_[i] = list of (neighbor_id, corr)` stored in memory.
- **Persistence**: `edges.bin` snapshot built on `flush()`.
- **Future**: append‑only edge log + compaction to avoid full rewrite.

### 4) ANN Index (HNSW)
- Inner‑product HNSW over normalized vectors (cosine similarity).
- **Load**: if index missing or empty, rebuild from vectors.
- **Config**: `M`, `ef_construction`, `ef_search`, `max_elements`.

## API Surface (Current)
C++ `VecGraphDB`:
- `add_vector(id, vec)`
- `add_pair(id_a, a, id_b, b, corr?)`
- `topk_similar(query, k)`
- `topk_correlated(query, k)`
- `topk_correlated_by_id(id, k)`
- `flush()`

Python mirrors C++ with `np.ndarray(float32)` inputs.

## Recommended API Extensions
**Batch ingest** (training‑loop friendly):
- `add_vectors_batch(ids, matrix)` → returns assigned internal ids
- `add_pairs_batch(pairs, matrix_a, matrix_b, corr?)`

**Index controls**:
- `reserve(max_elements)`
- `set_hnsw_params(M, ef_construction, ef_search)`

**Stats**:
- `get_stats()` → {count, dim, file_sizes, index_params}

**Accessors**:
- `has_id(id)`
- `get_vector(id|internal)` (optional; useful for debugging)

**Maintenance**:
- `compact()` (rebuild edges/index, optionally re‑write vectors)

## Concurrency & Performance Plan (Training‑Loop Use)

### Ingestion
1. **Batch insert**: accept `N x dim` matrix; normalize in batch (SIMD / vectorized).
2. **Threaded pipeline**:
   - Thread A: validate + normalize
   - Thread B: append vectors/ids
   - Thread C: HNSW addPoint (can be parallel with locks or chunked)
3. **Locking model**:
   - Write lock for `ids_`, `id_to_internal_`, `adj_`, and vector file append.
   - Separate lock for HNSW (hnswlib is not fully thread‑safe; guard `addPoint` & `resizeIndex`).
   - Read operations (`topk_similar`, `topk_correlated`) can be shared‑locked.

### Query
- `topk_similar` is read‑only; allow concurrent reads.
- `topk_correlated` reads adj list + HNSW; safe under shared lock.
- Provide `set_ef_search(k)` to control latency vs recall.

### Persistence
- `flush()` acquires write lock; snapshot index + edges.
- Optional **background flush**: spawn thread to serialize index/edges after a stable point.

### Memory
- Keep vectors on disk; only load on demand.
- Use mmap for vector reads at scale.
- Avoid storing duplicated edge lists by storing only top‑K per node (optional cap).

## Compatibility Strategy
- **Versioned files** + **manifest** prevent silent corruption.
- Loader should:
  1. Read manifest (if missing, assume v0 legacy layout).
  2. Validate dim/space.
  3. Gracefully rebuild index if version mismatch.

## Future Improvements
- Disk‑ANN (DiskANN / IVF‑PQ) for huge corpora.
- Delete & update with tombstones + compaction.
- Weighted graph edges & multi‑relation support.
- Quantized vector storage (fp16 / int8) for memory reduction.

## Migration Notes
- **Legacy v0 (no MANIFEST/header)**: treat as “version 0” and infer `dim` from first vector file size / ids count. If inference fails, require user‑provided `dim`.
- **On upgrade to v1**:
  - Create `data/MANIFEST.json` with detected `dim`, `space=cosine`, and `version=1`.
  - Wrap or rewrite `vectors.f32`/`edges.bin`/`index.hnsw` with headers (or store headers in sidecar `.hdr` files if in‑place rewrite is risky).
- **Index mismatch**: if `index.hnsw` version or `dim` doesn’t match manifest, rebuild index from `vectors.f32`.
- **Edge format change**: if future edge schema changes, add a converter that reads old adjacency snapshots and emits the new format during `compact()`.
- **Rollback strategy**: keep backups of pre‑migration files (`*.bak`) until first successful `flush()` completes.
