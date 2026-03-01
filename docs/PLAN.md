# vecgraphdb — Implementation Plan (quality-first)

## Current status
- C++ core implemented (vector store + HNSW ANN + correlation edges)
- Python bindings via pybind11
- Persistence: ids.txt, vectors.f32 (float32), edges.bin, index.hnsw
- ID uniqueness enforced; duplicates rejected
- Added `topk_correlated_by_id(id, k)`

Repo: https://github.com/SiYan-ui/vecgraphdb

## Decisions (owned by 鹿野)
- **Duplicate IDs**: rejected (throw) to keep semantics stable.
- **Similarity**: cosine similarity via L2-normalized vectors stored on disk; Inner Product index.
- **Storage**: append-only files for fast ingest; periodic snapshot via `flush()`.

## Next milestones

### M1 — API completeness for training loops
- [ ] Add C++ batch insert API: `add_vectors_batch(ids, matrix)`
- [ ] Add Python batch insert: accept `np.ndarray(float32)[N,dim]`
- [ ] Add `reserve(max_elements)` / configuration knobs for HNSW (`M`, `ef_construction`, `ef_search`)
- [ ] Add `get_stats()` (count, dim, file sizes)

### M2 — Performance + memory
- [ ] Optional mmap read path for vectors (avoid repeated file opens)
- [ ] Thread-safety notes + optional locks for concurrent read/write
- [ ] Fast edge lookup structure & compaction strategy (future)

### M3 — Usability
- [ ] CLI tool (optional) for import/query
- [ ] Better docs + examples
- [ ] Packaging: build wheels, CI (GitHub Actions)

## Risks / TODOs
- Current persistence does not compact edges; adjacency is fully rewritten on `flush()`.
- HNSW load/rebuild behavior needs careful versioning when file formats evolve.
