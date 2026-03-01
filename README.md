# vecgraphdb

A lightweight **vector + graph** local database:

- Stores vectors on disk
- Builds an ANN index for fast **top-k similarity search**
- Maintains a **graph edge** between inserted vector pairs with a correlation coefficient in **(-1, 1)**
- Query:
  - `topk_similar(query_vec)` → most similar vectors
  - `topk_correlated(query_vec)` → uses the *nearest stored vector* (anchor) and returns its highest-correlation neighbors
  - `topk_correlated_by_id(id)` → anchor explicitly by an existing stored vector id

## Design (MVP)

- Similarity search: **HNSW** (cosine) via vendored `hnswlib` (C++).
- Persistence:
  - `data/vectors.f32` raw float32 matrix (append-only)
  - `data/ids.txt` external ids in insertion order
  - `data/edges.bin` adjacency list (append-only, rebuilt on compact)
  - `data/index.hnsw` HNSW index snapshot

> Notes:
> - For maximum throughput we keep an in-memory HNSW graph + a compact on-disk vector store.
> - For very large corpora, you’d likely swap in a disk-ANN engine (DiskANN/IVF-PQ/etc). This repo keeps dependencies minimal.

## Build (C++)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

## Python install

Requires Python 3.10+.

```bash
python -m pip install -U pip
pip install -e .

python -c "import vecgraphdb; print(vecgraphdb.__version__)"
```

## Quick usage

```python
import numpy as np
from vecgraphdb import VecGraphDB

db = VecGraphDB('mydb', dim=3)

db.add_pair('a', np.array([1,2,3], np.float32),
            'b', np.array([1,2,2.9], np.float32))

print(db.topk_similar(np.array([1,2,3], np.float32), k=5))
print(db.topk_correlated(np.array([1,2,3], np.float32), k=5))
print(db.topk_correlated_by_id('a', k=5))

# Batch insert
ids = ["c", "d"]
mat = np.array([[0, 0, 1], [1, 0, 0]], np.float32)
db.add_vectors_batch(ids, mat)

# Stats
print(db.get_stats())
```

## Roadmap

- Dedup / update / delete
- mmap vector store
- edge compaction + background rebuild
- multi-threaded batch insert/search
- optional cosine vs L2
