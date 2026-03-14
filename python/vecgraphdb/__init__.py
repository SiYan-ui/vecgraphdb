"""VecGraphDB - A lightweight vector + graph local database.

This package provides a hybrid database that combines:
- Vector storage with ANN (Approximate Nearest Neighbor) search via HNSW
- Graph edges between vectors with correlation coefficients

Basic usage:
    from vecgraphdb import VecGraphDB
    import numpy as np
    
    db = VecGraphDB('mydb', dim=768)
    db.add_vector('doc1', np.array([0.1, 0.2, ...], dtype=np.float32))
    
    # Similarity search
    results = db.topk_similar(query_vector, k=10)
    
    # Correlation-based query
    results = db.topk_correlated(query_vector, k=10)
"""

from ._vecgraphdb import VecGraphDB, __version__

__all__ = ["VecGraphDB", "__version__"]

