#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <cstring>

#include "vecgraphdb/vecgraphdb.hpp"

namespace py = pybind11;
using vecgraphdb::VecGraphDB;

// Optimized conversion: avoid copy when possible
static std::vector<float> np_to_vec(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
  if (arr.ndim() != 1) throw std::invalid_argument("expected 1-D float32 array");
  std::vector<float> v(arr.shape(0));
  std::memcpy(v.data(), arr.data(), sizeof(float) * v.size());
  return v;
}

// Optimized batch conversion: direct memory access without intermediate vectors
static void add_vectors_batch_optimized(VecGraphDB& self, 
                                         const std::vector<std::string>& ids,
                                         py::array_t<float, py::array::c_style> mat) {
  if (mat.ndim() != 2) throw std::invalid_argument("expected 2-D float32 array");
  
  auto dim = self.dim();
  if (static_cast<std::size_t>(mat.shape(1)) != dim) {
    throw std::invalid_argument("matrix dimension mismatch: expected " + 
                                std::to_string(dim) + ", got " + 
                                std::to_string(mat.shape(1)));
  }
  
  std::size_t n = static_cast<std::size_t>(mat.shape(0));
  if (n != ids.size()) {
    throw std::invalid_argument("ids and matrix must have same length");
  }
  
  // Access raw data directly
  const float* data = static_cast<const float*>(mat.data());
  
  // Build vectors in the required format
  std::vector<std::vector<float>> vecs;
  vecs.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    vecs.emplace_back(data + i * dim, data + (i + 1) * dim);
  }
  
  self.add_vectors_batch(ids, vecs);
}

PYBIND11_MODULE(_vecgraphdb, m) {
  m.doc() = "vecgraphdb bindings";

  m.attr("__version__") = VECGRAPHDB_VERSION;

  py::class_<VecGraphDB>(m, "VecGraphDB")
    .def(py::init<std::string, std::size_t, std::string, std::size_t, std::size_t, std::size_t>(),
         py::arg("db_path"), py::arg("dim"), py::arg("space") = "cosine",
         py::arg("hnsw_M") = 16, py::arg("hnsw_ef_construction") = 200,
         py::arg("hnsw_ef_search") = 100)
    // Expose dim() and path() accessors
    .def_property_readonly("dim", &VecGraphDB::dim, "Vector dimension")
    .def_property_readonly("path", &VecGraphDB::path, "Database path")
    // Vector insertion APIs
    .def("add_vector", [](VecGraphDB& self, const std::string& id, py::array_t<float> v){
        self.add_vector(id, np_to_vec(v));
    }, py::arg("id"), py::arg("vec"), "Add a single vector")
    .def("add_vectors_batch", &add_vectors_batch_optimized,
         py::arg("ids"), py::arg("mat"), "Add multiple vectors in batch")
    .def("add_pair", [](VecGraphDB& self,
                        const std::string& ida, py::array_t<float> a,
                        const std::string& idb, py::array_t<float> b,
                        py::object corr){
        if (corr.is_none()) {
          self.add_pair(ida, np_to_vec(a), idb, np_to_vec(b), std::nullopt);
        } else {
          self.add_pair(ida, np_to_vec(a), idb, np_to_vec(b), corr.cast<float>());
        }
    }, py::arg("id_a"), py::arg("a"), py::arg("id_b"), py::arg("b"), 
       py::arg("corr") = py::none(), "Add two vectors with an edge")
    // Query APIs
    .def("topk_similar", [](const VecGraphDB& self, py::array_t<float> q, std::size_t k) {
        auto out = self.topk_similar(np_to_vec(q), k);
        py::list lst;
        for (auto& r : out) lst.append(py::make_tuple(r.id, r.score));
        return lst;
    }, py::arg("query"), py::arg("k"), "Find top-k most similar vectors")
    .def("topk_correlated", [](const VecGraphDB& self, py::array_t<float> q, std::size_t k) {
        auto out = self.topk_correlated(np_to_vec(q), k);
        py::list lst;
        for (auto& r : out) lst.append(py::make_tuple(r.id, r.corr));
        return lst;
    }, py::arg("query"), py::arg("k"), "Find top-k correlated vectors using anchor")
    .def("topk_correlated_by_id", [](const VecGraphDB& self, const std::string& id, std::size_t k) {
        auto out = self.topk_correlated_by_id(id, k);
        py::list lst;
        for (auto& r : out) lst.append(py::make_tuple(r.id, r.corr));
        return lst;
    }, py::arg("id"), py::arg("k"), "Find top-k correlated vectors by ID")
    // Configuration APIs
    .def("reserve", &VecGraphDB::reserve, py::arg("max_elements"), 
         "Reserve HNSW capacity")
    .def("set_hnsw_ef_search", &VecGraphDB::set_hnsw_ef_search, py::arg("ef_search"),
         "Set HNSW search ef parameter")
    // Stats and persistence
    .def("get_stats", [](const VecGraphDB& self) {
        auto s = self.get_stats();
        py::dict d;
        d["count"] = s.count;
        d["dim"] = s.dim;
        d["vectors_bytes"] = s.vectors_bytes;
        d["ids_bytes"] = s.ids_bytes;
        d["edges_bytes"] = s.edges_bytes;
        d["index_bytes"] = s.index_bytes;
        return d;
    }, "Get database statistics")
    .def("flush", &VecGraphDB::flush, "Persist index and edges to disk");
}
