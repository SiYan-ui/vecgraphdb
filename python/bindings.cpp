#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <cstring>

#include "vecgraphdb/vecgraphdb.hpp"

namespace py = pybind11;
using vecgraphdb::VecGraphDB;

static std::vector<float> np_to_vec(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
  if (arr.ndim() != 1) throw std::invalid_argument("expected 1-D float32 array");
  std::vector<float> v(arr.shape(0));
  std::memcpy(v.data(), arr.data(), sizeof(float) * v.size());
  return v;
}

static std::vector<std::vector<float>> np_to_mat(
    py::array_t<float, py::array::c_style | py::array::forcecast> arr,
    std::size_t dim) {
  if (arr.ndim() != 2) throw std::invalid_argument("expected 2-D float32 array");
  if (static_cast<std::size_t>(arr.shape(1)) != dim) {
    throw std::invalid_argument("matrix dimension mismatch");
  }
  std::size_t n = static_cast<std::size_t>(arr.shape(0));
  const float* data = static_cast<const float*>(arr.data());
  std::vector<std::vector<float>> out;
  out.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    out.emplace_back(data + i * dim, data + (i + 1) * dim);
  }
  return out;
}

PYBIND11_MODULE(_vecgraphdb, m) {
  m.doc() = "vecgraphdb bindings";

  m.attr("__version__") = VECGRAPHDB_VERSION;

  py::class_<VecGraphDB>(m, "VecGraphDB")
    .def(py::init<std::string, std::size_t, std::string, std::size_t, std::size_t, std::size_t>(),
         py::arg("db_path"), py::arg("dim"), py::arg("space") = "cosine",
         py::arg("hnsw_M") = 16, py::arg("hnsw_ef_construction") = 200,
         py::arg("hnsw_ef_search") = 100)
    .def("add_vector", [](VecGraphDB& self, const std::string& id, py::array_t<float> v){
        self.add_vector(id, np_to_vec(v));
    })
    .def("add_vectors_batch", [](VecGraphDB& self, const std::vector<std::string>& ids, py::array_t<float> mat){
        self.add_vectors_batch(ids, np_to_mat(mat, self.dim()));
    })
    .def("add_pair", [](VecGraphDB& self,
                        const std::string& ida, py::array_t<float> a,
                        const std::string& idb, py::array_t<float> b,
                        py::object corr){
        if (corr.is_none()) {
          self.add_pair(ida, np_to_vec(a), idb, np_to_vec(b), std::nullopt);
        } else {
          self.add_pair(ida, np_to_vec(a), idb, np_to_vec(b), corr.cast<float>());
        }
    }, py::arg("id_a"), py::arg("a"), py::arg("id_b"), py::arg("b"), py::arg("corr") = py::none())
    .def("topk_similar", [](const VecGraphDB& self, py::array_t<float> q, std::size_t k) {
        auto out = self.topk_similar(np_to_vec(q), k);
        py::list lst;
        for (auto& r : out) lst.append(py::make_tuple(r.id, r.score));
        return lst;
    })
    .def("topk_correlated", [](const VecGraphDB& self, py::array_t<float> q, std::size_t k) {
        auto out = self.topk_correlated(np_to_vec(q), k);
        py::list lst;
        for (auto& r : out) lst.append(py::make_tuple(r.id, r.corr));
        return lst;
    })
    .def("topk_correlated_by_id", [](const VecGraphDB& self, const std::string& id, std::size_t k) {
        auto out = self.topk_correlated_by_id(id, k);
        py::list lst;
        for (auto& r : out) lst.append(py::make_tuple(r.id, r.corr));
        return lst;
    })
    .def("reserve", &VecGraphDB::reserve)
    .def("set_hnsw_ef_search", &VecGraphDB::set_hnsw_ef_search)
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
    })
    .def("flush", &VecGraphDB::flush);
}
