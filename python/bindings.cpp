#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "vecgraphdb/vecgraphdb.hpp"

namespace py = pybind11;
using vecgraphdb::VecGraphDB;

static std::vector<float> np_to_vec(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
  if (arr.ndim() != 1) throw std::invalid_argument("expected 1-D float32 array");
  std::vector<float> v(arr.shape(0));
  std::memcpy(v.data(), arr.data(), sizeof(float) * v.size());
  return v;
}

PYBIND11_MODULE(_vecgraphdb, m) {
  m.doc() = "vecgraphdb bindings";

  m.attr("__version__") = VECGRAPHDB_VERSION;

  py::class_<VecGraphDB>(m, "VecGraphDB")
    .def(py::init<std::string, std::size_t, std::string>(),
         py::arg("db_path"), py::arg("dim"), py::arg("space") = "cosine")
    .def("add_vector", [](VecGraphDB& self, const std::string& id, py::array_t<float> v){
        self.add_vector(id, np_to_vec(v));
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
    .def("flush", &VecGraphDB::flush);
}
