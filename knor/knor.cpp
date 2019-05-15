#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "cknor/libkcommon/kcommon.hpp"
#include "cknor/libauto/kmeans.hpp"
#include "cknor/libauto/kmeans.hpp"
#include "knor/cknor/binding/knori.hpp"

namespace py = pybind11;
namespace kbase = kpmeans::base;

class Kmeans {
    public:
        kbase::kmeans_t fit(double* data, const size_t nrow,
                const size_t ncol, const unsigned k,
                size_t max_iters=std::numeric_limits<size_t>::max(),
                unsigned nnodes=kbase::get_num_nodes(),
                unsigned nthread=kbase::get_num_omp_threads(),
                double* p_centers=NULL, std::string init="kmeanspp",
                double tolerance=-1, std::string dist_type="eucl",
                bool omp=false, bool numa_opt=false) {
            return kbase::kmeans(data, nrow, ncol, k, max_iters,
                    nnodes, nthread, p_centers, init,
                    tolerance, dist_type, omp, numa_opt);
        }

        kbase::kmeans_t fit(const std::string datafn, const size_t nrow,
                const size_t ncol, const unsigned k,
                size_t max_iters=std::numeric_limits<size_t>::max(),
                unsigned nnodes=kbase::get_num_nodes(),
                unsigned nthread=kbase::get_num_omp_threads(),
                double* p_centers=NULL, std::string init="kmeanspp",
                double tolerance=-1, std::string dist_type="eucl",
                bool omp=false) {

            return kbase::kmeans(datafn, nrow, ncol, k, max_iters, nnodes,
                    nthread, p_centers, init, tolerance, dist_type, omp);
        }
};

PYBIND11_MODULE(knor, m) {
    m.doc() = R"pbdoc(
        Python knor API
        ---------------------------

        .. currentmodule:: knor

        .. autosummary::
           :toctree: _generate

    )pbdoc";

    // GMap
    py::class_<kbase::kmeans_t>(m, "kmeans_t")
            .def(py::init(), "Create a kmeans_t return object")
            .def("__repr__", &kbase::kmeans_t::to_str);

    py::class_<Kmeans>(m, "Kmeans")
            .def(py::init(), "Create a Kmeans object")
            .def("fit", (kbase::kmeans_t (Kmeans::*)(double*, const size_t,
                            const size_t, const unsigned, size_t, unsigned,
                            unsigned, double*, std::string, double, std::string,
                            bool, bool)) &Kmeans::fit,
                    "Compute kmeans on the dataset provided")
            .def("fit", (kbase::kmeans_t (Kmeans::*)(const std::string,
                            const size_t, const size_t, const unsigned,
                            size_t, unsigned, unsigned, double*, std::string,
                            double, std::string, bool)) &Kmeans::fit,
                    "Compute kmeans on the dataset provided");

    // Versioning information
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
