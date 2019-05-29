#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "cknor/libkcommon/kcommon.hpp"
#include "cknor/binding/kmeanspp.hpp"
#include "cknor/libman/kmeans_task_coordinator.hpp"
//#include "cknor/libman/kmeans_coordinator.hpp"
//#include "cknor/libman/skmeans_coordinator.hpp"
//#include "cknor/libman/fcm_coordinator.hpp"
//#include "cknor/libman/medoid_coordinator.hpp"
//#include "cknor/libman/hclust_coordinator.hpp"
//#include "cknor/libman/xmeans_coordinator.hpp"
//#include "cknor/libman/gmeans_coordinator.hpp"

namespace py = pybind11;
namespace kbase = knor::base;
namespace kprune = knor::prune;

class Kmeans {
    public:
        kbase::cluster_t fit(double* data, const size_t nrow,
                const size_t ncol, const unsigned k,
                size_t max_iters=std::numeric_limits<size_t>::max(),
                unsigned nnodes=kbase::get_num_nodes(),
                unsigned nthread=kbase::get_num_omp_threads(),
                double* p_centers=NULL, std::string init="kmeanspp",
                double tolerance=-1, std::string dist_type="eucl") {

            return kprune::kmeans_task_coordinator::create(
                    "", nrow, ncol, k, max_iters, nnodes,
                    nthread, p_centers, init, tolerance,
                    dist_type)->run(&data[0]);
        }

        kbase::cluster_t fit(const std::string datafn, const size_t nrow,
                const size_t ncol, const unsigned k,
                size_t max_iters=std::numeric_limits<size_t>::max(),
                unsigned nnodes=kbase::get_num_nodes(),
                unsigned nthread=kbase::get_num_omp_threads(),
                double* p_centers=NULL, std::string init="kmeanspp",
                double tolerance=-1, std::string dist_type="eucl") {

            return kprune::kmeans_task_coordinator::create(
                    datafn, nrow, ncol, k, max_iters, nnodes,
                    nthread, p_centers, init, tolerance, dist_type)->run();
        }
};

class KmeansPP {
    public:
        kbase::pp_pair fit(
                double* data, const size_t nrow,
                const size_t ncol, const unsigned k,
                const unsigned nstart=1,
                unsigned nthread=kbase::get_num_omp_threads(),
                std::string dist_type="eucl") {

            return kbase::kmeansPP(data, nrow, ncol, k, nstart, nthread,
                    dist_type);
        }

        kbase::pp_pair fit(
                const std::string datafn, const size_t nrow,
                const size_t ncol, const unsigned k,
                const unsigned nstart=1,
                unsigned nthread=kbase::get_num_omp_threads(),
                std::string dist_type="eucl") {

            return kbase::kmeansPP(datafn, nrow, ncol, k,
                    nstart, nthread, dist_type);
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

    // Kmeans
    py::class_<kbase::cluster_t>(m, "cluster_t")
            .def(py::init(), "Create a cluster_t return object")
            .def("__repr__", &kbase::cluster_t::to_str);

    py::class_<Kmeans>(m, "Kmeans")
            .def(py::init(), "Create a Kmeans object")
            .def("fit", (kbase::cluster_t (Kmeans::*)(double*, const size_t,
                            const size_t, const unsigned, size_t, unsigned,
                            unsigned, double*, std::string, double, std::string
                            )) &Kmeans::fit,
                    "Compute kmeans on the dataset provided")
            .def("fit", (kbase::cluster_t (Kmeans::*)(const std::string,
                            const size_t, const size_t, const unsigned,
                            size_t, unsigned, unsigned, double*, std::string,
                            double, std::string)) &Kmeans::fit,
                    "Compute kmeans on the dataset provided");

    // Kmeans++
    py::class_<KmeansPP>(m, "KmeansPP")
            .def(py::init(), "Create a KmeansPP object")
            .def("fit", (kbase::pp_pair (KmeansPP::*)(double*, const size_t,
                            const size_t, const unsigned, const unsigned,
                            unsigned, std::string)) &KmeansPP::fit,
                    "Compute kmeans++ on the dataset provided")
            .def("fit", (kbase::pp_pair (KmeansPP::*)(
                            const std::string, const size_t, const size_t,
                            const unsigned, const unsigned,
                unsigned nthread, std::string)) &KmeansPP::fit,
                    "Compute kmeans++ on the dataset provided");

    // Versioning information
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
