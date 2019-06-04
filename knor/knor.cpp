#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "cknor/libkcommon/kcommon.hpp"
#include "cknor/binding/kmeanspp.hpp"
#include "cknor/libman/kmeans_task_coordinator.hpp"
#include "cknor/libman/skmeans_coordinator.hpp"
#include "cknor/libman/fcm_coordinator.hpp"
#include "cknor/libman/medoid_coordinator.hpp"
#include "cknor/libman/hclust_coordinator.hpp"
#include "cknor/libman/xmeans_coordinator.hpp"
#include "cknor/libman/gmeans_coordinator.hpp"

namespace py = pybind11;
namespace kbase = knor::base;
namespace kprune = knor::prune;

class Kmeans {
    private:
        const unsigned k;
        size_t max_iters;
        unsigned nthread;
        std::vector<double> centers;
        std::string init;
        double tolerance;
        std::string dist_type;

    public:
        Kmeans(const unsigned k, size_t max_iters, unsigned nthread,
                std::vector<double>& centers, std::string& init,
                double tolerance, std::string& dist_type) : k(k),
        max_iters(max_iters), nthread(nthread), centers(centers),
        init(init), tolerance(tolerance), dist_type(dist_type) {

        }

        Kmeans(const unsigned k, size_t max_iters, unsigned nthread,
                py::buffer centers, std::string& init,
                double tolerance, std::string& dist_type) : k(k),
        max_iters(max_iters), nthread(nthread),
        init(init), tolerance(tolerance), dist_type(dist_type){
            py::buffer_info info = centers.request();

            /* Some sanity checks ... */
            if (info.format != "d")
                throw std::runtime_error(
                        "Incompatible centers format: expected a double array!");

            if (info.ndim != 2)
                throw std::runtime_error("Incompatible centers dimension!");

            this->centers.resize(info.shape[0]*info.shape[1]);
            std::copy(this->centers.begin(), this->centers.end(),
                    static_cast<double*>(info.ptr));
        }

        inline kbase::cluster_t fit(std::vector<double>& data, const size_t nrow,
                const size_t ncol) {

            return kprune::kmeans_task_coordinator::create(
                    "", nrow, ncol, k, max_iters, kbase::get_num_nodes(),
                    nthread, (centers.size() ? &centers[0]:NULL), init,
                    tolerance, dist_type)->run(&data[0]);
        }

        inline kbase::cluster_t fit(const std::string& datafn, const size_t nrow,
                const size_t ncol) {

            return kprune::kmeans_task_coordinator::create(
                    datafn, nrow, ncol, k, max_iters, kbase::get_num_nodes(),
                    nthread, (centers.size() ? &centers[0]:NULL), init,
                    tolerance, dist_type)->run();
        }

        inline kbase::cluster_t fit(py::buffer buf) {
            /* Request a buffer descriptor from Python */
            py::buffer_info info = buf.request();

            /* Some sanity checks ... */
            if (info.format != "d")
                throw std::runtime_error(
                        "Incompatible format: expected a double array!");

            if (info.ndim != 2)
                throw std::runtime_error("Incompatible buffer dimension!");

            double* data = static_cast<double*>(info.ptr);

            return kprune::kmeans_task_coordinator::create(
                    "", info.shape[0], info.shape[1], k, max_iters,
                    kbase::get_num_nodes(),
                    nthread, (centers.size() ? &centers[0]:NULL), init,
                    tolerance, dist_type)->run(&data[0]);
        }
};

class KmeansPP {
    private:
        const unsigned k;
        const unsigned nstart;
        unsigned nthread;
        std::string dist_type;

    public:
        KmeansPP(const unsigned k, const unsigned nstart,
                unsigned nthread, std::string& dist_type) :
            k(k), nstart(nstart), nthread(nthread), dist_type(dist_type) {
            }

        inline kbase::pp_pair fit(std::vector<double>& data, const size_t nrow,
                const size_t ncol) {

            return kbase::kmeansPP(&data[0], nrow, ncol, k, nstart, nthread,
                    dist_type);
        }

        inline kbase::pp_pair fit(const std::string& datafn, const size_t nrow,
                const size_t ncol) {

            return kbase::kmeansPP(datafn, nrow, ncol, k,
                    nstart, nthread, dist_type);
        }

        inline kbase::pp_pair fit(py::buffer buf) {
            /* Request a buffer descriptor from Python */
            py::buffer_info info = buf.request();

            /* Some sanity checks ... */
            if (info.format != "d")
                throw std::runtime_error(
                        "Incompatible format: expected a double array!");

            if (info.ndim != 2)
                throw std::runtime_error("Incompatible buffer dimension!");

            double* data = static_cast<double*>(info.ptr);

            return kbase::kmeansPP(&data[0], info.shape[0], info.shape[1], k,
                    nstart, nthread, dist_type);
        }
};

class SKmeans {
    private:
        const unsigned k;
        size_t max_iters;
        unsigned nthread;
        std::vector<double> centers;
        std::string init;
        double tolerance;

    public:
        SKmeans(const unsigned k, size_t max_iters, unsigned nthread,
                std::vector<double>& centers, std::string& init,
                double tolerance) : k(k), max_iters(max_iters),
        nthread(nthread), init(init), tolerance(tolerance) {
        }

        SKmeans(const unsigned k, size_t max_iters, unsigned nthread,
                py::buffer centers, std::string& init, double tolerance):
            k(k), max_iters(max_iters), nthread(nthread),
            init(init), tolerance(tolerance) {

            py::buffer_info info = centers.request();

            /* Some sanity checks ... */
            if (info.format != "d")
                throw std::runtime_error(
                        "Incompatible centers format: expected a double array!");

            if (info.ndim != 2)
                throw std::runtime_error("Incompatible centers dimension!");

            this->centers.resize(info.shape[0]*info.shape[1]);
            std::copy(this->centers.begin(), this->centers.end(),
                    static_cast<double*>(info.ptr));
        }

        inline kbase::cluster_t fit(std::vector<double>& data,
                const size_t nrow, const size_t ncol) {

            return knor::skmeans_coordinator::create(
                    "", nrow, ncol, k, max_iters, kbase::get_num_nodes(),
                    nthread, (centers.size() ? &centers[0]:NULL), init,
                    tolerance, "cos")->run(&data[0]);
        }

        inline kbase::cluster_t fit(const std::string& datafn, const size_t nrow,
                const size_t ncol) {

            return knor::skmeans_coordinator::create(
                    datafn, nrow, ncol, k, max_iters, kbase::get_num_nodes(),
                    nthread, (centers.size() ? &centers[0]:NULL),
                    init, tolerance, "cos")->run();
        }

        inline kbase::cluster_t fit(py::buffer buf) {
            /* Request a buffer descriptor from Python */
            py::buffer_info info = buf.request();

            /* Some sanity checks ... */
            if (info.format != "d")
                throw std::runtime_error(
                        "Incompatible format: expected a double array!");

            if (info.ndim != 2)
                throw std::runtime_error("Incompatible buffer dimension!");

            double* data = static_cast<double*>(info.ptr);

            return knor::skmeans_coordinator::create(
                    "", info.shape[0], info.shape[1], k, max_iters,
                    kbase::get_num_nodes(),
                    nthread, (centers.size() ? &centers[0]:NULL), init,
                    tolerance, "cos")->run(&data[0]);
        }
};

class FuzzyCMeans {
    private:
        const unsigned k;
        const size_t max_iters;
        const unsigned nthread;
        std::vector<double> centers;
        std::string init;
        const double tolerance;
        const std::string dist_type;
        const unsigned fuzzindex;

    public:
        FuzzyCMeans(const unsigned k, const unsigned max_iters,
                const unsigned nthread, std::vector<double>& centers,
                const std::string& init, const double tolerance,
                const std::string& dist_type, const unsigned fuzzindex) :
            k(k), max_iters(max_iters), nthread(nthread), centers(centers),
            init(init), tolerance(tolerance), dist_type(dist_type),
            fuzzindex(fuzzindex) {

            }

        FuzzyCMeans(const unsigned k, const unsigned max_iters,
                const unsigned nthread, py::buffer centers,
                const std::string& init, const double tolerance,
                const std::string& dist_type, const unsigned fuzzindex) :
            k(k), max_iters(max_iters), nthread(nthread),
            init(init), tolerance(tolerance), dist_type(dist_type),
            fuzzindex(fuzzindex) {

            py::buffer_info info = centers.request();

            /* Some sanity checks ... */
            if (info.format != "d")
                throw std::runtime_error(
                        "Incompatible centers format: expected a double array!");

            if (info.ndim != 2)
                throw std::runtime_error("Incompatible centers dimension!");

            this->centers.resize(info.shape[0]*info.shape[1]);
            std::copy(this->centers.begin(), this->centers.end(),
                    static_cast<double*>(info.ptr));
        }

        inline kbase::cluster_t fit(std::vector<double>& data,
                const size_t nrow, const size_t ncol) {

            return knor::fcm_coordinator::create(
                    "", nrow, ncol, k, max_iters, kbase::get_num_nodes(),
                    nthread, (centers.size() ? &centers[0]:NULL), init,
                    tolerance, dist_type, fuzzindex)->run(&data[0]);
        }

        inline kbase::cluster_t fit(const std::string& datafn, const size_t nrow,
                const size_t ncol) {

            return knor::fcm_coordinator::create(
                    datafn, nrow, ncol, k, max_iters, kbase::get_num_nodes(),
                    nthread, (centers.size() ? &centers[0]:NULL),
                    init, tolerance, dist_type, fuzzindex)->run();
        }

        inline kbase::cluster_t fit(py::buffer buf) {
            /* Request a buffer descriptor from Python */
            py::buffer_info info = buf.request();

            /* Some sanity checks ... */
            if (info.format != "d")
                throw std::runtime_error(
                        "Incompatible format: expected a double array!");

            if (info.ndim != 2)
                throw std::runtime_error("Incompatible buffer dimension!");

            double* data = static_cast<double*>(info.ptr);

            return knor::fcm_coordinator::create(
                    "", info.shape[0], info.shape[1], k, max_iters,
                    kbase::get_num_nodes(),
                    nthread, (centers.size() ? &centers[0]:NULL), init,
                    tolerance, dist_type, fuzzindex)->run(&data[0]);
        }
};

class Kmedoids {
    private:
        const unsigned k;
        const size_t max_iters;
        const unsigned nthread;
        std::vector<double> centers;
        const std::string init;
        const double tolerance;
        const std::string dist_type;
        const unsigned sample_rate;

    public:
        Kmedoids(const unsigned k, const unsigned max_iters,
                const unsigned nthread, std::vector<double>& centers,
                const std::string& init, const double tolerance,
                const std::string& dist_type, const double sample_rate):
            k(k), max_iters(max_iters), nthread(nthread), centers(centers),
            init(init), tolerance(tolerance), dist_type(dist_type),
            sample_rate(sample_rate) {
            }

        Kmedoids(const unsigned k, const unsigned max_iters,
                const unsigned nthread, py::buffer centers,
                const std::string& init, const double tolerance,
                const std::string& dist_type, const double sample_rate):
            k(k), max_iters(max_iters), nthread(nthread),
            init(init), tolerance(tolerance), dist_type(dist_type),
            sample_rate(sample_rate) {

            py::buffer_info info = centers.request();

            /* Some sanity checks ... */
            if (info.format != "d")
                throw std::runtime_error(
                        "Incompatible centers format: expected a double array!");

            if (info.ndim != 2)
                throw std::runtime_error("Incompatible centers dimension!");

            this->centers.resize(info.shape[0]*info.shape[1]);
            std::copy(this->centers.begin(), this->centers.end(),
                    static_cast<double*>(info.ptr));
        }

        inline kbase::cluster_t fit(std::vector<double>& data, const size_t nrow,
                const size_t ncol) {

            return knor::medoid_coordinator::create("",
                    nrow, ncol, k, max_iters,
                    kbase::get_num_nodes(), nthread,
                    (centers.size() ? &centers[0]:NULL),
                    init, tolerance, dist_type, sample_rate)->run(&data[0]);
        }

        inline kbase::cluster_t fit(const std::string& datafn, const size_t nrow,
                const size_t ncol) {

            auto coord =
                knor::medoid_coordinator::create("",
                        nrow, ncol, k, max_iters, kbase::get_num_nodes(),
                        nthread, (centers.size() ? &centers[0]:NULL),
                        init, tolerance, dist_type, sample_rate);

            std::vector<double> data(nrow*ncol);
            kbase::bin_io<double> br(datafn, nrow, ncol);
            br.read(&data);
            return coord->run(&data[0]);
            // TODO: if (centers) delete [] centers;
            // TODO: delete data
        }

        inline kbase::cluster_t fit(py::buffer buf) {
            /* Request a buffer descriptor from Python */
            py::buffer_info info = buf.request();

            /* Some sanity checks ... */
            if (info.format != "d")
                throw std::runtime_error(
                        "Incompatible format: expected a double array!");

            if (info.ndim != 2)
                throw std::runtime_error("Incompatible buffer dimension!");

            double* data = static_cast<double*>(info.ptr);

            return knor::medoid_coordinator::create("",
                    info.shape[0], info.shape[1], k, max_iters,
                    kbase::get_num_nodes(), nthread,
                    (centers.size() ? &centers[0]:NULL),
                    init, tolerance, dist_type, sample_rate)->run(&data[0]);
        }
};

class Hmeans {
    private:
        const unsigned kmax;
        const size_t max_iters;
        const unsigned nthread;
        std::vector<double> centers;
        const std::string init;
        const double tolerance;
        const std::string dist_type;
        const unsigned min_clust_size;

    public:
        Hmeans(const unsigned kmax, const unsigned max_iters,
                const unsigned nthread,
                std::vector<double>& centers, const std::string& init,
                const double tolerance, const std::string& dist_type,
                const unsigned min_clust_size) :
            kmax(kmax), max_iters(max_iters), nthread(nthread), centers(centers),
            init(init), tolerance(tolerance), dist_type(dist_type),
            min_clust_size(min_clust_size)  {
            }


        Hmeans(const unsigned kmax, const unsigned max_iters,
                const unsigned nthread, py::buffer centers,
                const std::string& init, const double tolerance,
                const std::string& dist_type, const unsigned min_clust_size) :
            kmax(kmax), max_iters(max_iters), nthread(nthread),
            init(init), tolerance(tolerance), dist_type(dist_type),
            min_clust_size(min_clust_size)  {

            py::buffer_info info = centers.request();

            /* Some sanity checks ... */
            if (info.format != "d")
                throw std::runtime_error(
                        "Incompatible centers format: expected a double array!");

            if (info.ndim != 2)
                throw std::runtime_error("Incompatible centers dimension!");

            this->centers.resize(info.shape[0]*info.shape[1]);
            std::copy(this->centers.begin(), this->centers.end(),
                    static_cast<double*>(info.ptr));
        }

        inline kbase::cluster_t fit(std::vector<double>& data,
                const size_t nrow, const size_t ncol) {

            return knor::hclust_coordinator::create("", nrow, ncol, kmax,
                    max_iters, kbase::get_num_nodes(), nthread,
                    (centers.size() ? &centers[0]:NULL),
                    init, tolerance, dist_type, min_clust_size)->run(&data[0]);
        }

        inline kbase::cluster_t fit(const std::string& datafn,
                const size_t nrow, const size_t ncol) {

            return knor::hclust_coordinator::create(datafn, nrow, ncol, kmax,
                    max_iters, kbase::get_num_nodes(), nthread,
                    (centers.size() ? &centers[0]:NULL),
                    init, tolerance, dist_type, min_clust_size)->run();
        }

        inline kbase::cluster_t fit(py::buffer buf) {
            /* Request a buffer descriptor from Python */
            py::buffer_info info = buf.request();

            /* Some sanity checks ... */
            if (info.format != "d")
                throw std::runtime_error(
                        "Incompatible format: expected a double array!");

            if (info.ndim != 2)
                throw std::runtime_error("Incompatible buffer dimension!");

            double* data = static_cast<double*>(info.ptr);

            return knor::hclust_coordinator::create("", info.shape[0],
                    info.shape[1], kmax,
                    max_iters, kbase::get_num_nodes(), nthread,
                    (centers.size() ? &centers[0]:NULL),
                    init, tolerance, dist_type, min_clust_size)->run(&data[0]);
        }
};

class  Xmeans {
    private:
        const unsigned kmax;
        const size_t max_iters;
        const unsigned nthread;
        std::vector<double> centers;
        const std::string init;
        const double tolerance;
        const std::string dist_type;
        const unsigned min_clust_size;

    public:
        Xmeans(const unsigned kmax, const unsigned max_iters,
                const unsigned nthread,
                std::vector<double>& centers, const std::string& init,
                const double tolerance, const std::string& dist_type,
                const unsigned min_clust_size) :
            kmax(kmax), max_iters(max_iters), nthread(nthread), centers(centers),
            init(init), tolerance(tolerance), dist_type(dist_type),
            min_clust_size(min_clust_size) {

            }

        Xmeans(const unsigned kmax, const unsigned max_iters,
                const unsigned nthread, py::buffer centers,
                const std::string& init, const double tolerance,
                const std::string& dist_type, const unsigned min_clust_size) :
            kmax(kmax), max_iters(max_iters), nthread(nthread),
            init(init), tolerance(tolerance), dist_type(dist_type),
            min_clust_size(min_clust_size)  {

            py::buffer_info info = centers.request();

            /* Some sanity checks ... */
            if (info.format != "d")
                throw std::runtime_error(
                        "Incompatible centers format: expected a double array!");

            if (info.ndim != 2)
                throw std::runtime_error("Incompatible centers dimension!");

            this->centers.resize(info.shape[0]*info.shape[1]);
            std::copy(this->centers.begin(), this->centers.end(),
                    static_cast<double*>(info.ptr));
        }

        inline kbase::cluster_t fit(std::vector<double>& data, const size_t nrow,
                const size_t ncol) {

            return knor::xmeans_coordinator::create("", nrow, ncol, kmax,
                    max_iters, kbase::get_num_nodes(),
                    nthread, (centers.size() ? &centers[0]:NULL),
                    init, tolerance, dist_type,
                    min_clust_size)->run(&data[0]);

        }

        inline kbase::cluster_t fit(const std::string& datafn, const size_t nrow,
                const size_t ncol) {

            return knor::xmeans_coordinator::create(datafn, nrow, ncol, kmax,
                    max_iters, kbase::get_num_nodes(),
                    nthread, (centers.size() ? &centers[0]:NULL), init,
                    tolerance, dist_type, min_clust_size)->run();
        }

        inline kbase::cluster_t fit(py::buffer buf) {
            /* Request a buffer descriptor from Python */
            py::buffer_info info = buf.request();

            /* Some sanity checks ... */
            if (info.format != "d")
                throw std::runtime_error(
                        "Incompatible format: expected a double array!");

            if (info.ndim != 2)
                throw std::runtime_error("Incompatible buffer dimension!");

            double* data = static_cast<double*>(info.ptr);

            return knor::xmeans_coordinator::create("", info.shape[0],
                    info.shape[1], kmax,
                    max_iters, kbase::get_num_nodes(),
                    nthread, (centers.size() ? &centers[0]:NULL),
                    init, tolerance, dist_type,
                    min_clust_size)->run(&data[0]);
        }
};

class Gmeans {
    private:
        const unsigned kmax;
        const size_t max_iters;
        const unsigned nthread;
        std::vector<double> centers;
        const std::string init;
        const double tolerance;
        const std::string dist_type;
        const unsigned min_clust_size;
        const short strictness;

    public:
        Gmeans(const unsigned kmax, const unsigned max_iters,
                const unsigned nthread,
                std::vector<double>& centers, const std::string& init,
                const double tolerance, const std::string& dist_type,
                const unsigned min_clust_size, const short strictness) :
            kmax(kmax), max_iters(max_iters), nthread(nthread), centers(centers),
            init(init), tolerance(tolerance), dist_type(dist_type),
            min_clust_size(min_clust_size), strictness(strictness) {

            }

        Gmeans(const unsigned kmax, const unsigned max_iters,
                const unsigned nthread, py::buffer centers,
                const std::string& init, const double tolerance,
                const std::string& dist_type, const unsigned min_clust_size,
                const short strictness) :
            kmax(kmax), max_iters(max_iters), nthread(nthread),
            init(init), tolerance(tolerance), dist_type(dist_type),
            min_clust_size(min_clust_size), strictness(strictness) {

            py::buffer_info info = centers.request();

            /* Some sanity checks ... */
            if (info.format != "d")
                throw std::runtime_error(
                        "Incompatible centers format: expected a double array!");

            if (info.ndim != 2)
                throw std::runtime_error("Incompatible centers dimension!");

            this->centers.resize(info.shape[0]*info.shape[1]);
            std::copy(this->centers.begin(), this->centers.end(),
                    static_cast<double*>(info.ptr));
        }

        inline kbase::cluster_t fit(std::vector<double>& data, const size_t nrow,
                const size_t ncol) {

            return knor::gmeans_coordinator::create("", nrow, ncol, kmax,
                    max_iters, kbase::get_num_nodes(),
                    nthread, (centers.size() ? &centers[0]:NULL), init,
                    tolerance, dist_type, min_clust_size,
                    strictness)->run(&data[0]);
        }

        inline kbase::cluster_t fit(const std::string& datafn, const size_t nrow,
                const size_t ncol) {

            return knor::gmeans_coordinator::create(datafn, nrow, ncol, kmax,
                    max_iters, kbase::get_num_nodes(),
                    nthread, (centers.size() ? &centers[0]:NULL), init,
                    tolerance, dist_type,
                    min_clust_size, strictness)->run();
        }

        inline kbase::cluster_t fit(py::buffer buf) {
            /* Request a buffer descriptor from Python */
            py::buffer_info info = buf.request();

            /* Some sanity checks ... */
            if (info.format != "d")
                throw std::runtime_error(
                        "Incompatible format: expected a double array!");

            if (info.ndim != 2)
                throw std::runtime_error("Incompatible buffer dimension!");

            double* data = static_cast<double*>(info.ptr);

            return knor::gmeans_coordinator::create("", info.shape[0],
                    info.shape[1], kmax,
                    max_iters, kbase::get_num_nodes(),
                    nthread, (centers.size() ? &centers[0]:NULL), init,
                    tolerance, dist_type, min_clust_size,
                    strictness)->run(&data[0]);
        }
};

PYBIND11_MODULE(knor, m) {
    m.doc() = R"pbdoc(
        Python knor API
        ---------------------------

        .. currentmodule:: knor

        .. autosummary::
           :toctree: _generate

           cluster_t
           Kmeans
           KmeansPP
           SKmeans
           FuzzyCMeans
           Kmedoids
           Hmeans
           Xmeans
           Gmeans
    )pbdoc";

    ////////////////////////// cluster_t //////////////////////////

    py::class_<kbase::cluster_t>(m, "cluster_t")
            .def(py::init(), "Create a cluster_t return object")
            .def_readonly("k", &kbase::cluster_t::k)
            .def_readonly("nrow", &kbase::cluster_t::nrow)
            .def_readonly("ncol", &kbase::cluster_t::ncol)
            .def_readonly("sizes", &kbase::cluster_t::assignment_count)
            .def_readonly("iters", &kbase::cluster_t::iters)
            .def_readonly("centroids", &kbase::cluster_t::centroids)
            .def_readonly("cluster", &kbase::cluster_t::assignments)
            .def("__repr__", &kbase::cluster_t::to_str)
            .def("__eq__", [](const kbase::cluster_t& ob1,
                        const kbase::cluster_t& ob2) -> bool {
                    return ob1 == ob2;
            }, "Equality operator for cluster_t object")
            .def("__ne__", [](const kbase::cluster_t& ob1,
                        const kbase::cluster_t& ob2) -> bool {
                    return !(ob1 == ob2);
            }, "Not Equal operator for cluster_t object");

    ////////////////////////// Kmeans //////////////////////////

    py::class_<Kmeans>(m, "Kmeans")
            .def(py::init<const unsigned, size_t,
                unsigned, std::vector<double>&, std::string&,
                double, std::string&>(),
                R"pbdoc(
    K-means provides *k* disjoint sets for a dataset using a
    parallel and fast NUMA optimized version of Lloyd's algorithm.
    The details of which are found in this paper
    https://arxiv.org/pdf/1606.08905.pdf.

    Positional arguments:
    ---------------------
    k:
        - The maximum number of clusters

    Optional arguments:
    -------------------
    max_iters:
        - Maximum number of iterations to perform
    nthread:
        - The number of parallel threads to run
    centers:
        - Initialized centroids
    init:
        -  The type of initialization to use "kmeanspp",
        "random" or "forgy"
    tolerance:
        - The convergence tolerance
    dist_type: What dissimilarity metric to use: "eucl", "cos",
        "taxi", "sqeucl"

    Example:
    --------
    import numpy as np
    n = 100; m = 10; k = 5
    data = np.random.random((n, m))
    km = knor.Kmeans(k)
    ret = km.fit(data)
       )pbdoc",
            py::arg("k"), py::arg("max_iters")=20, py::arg("nthread")=2,
            py::arg("centers")=std::vector<double>(),
            py::arg("init")="kmeanspp", py::arg("tolerance")=-1,
            py::arg("dist_type")="eucl"
            )
            .def(py::init<const unsigned, size_t, unsigned, py::buffer,
                    std::string&, double, std::string&>(),
            py::arg("k"), py::arg("max_iters")=20, py::arg("nthread")=2,
            py::arg("centers")=NULL,
            py::arg("init")="kmeanspp", py::arg("tolerance")=-1,
            py::arg("dist_type")="eucl")
            .def("fit", (kbase::cluster_t (Kmeans::*)(std::vector<double>&,
                            const size_t, const size_t)) &Kmeans::fit,
                R"pbdoc(
    Run the k-means algorithm on the dataset

    Positional arguments:
    ---------------------
    data:
        - List or numpy.ndarray
    nrow:
        - The number of samples in the dataset
    ncol:
        - The number of features in the dataset
       )pbdoc",
            py::arg("data"), py::arg("nrow"), py::arg("ncol")
            )
            .def("fit", (kbase::cluster_t (Kmeans::*)(const std::string&,
                            const size_t, const size_t)) &Kmeans::fit,
                R"pbdoc(
    Run the k-means algorithm on the dataset

    Positional arguments:
    ---------------------
    datafn:
        - File name of data in raw row-major binary format
    nrow:
        - The number of samples in the dataset
    ncol:
        - The number of features in the dataset
       )pbdoc",
                    py::arg("datafn"), py::arg("nrow"), py::arg("ncol")
                )
            .def("fit", (kbase::cluster_t (Kmeans::*)(py::buffer)) &Kmeans::fit,
                R"pbdoc(
    Run the k-means algorithm on the dataset

    Positional arguments:
    ---------------------
    data:
        - 2D numpy.ndarray
       )pbdoc",
            py::arg("data")
            );

    ////////////////////////// Kmeans++ //////////////////////////

    py::class_<KmeansPP>(m, "KmeansPP")
            .def(py::init<const unsigned, const unsigned,
                    unsigned, std::string&>(),
                    R"pbdoc(
    K-means++ provides *k* disjoint sets for a dataset using a
    parallel and scalable implementation of the algorithm described
    in Ostrovsky, Rafail, et al. "The effectiveness of Lloyd-type
    methods for the k-means problem." Journal of the ACM (JACM) 59.6 (2012): 28.

    Positional arguments:
    ---------------------
    k:
        - The maximum number of clusters

    Optional arguments:
    -------------------
    nstart:
        -  The number of iterations of kmeans++ to run
    nthread:
        - The number of parallel threads to run
    dist_type: What dissimilarity metric to use: "eucl", "cos",
        "taxi", "sqeucl"

    Example:
    --------
    import numpy as np
    n = 100; m = 10; k = 5
    data = np.random.random((n, m))
    km = knor.KmeansPP(k, nstart=5)
    ret = km.fit(data, n, m)
       )pbdoc",
            py::arg("k"), py::arg("nstart")=1, py::arg("nthread")=2,
            py::arg("dist_type")="eucl")
            .def("fit", (kbase::pp_pair (KmeansPP::*)(std::vector<double>&,
                            const size_t, const size_t)) &KmeansPP::fit,
    R"pbdoc(
    Run the K-means++ algorithm on a dataset

    Positional arguments:
    ---------------------
    data:
        - List or flattened 1D numpy.ndarray
    nrow:
        - The number of samples in the dataset
    ncol:
        - The number of features in the dataset
       )pbdoc",
            py::arg("data"), py::arg("nrow"), py::arg("ncol"))
            .def("fit", (kbase::pp_pair (KmeansPP::*)(
                const std::string&, const size_t, const size_t)) &KmeansPP::fit,
    R"pbdoc(
    Run the K-means++ algorithm on a dataset

    Positional arguments:
    ---------------------
    datafn:
        - File name of data in raw row-major binary format
    nrow:
        - The number of samples in the dataset
    ncol:
        - The number of features in the dataset
       )pbdoc",
                    py::arg("datafn"), py::arg("nrow"), py::arg("ncol")
                )
            .def("fit", (kbase::pp_pair (KmeansPP::*)(py::buffer)) &KmeansPP::fit,
                R"pbdoc(
    Run the K-means++ algorithm on the dataset

    Positional arguments:
    ---------------------
    data:
        - 2D numpy.ndarray
       )pbdoc",
            py::arg("data")
            );

    //////////////////////////  SKmeans //////////////////////////

    py::class_<SKmeans>(m, "SKmeans")
            .def(py::init<const unsigned, size_t, unsigned,
                std::vector<double>&, std::string&, double>(),
   R"pbdoc(
   Perform spherical k-means clustering on a data matrix. Similar to the
   k-means algorithm differing only in that data features are min-max
   normalized the dissimilarity metric is Cosine distance.

    Positional arguments:
    ---------------------
    k:
        - The maximum number of clusters

    Optional arguments:
    -------------------
    max_iters:
        - Maximum number of iterations to perform
    nthread:
        - The number of parallel threads to run
    centers:
        - Initialized centroids
    init:
        -  The type of initialization to use "kmeanspp",
        "random" or "forgy"
    tolerance:
        - The convergence tolerance

    Example:
    --------
    import numpy as np
    n = 100; m = 10; k = 5
    data = np.random.random((n, m))
    km = knor.SKmeans(k)
    ret = km.fit(data)
   )pbdoc",
            py::arg("k"), py::arg("max_iters")=20, py::arg("nthread")=2,
        py::arg("centers")=std::vector<double>(),
        py::arg("init")="kmeanspp", py::arg("tolerance")=-1)

            .def(py::init<const unsigned, size_t, unsigned,
                py::buffer, std::string&, double>(),
            py::arg("k"), py::arg("max_iters")=20, py::arg("nthread")=2,
            py::arg("centers")=NULL,
            py::arg("init")="kmeanspp", py::arg("tolerance")=-1)
            .def("fit", (kbase::cluster_t (SKmeans::*)(std::vector<double>&,
                            const size_t, const size_t)) &SKmeans::fit,
    R"pbdoc(
    Run the spherical k-means algorithm on the dataset

    Positional arguments:
    ---------------------
    data:
        - List or flattened 1D numpy.ndarray
    nrow:
        - The number of samples in the dataset
    ncol:
        - The number of features in the dataset
       )pbdoc",
                    py::arg("data"), py::arg("nrow"), py::arg("ncol")
                )
            .def("fit", (kbase::cluster_t (SKmeans::*)(const std::string&,
                            const size_t, const size_t)) &SKmeans::fit,
    R"pbdoc(
    Run the spherical K-means algorithm on a dataset

    Positional arguments:
    ---------------------
    datafn:
        - File name of data in raw row-major binary format
    nrow:
        - The number of samples in the dataset
    ncol:
        - The number of features in the dataset
       )pbdoc",
                    py::arg("datafn"), py::arg("nrow"), py::arg("ncol")
                )
            .def("fit", (kbase::cluster_t (SKmeans::*)(py::buffer)) &SKmeans::fit,
    R"pbdoc(
    Run the spherical k-means algorithm on the dataset

    Positional arguments:
    ---------------------
    data:
        - 2D numpy.ndarray
       )pbdoc",
                    py::arg("data")
                );

    //////////////////////////  FuzzyCMeans //////////////////////////

    py::class_<FuzzyCMeans>(m, "FuzzyCMeans")
            .def(py::init<const unsigned, const unsigned,
                    const unsigned, std::vector<double>&,
                    const std::string&, const double, const std::string&,
                    const unsigned>(),
    R"pbdoc(
    Perform Fuzzy C-means clustering on a data matrix. A soft variant of
    the kmeans algorithm where each data point are assigned a contribution
    weight to each cluster


    Positional arguments:
    ---------------------
    k:
        - The maximum number of clusters

    Optional arguments:
    -------------------
    max_iters:
        - Maximum number of iterations to perform
    nthread:
        - The number of parallel threads to run
    centers:
        - Initialized centroids
    init:
        -  The type of initialization to use "kmeanspp",
        "random" or "forgy"
    tolerance:
        - The convergence tolerance
    dist_type: What dissimilarity metric to use: "eucl", "cos",
        "taxi", "sqeucl"
    fuzzindex: Randomization paramerter `fuzziness coefficient/index'
        (> 1 and < inf)

    Example:
    --------
    import numpy as np
    n = 100; m = 10; k = 5
    data = np.random.random((n, m))
    km = knor.FuzzyCMeans(k)
    ret = km.fit(data)
   )pbdoc",
                    py::arg("k"), py::arg("max_iters")=20, py::arg("nthread")=2,
                    py::arg("centers")=std::vector<double>(),
                    py::arg("init")="forgy", py::arg("tolerance")=-1,
                    py::arg("dist_type")="cos", py::arg("fuzzindex")=2
                        )
            .def(py::init<const unsigned, const unsigned,
                    const unsigned, py::buffer,
                    const std::string&, const double, const std::string&,
                    const unsigned>(),
                    py::arg("k"), py::arg("max_iters")=20, py::arg("nthread")=2,
                    py::arg("centers")=NULL,
                    py::arg("init")="forgy", py::arg("tolerance")=-1,
                    py::arg("dist_type")="cos", py::arg("fuzzindex")=2)
            .def("fit", (kbase::cluster_t (FuzzyCMeans::*)(std::vector<double>&,
                        const size_t, const size_t)) &FuzzyCMeans::fit,
    R"pbdoc(
    Compute FuzzyCMeans on a dataset

    Positional arguments:
    ---------------------
    data:
        - List or flattened 1D numpy.ndarray
    nrow:
        - The number of samples in the dataset
    ncol:
        - The number of features in the dataset
       )pbdoc",
                    py::arg("data"), py::arg("nrow"), py::arg("ncol")
                )
            .def("fit", (kbase::cluster_t (FuzzyCMeans::*)(const std::string&,
                            const size_t, const size_t)) &FuzzyCMeans::fit,
    R"pbdoc(
    Compute FuzzyCMeans on a dataset

    Positional arguments:
    ---------------------
    datafn:
        - File name of data in raw row-major binary format
    nrow:
        - The number of samples in the dataset
    ncol:
        - The number of features in the dataset
       )pbdoc",
                    py::arg("datafn"), py::arg("nrow"), py::arg("ncol")
                    )
            .def("fit", (kbase::cluster_t (FuzzyCMeans::*)
                        (py::buffer)) &FuzzyCMeans::fit,
    R"pbdoc(
    Compute FuzzyCMeans on a dataset

    Positional arguments:
    ---------------------
    data:
        - A 2D numpy.ndarray
       )pbdoc",
                    py::arg("data")
                );

    ////////////////////////// Kmedoids //////////////////////////

    py::class_<Kmedoids>(m, "Kmedoids")
            .def(py::init<const unsigned, const unsigned,
                    const unsigned, std::vector<double>&,
                    const std::string&,
                    const double, const std::string&,
                    const double>(),
    R"pbdoc(
	Perform k-medoids clustering on a data matrix.
	After initialization the k-medoids algorithm partitions data by testing which
	data member of a cluster Ci may make a better candidate as medoid (centroid)
	by reducing the sum of distance (usually taxi), then running a reclustering
	step with updated medoids


    Positional arguments:
    ---------------------
    k:
        - The maximum number of clusters

    Optional arguments:
    -------------------
    max_iters:
        - Maximum number of iterations to perform
    nthread:
        - The number of parallel threads to run
    centers:
        - Initialized medoids
    init:
        -  The type of initialization to use "forgy" (only)
    tolerance:
        - The convergence tolerance
    dist_type: What dissimilarity metric to use: "eucl", "cos",
        "taxi", "sqeucl"
	sample_rate: Fraction of dataset to use for each iteration

    Example:
    --------
    import numpy as np
    n = 100; m = 10; k = 5
    data = np.random.random((n, m))
    km = knor.Kmedoids(k, sample_rate=.5)
    ret = km.fit(data)
   )pbdoc",

                    py::arg("k"), py::arg("max_iters")=20, py::arg("nthread")=2,
                    py::arg("centers")=std::vector<double>(),
                    py::arg("init")="forgy", py::arg("tolerance")=-1,
                    py::arg("dist_type")="taxi", py::arg("sample_rate")=.2)
            .def(py::init<const unsigned, const unsigned, const unsigned,
                    py::buffer, const std::string&, const double,
                    const std::string&, const double>(),
                    py::arg("k"), py::arg("max_iters")=20, py::arg("nthread")=2,
                    py::arg("centers")=NULL,
                    py::arg("init")="forgy", py::arg("tolerance")=-1,
                    py::arg("dist_type")="taxi", py::arg("sample_rate")=.2)
            .def("fit", (kbase::cluster_t (Kmedoids::*)(std::vector<double>&,
                            const size_t, const size_t)) &Kmedoids::fit,
    R"pbdoc(
    Compute CLARA on a dataset

    Positional arguments:
    ---------------------
    data:
        - List or flattened 1D numpy.ndarray
    nrow:
        - The number of samples in the dataset
    ncol:
        - The number of features in the dataset
       )pbdoc",
                    py::arg("data"), py::arg("nrow"), py::arg("ncol")
                )
            .def("fit", (kbase::cluster_t (Kmedoids::*)(const std::string&,
                            const size_t, const size_t)) &Kmedoids::fit,
    R"pbdoc(
    Compute CLARA on a dataset

    Positional arguments:
    ---------------------
    datafn:
        - File name of data in raw row-major binary format
    nrow:
        - The number of samples in the dataset
    ncol:
        - The number of features in the dataset
       )pbdoc",
            py::arg("datafn"), py::arg("nrow"), py::arg("ncol")
                    )
            .def("fit", (kbase::cluster_t (Kmedoids::*)
                        (py::buffer)) &Kmedoids::fit,
    R"pbdoc(
    Compute CLARA on a dataset

    Positional arguments:
    ---------------------
    data:
        - A 2D numpy.ndarray
       )pbdoc",
                    py::arg("data")
                    );

    ////////////////////////// Hmeans //////////////////////////

    py::class_<Hmeans>(m, "Hmeans")
            .def(py::init<const unsigned,
                    const unsigned, const unsigned,
                    std::vector<double>&, const std::string&,
                    const double, const std::string&,
                    const unsigned>(),
    R"pbdoc(
	A recursive (not actually implemented as recursion) partitioning
    of data into two disjoint sets at every level as described in
    https://en.wikipedia.org/wiki/Hierarchical_clustering


    Positional arguments:
    ---------------------
    kmax:
        - The maximum number of clusters

    Optional arguments:
    -------------------
    max_iters:
        - Maximum number of iterations to perform
    nthread:
        - The number of parallel threads to run
    centers:
        - Initialized centroids
    init:
        -  The type of initialization to use "forgy" (only)
    tolerance:
        - The convergence tolerance
    dist_type: What dissimilarity metric to use: "eucl", "cos",
        "taxi", "sqeucl"
    min_clust_size:
        - The minimum cluster size before no splitting is permitted

    Example:
    --------
    import numpy as np
    n = 100; m = 10; kmax = 5
    data = np.random.random((n, m))
    km = knor.Hmeans(kmax)
    ret = km.fit(data)
   )pbdoc",

                    py::arg("kmax"), py::arg("max_iters")=20,
                    py::arg("nthread")=2,
                    py::arg("centers")=std::vector<double>(),
                    py::arg("init")="forgy", py::arg("tolerance")=-1,
                    py::arg("dist_type")="eucl", py::arg("min_clust_size")=2
                            )
            .def(py::init<const unsigned,
                    const unsigned, const unsigned,
                    py::buffer, const std::string&,
                    const double, const std::string&,
                    const unsigned>(),
                    py::arg("kmax"), py::arg("max_iters")=20,
                    py::arg("nthread")=2, py::arg("centers")=NULL,
                    py::arg("init")="forgy", py::arg("tolerance")=-1,
                    py::arg("dist_type")="eucl", py::arg("min_clust_size")=2)
            .def("fit", (kbase::cluster_t (Hmeans::*)(std::vector<double>&,
                            const size_t, const size_t)) &Hmeans::fit,
    R"pbdoc(
    Compute Hierarchical clustering means on a dataset

    Positional arguments:
    ---------------------
    data:
        - List or flattened 1D numpy.ndarray
    nrow:
        - The number of samples in the dataset
    ncol:
        - The number of features in the dataset
       )pbdoc",
                    py::arg("data"), py::arg("nrow"), py::arg("ncol")
                )
            .def("fit", (kbase::cluster_t (Hmeans::*)(const std::string&,
                            const size_t, const size_t)) &Hmeans::fit,
    R"pbdoc(
    Compute Hierarchical clustering means on a dataset

    Positional arguments:
    ---------------------
    datafn:
        - File name of data in raw row-major binary format
    nrow:
        - The number of samples in the dataset
    ncol:
        - The number of features in the dataset
       )pbdoc",
            py::arg("datafn"), py::arg("nrow"), py::arg("ncol")
                )
            .def("fit", (kbase::cluster_t (Hmeans::*)(py::buffer)) &Hmeans::fit,
    R"pbdoc(
    Compute Hierarchical clustering means on a dataset

    Positional arguments:
    ---------------------
    data:
        - A 2D numpy.ndarray
       )pbdoc",
                    py::arg("data")
                );

    ////////////////////////// Xmeans //////////////////////////

    py::class_<Xmeans>(m, "Xmeans")
        .def(py::init<const unsigned, const unsigned,
                const unsigned, std::vector<double>&,
                const std::string&, const double, const std::string&,
                const unsigned>(),
    R"pbdoc(
    A recursive (not acutally implemented as recursion) partitioning
    of data into two disjoint sets at every level as described in:
    http://cs.uef.fi/~zhao/Courses/Clustering2012/Xmeans.pdf


    Positional arguments:
    ---------------------
    kmax:
        - The maximum number of clusters

    Optional arguments:
    -------------------
    max_iters:
        - Maximum number of iterations to perform
    nthread:
        - The number of parallel threads to run
    centers:
        - Initialized centroids
    init:
        -  The type of initialization to use "forgy" (only)
    tolerance:
        - The convergence tolerance
    dist_type: What dissimilarity metric to use: "eucl", "cos",
        "taxi", "sqeucl"
    min_clust_size:
        - The minimum cluster size before no splitting is permitted

    Example:
    --------
    import numpy as np
    n = 100; m = 10; kmax = 5
    data = np.random.random((n, m))
    km = knor.Xmeans(kmax)
    ret = km.fit(data)
   )pbdoc",
                py::arg("kmax"), py::arg("max_iters")=20, py::arg("nthread")=2,
                py::arg("centers")=std::vector<double>(),
                py::arg("init")="forgy", py::arg("tolerance")=-1,
                py::arg("dist_type")="eucl", py::arg("min_clust_size")=2)
        .def(py::init<const unsigned, const unsigned,
                const unsigned, py::buffer, const std::string&,
                const double, const std::string&, const unsigned>(),
                py::arg("kmax"), py::arg("max_iters")=20,
                py::arg("nthread")=2, py::arg("centers")=NULL,
                py::arg("init")="forgy", py::arg("tolerance")=-1,
                py::arg("dist_type")="eucl", py::arg("min_clust_size")=2)
            .def("fit", (kbase::cluster_t (Xmeans::*)(std::vector<double>&,
                            const size_t, const size_t)) &Xmeans::fit,
    R"pbdoc(
    Compute X-means clustering means on a dataset

    Positional arguments:
    ---------------------
    data:
        - List or flattened 1D numpy.ndarray
    nrow:
        - The number of samples in the dataset
    ncol:
        - The number of features in the dataset
       )pbdoc",
                    py::arg("data"), py::arg("nrow"), py::arg("ncol")
                )
            .def("fit", (kbase::cluster_t (Xmeans::*)(const std::string&,
                            const size_t, const size_t)) &Xmeans::fit,
    R"pbdoc(
    Compute X-means clustering means on a dataset

    Positional arguments:
    ---------------------
    datafn:
        - File name of data in raw row-major binary format
    nrow:
        - The number of samples in the dataset
    ncol:
        - The number of features in the dataset
       )pbdoc",
                    py::arg("datafn"), py::arg("nrow"), py::arg("ncol")
                    )
            .def("fit", (kbase::cluster_t (Xmeans::*)(py::buffer)) &Xmeans::fit,
    R"pbdoc(
    Compute X-means clustering means on a dataset

    Positional arguments:
    ---------------------
    data:
        - A 2D numpy.ndarray
       )pbdoc",
                    py::arg("data")
                );

    ////////////////////////// Gmeans //////////////////////////

    py::class_<Gmeans>(m, "Gmeans")
            .def(py::init<const unsigned, const unsigned, const unsigned,
                    std::vector<double>&, const std::string&,
                    const double, const std::string&,
                    const unsigned, const short>(),

    R"pbdoc(
    Perform a parallel hierarchical clustering using the g-means algorithm

    A hierarchical cluster algorithm that chooses the number of clusters based on
    the Anderson Darling statistic described in:
    http://papers.nips.cc/paper/2526-learning-the-k-in-k-means.pdf


    Positional arguments:
    ---------------------
    kmax:
        - The maximum number of clusters

    Optional arguments:
    -------------------
    max_iters:
        - Maximum number of iterations to perform
    nthread:
        - The number of parallel threads to run
    centers:
        - Initialized centroids
    init:
        -  The type of initialization to use "forgy" (only)
    tolerance:
        - The convergence tolerance
    dist_type: What dissimilarity metric to use: "eucl", "cos",
        "taxi", "sqeucl"
    min_clust_size:
        - The minimum cluster size before no splitting is permitted
    strictness: The Anderson-Darling strictness level. Between 1 and 4 inclusive

    Example:
    --------
    import numpy as np
    n = 100; m = 10; kmax = 5
    data = np.random.random((n, m))
    km = knor.Gmeans(kmax)
    ret = km.fit(data)
   )pbdoc",
                    py::arg("kmax"), py::arg("max_iters")=20,
                    py::arg("nthread")=2,
                    py::arg("centers")=std::vector<double>(),
                    py::arg("init")="forgy", py::arg("tolerance")=-1,
                    py::arg("dist_type")="eucl", py::arg("min_clust_size")=2,
                    py::arg("strictness")=4)
            .def(py::init<const unsigned, const unsigned, const unsigned,
                    py::buffer, const std::string&,
                    const double, const std::string&,
                    const unsigned, const short>(),
                    py::arg("kmax"), py::arg("max_iters")=20,
                    py::arg("nthread")=2, py::arg("centers")=NULL,
                    py::arg("init")="forgy", py::arg("tolerance")=-1,
                    py::arg("dist_type")="eucl", py::arg("min_clust_size")=2,
                    py::arg("strictness")=4)

            .def("fit", (kbase::cluster_t (Gmeans::*)(std::vector<double>&,
                            const size_t, const size_t)) &Gmeans::fit,
    R"pbdoc(
    Compute G-means clustering means on a dataset

    Positional arguments:
    ---------------------
    data:
        - List or flattened 1D numpy.ndarray
    nrow:
        - The number of samples in the dataset
    ncol:
        - The number of features in the dataset
       )pbdoc",
                    py::arg("data"), py::arg("nrow"), py::arg("ncol")
                )
            .def("fit", (kbase::cluster_t (Gmeans::*)(const std::string&,
                            const size_t, const size_t)) &Gmeans::fit,
    R"pbdoc(
    Compute G-means clustering means on a dataset

    Positional arguments:
    ---------------------
    datafn:
        - File name of data in raw row-major binary format
    nrow:
        - The number of samples in the dataset
    ncol:
        - The number of features in the dataset
       )pbdoc",
                    py::arg("datafn"), py::arg("nrow"), py::arg("ncol")
                )
            .def("fit", (kbase::cluster_t (Gmeans::*)(py::buffer)) &Gmeans::fit,
    R"pbdoc(
    Compute G-means clustering means on a dataset

    Positional arguments:
    ---------------------
    data:
        - A 2D numpy.ndarray
       )pbdoc",
                    py::arg("data")
                );

    // Versioning information
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
