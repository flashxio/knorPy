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
        unsigned k;
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
        init(init), tolerance(tolerance), dist_type(dist_type){

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
};

class KmeansPP {
    private:
        unsigned k;
        unsigned nstart;
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
};

class SKmeans {
    private:
        unsigned k;
        size_t max_iters;
        unsigned nthread;
        std::vector<double> centers;
        std::string init;
        double tolerance;

    public:
        SKmeans(const unsigned k, size_t max_iters, unsigned nthread,
                std::vector<double>& centers, std::string& init,
                double tolerance) : k(k),
        max_iters(max_iters), nthread(nthread), centers(centers),
        init(init), tolerance(tolerance) {
        }

        inline kbase::cluster_t fit(std::vector<double>& data,
                const size_t nrow, const size_t ncol) {

            return knor::skmeans_coordinator::create(
                    "", nrow, ncol, k, max_iters, kbase::get_num_nodes(),
                    nthread, (centers.size() ? &centers[0]:NULL), init,
                    tolerance, "cos")->run(&data[0]);
        }

        inline kbase::cluster_t fit(const std::string datafn, const size_t nrow,
                const size_t ncol) {

            return knor::skmeans_coordinator::create(
                    datafn, nrow, ncol, k, max_iters, kbase::get_num_nodes(),
                    nthread, (centers.size() ? &centers[0]:NULL),
                    init, tolerance, "cos")->run();
        }
};

class FuzzyCMeans {
    public:
        inline kbase::cluster_t fit(std::vector<double>& data, const size_t nrow,
                const size_t ncol, const unsigned k, const unsigned max_iters,
                const unsigned nthread, std::vector<double>& centers,
                const std::string init, const double tolerance,
                const std::string dist_type, const unsigned fuzzindex) {


            return knor::fcm_coordinator::create(
                    "", nrow, ncol, k, max_iters, kbase::get_num_nodes(),
                    nthread, (centers.size() ? &centers[0]:NULL), init,
                    tolerance, dist_type, fuzzindex)->run(&data[0]);
        }

        inline kbase::cluster_t fit(const std::string fn, const size_t nrow,
                const size_t ncol, const unsigned k, const unsigned max_iters,
                const unsigned nthread, std::vector<double>& centers,
                const std::string init, const double tolerance,
                const std::string dist_type, const unsigned fuzzindex) {

            return knor::fcm_coordinator::create(
                    fn, nrow, ncol, k, max_iters, kbase::get_num_nodes(),
                    nthread, (centers.size() ? &centers[0]:NULL),
                    init, tolerance, dist_type, fuzzindex)->run();
        }
};

class Kmedoids {
    public:
        inline kbase::cluster_t fit(std::vector<double>& data, const size_t nrow,
                const size_t ncol, const unsigned k, const unsigned max_iters,
                const unsigned nthread, std::vector<double>& centers,
                const std::string init, const double tolerance,
                const std::string dist_type, const double sample_rate) {

                return knor::medoid_coordinator::create("",
                        nrow, ncol, k, max_iters,
                        kbase::get_num_nodes(), nthread,
                        (centers.size() ? &centers[0]:NULL),
                        init, tolerance, dist_type, sample_rate)->run(&data[0]);
        }

        inline kbase::cluster_t fit(const std::string fn, const size_t nrow,
                const size_t ncol, const unsigned k, const unsigned max_iters,
                const unsigned nthread, std::vector<double>& centers,
                const std::string init, const double tolerance,
                const std::string dist_type, const double sample_rate) {

            auto coord =
                knor::medoid_coordinator::create("",
                        nrow, ncol, k, max_iters, kbase::get_num_nodes(),
                        nthread, (centers.size() ? &centers[0]:NULL),
                        init, tolerance, dist_type, sample_rate);

            std::vector<double> data(nrow*ncol);
            kbase::bin_io<double> br(fn, nrow, ncol);
            br.read(&data);
            return coord->run(&data[0]);
            // TODO: if (centers) delete [] centers;
            // TODO: delete data
        }
};

class Hmeans {
    public:
        inline kbase::cluster_t fit(std::vector<double>& data,
                const size_t nrow, const size_t ncol,
                const unsigned kmax, const unsigned max_iters,
                const unsigned nthread,
                std::vector<double>& centers, const std::string init,
                const double tolerance, const std::string dist_type,
                const unsigned min_clust_size) {

        return knor::hclust_coordinator::create("", nrow, ncol, kmax,
                max_iters, kbase::get_num_nodes(), nthread,
                (centers.size() ? &centers[0]:NULL),
                init, tolerance, dist_type, min_clust_size)->run(&data[0]);
        }

        inline kbase::cluster_t fit(const std::string fn,
                const size_t nrow,
                const size_t ncol, const unsigned kmax, const unsigned max_iters,
                const unsigned nthread, std::vector<double>& centers,
                const std::string init, const double tolerance,
                const std::string dist_type, const unsigned min_clust_size) {

        return knor::hclust_coordinator::create(fn, nrow, ncol, kmax,
                max_iters, kbase::get_num_nodes(), nthread,
                (centers.size() ? &centers[0]:NULL),
                init, tolerance, dist_type, min_clust_size)->run();
        }
};

class  Xmeans {
    public:
        inline kbase::cluster_t fit(std::vector<double>& data, const size_t nrow,
                const size_t ncol, const unsigned kmax, const unsigned max_iters,
                const unsigned nthread,
                std::vector<double>& centers, const std::string init,
                const double tolerance, const std::string dist_type,
                const unsigned min_clust_size) {

        return knor::xmeans_coordinator::create("", nrow, ncol, kmax,
                max_iters, kbase::get_num_nodes(),
                nthread, (centers.size() ? &centers[0]:NULL),
                init, tolerance, dist_type,
                min_clust_size)->run(&data[0]);

        }

        inline kbase::cluster_t fit(const std::string fn, const size_t nrow,
                const size_t ncol, const unsigned kmax, const unsigned max_iters,
                const unsigned nthread,
                std::vector<double>& centers, const std::string init,
                const double tolerance, const std::string dist_type,
                const unsigned min_clust_size) {

        return knor::xmeans_coordinator::create(fn, nrow, ncol, kmax,
                max_iters, kbase::get_num_nodes(),
                nthread, (centers.size() ? &centers[0]:NULL), init,
                tolerance, dist_type, min_clust_size)->run();
        }
};

class Gmeans {
    public:
        inline kbase::cluster_t fit(std::vector<double>& data, const size_t nrow,
                const size_t ncol, const unsigned kmax, const unsigned max_iters,
                const unsigned nthread,
                std::vector<double>& centers, const std::string init,
                const double tolerance, const std::string dist_type,
                const unsigned min_clust_size, const short strictness) {

        return knor::gmeans_coordinator::create("", nrow, ncol, kmax,
                max_iters, kbase::get_num_nodes(),
                nthread, (centers.size() ? &centers[0]:NULL), init,
                tolerance, dist_type, min_clust_size,
                strictness)->run(&data[0]);
        }

        inline kbase::cluster_t fit(const std::string fn, const size_t nrow,
                const size_t ncol, const unsigned kmax, const unsigned max_iters,
                const unsigned nthread,
                std::vector<double>& centers, const std::string init,
                const double tolerance, const std::string dist_type,
                const unsigned min_clust_size, const short strictness) {

        return knor::gmeans_coordinator::create(fn, nrow, ncol, kmax,
                max_iters, kbase::get_num_nodes(),
                nthread, (centers.size() ? &centers[0]:NULL), init,
                tolerance, dist_type,
                min_clust_size, strictness)->run();
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

    ////////////////////////// Kmeans //////////////////////////
    py::class_<kbase::cluster_t>(m, "cluster_t")
            .def(py::init(), "Create a cluster_t return object")
            .def("__repr__", &kbase::cluster_t::to_str);

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
        - The maximum number of iteration of k-means to perform

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
       )pbdoc",
            py::arg("k"), py::arg("max_iters")=20, py::arg("nthread")=2,
            py::arg("centers")=std::vector<double>(),
            py::arg("init")="kmeanspp", py::arg("tolerance")=-1,
            py::arg("dist_type")="eucl"
            )
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
       )pbdoc"
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
        - The maximum number of iteration of k-means to perform

    Optional arguments:
    -------------------
    nstart:
        -  The number of iterations of kmeans++ to run
    nthread:
        - The number of parallel threads to run
    dist_type: What dissimilarity metric to use: "eucl", "cos",
        "taxi", "sqeucl"
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
        - List or numpy.ndarray
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
        - The maximum number of iteration of k-means to perform

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
   )pbdoc",
            py::arg("k"), py::arg("max_iters")=20, py::arg("nthread")=2,
        py::arg("centers")=std::vector<double>(),
        py::arg("init")="kmeanspp", py::arg("tolerance")=-1)
            .def("fit", (kbase::cluster_t (SKmeans::*)(std::vector<double>&,
                            const size_t, const size_t)) &SKmeans::fit,
    R"pbdoc(
    Run the spherical k-means algorithm on the dataset

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
            .def("fit", (kbase::cluster_t (SKmeans::*)(const std::string,
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
                );

    // FuzzyCMeans
    py::class_<FuzzyCMeans>(m, "FuzzyCMeans")
            .def(py::init(), "Create a FuzzyCMeans object")
            .def("fit", (kbase::cluster_t (FuzzyCMeans::*)(std::vector<double>&,
                        const size_t, const size_t, const unsigned,
                        const unsigned, const unsigned, std::vector<double>&,
                        const std::string, const double, const std::string,
                        const unsigned)) &FuzzyCMeans::fit,
                    "Compute FuzzyCMeans on the dataset provided",
                    py::arg("data"), py::arg("nrow"), py::arg("ncol"),
                    py::arg("k"), py::arg("max_iters")=20, py::arg("nthread")=2,
                    py::arg("centers")=std::vector<double>(),
                    py::arg("init")="forgy", py::arg("tolerance")=-1,
                    py::arg("dist_type")="cos", py::arg("fuzzindex")=2
                )
            .def("fit", (kbase::cluster_t (FuzzyCMeans::*)(
                            const std::string, const size_t, const size_t,
                            const unsigned, const unsigned, const unsigned,
                            std::vector<double>&, const std::string, const double,
                            const std::string, const unsigned)) &FuzzyCMeans::fit,
                    "Compute FuzzyCMeans on the dataset provided",
                    py::arg("fn"), py::arg("nrow"), py::arg("ncol"),
                    py::arg("k"), py::arg("max_iters")=20, py::arg("nthread")=2,
                    py::arg("centers")=std::vector<double>(),
                    py::arg("init")="forgy", py::arg("tolerance")=-1,
                    py::arg("dist_type")="cos", py::arg("fuzzindex")=2
                    );

    // Kmedoids
    py::class_<Kmedoids>(m, "Kmedoids")
            .def(py::init(), "Create a CLARA object")
            .def("fit", (kbase::cluster_t (Kmedoids::*)(std::vector<double>&,
                            const size_t, const size_t,
                            const unsigned, const unsigned,
                            const unsigned, std::vector<double>&,
                            const std::string,
                            const double, const std::string,
                            const double)) &Kmedoids::fit,
                    "Compute CLARA on the dataset provided",
                    py::arg("data"), py::arg("nrow"), py::arg("ncol"),
                    py::arg("k"), py::arg("max_iters")=20, py::arg("nthread")=2,
                    py::arg("centers")=std::vector<double>(),
                    py::arg("init")="random", py::arg("tolerance")=-1,
                    py::arg("dist_type")="taxi", py::arg("sample_rate")=.2
                )
            .def("fit", (kbase::cluster_t (Kmedoids::*)(
                            const std::string, const size_t,
                            const size_t, const unsigned, const unsigned,
                            const unsigned, std::vector<double>&,
                            const std::string,
                            const double, const std::string,
                            const double)) &Kmedoids::fit,
                    "Compute CLARA on the dataset provided",
                    py::arg("fn"), py::arg("nrow"), py::arg("ncol"),
                    py::arg("k"), py::arg("max_iters")=20, py::arg("nthread")=2,
                    py::arg("centers")=std::vector<double>(),
                    py::arg("init")="random", py::arg("tolerance")=-1,
                    py::arg("dist_type")="taxi", py::arg("sample_rate")=.2
                    );

    // Hmeans
    py::class_<Hmeans>(m, "Hmeans")
            .def(py::init(), "Create a Hierarchical clustering means object")
            .def("fit", (kbase::cluster_t (Hmeans::*)(std::vector<double>&,
                            const size_t, const size_t, const unsigned,
                            const unsigned, const unsigned,
                            std::vector<double>&, const std::string,
                            const double, const std::string,
                            const unsigned)) &Hmeans::fit,
                    "Compute Hierarchical clustering means on the dataset",
                    py::arg("data"), py::arg("nrow"), py::arg("ncol"),
                    py::arg("k"), py::arg("max_iters")=20, py::arg("nthread")=2,
                    py::arg("centers")=std::vector<double>(),
                    py::arg("init")="kmeanspp", py::arg("tolerance")=-1,
                    py::arg("dist_type")="eucl", py::arg("min_clust_size")=2
                )
            .def("fit", (kbase::cluster_t (Hmeans::*)(const std::string,
                            const size_t, const size_t, const unsigned,
                            const unsigned, const unsigned, std::vector<double>&,
                            const std::string, const double, const std::string,
                            const unsigned)) &Hmeans::fit,
                    "Compute Hierarchical clustering means on the dataset",
                    py::arg("fn"), py::arg("nrow"), py::arg("ncol"),
                    py::arg("k"), py::arg("max_iters")=20, py::arg("nthread")=2,
                    py::arg("centers")=std::vector<double>(),
                    py::arg("init")="kmeanspp", py::arg("tolerance")=-1,
                    py::arg("dist_type")="eucl", py::arg("min_clust_size")=2
                    );

    // Xmeans
    py::class_<Xmeans>(m, "Xmeans")
            .def(py::init(), "Create an Xmeans clustering object")
            .def("fit", (kbase::cluster_t (Xmeans::*)(std::vector<double>&,
                            const size_t, const size_t,
                            const unsigned, const unsigned,
                            const unsigned, std::vector<double>&,
                            const std::string, const double, const std::string,
                            const unsigned)) &Xmeans::fit,
                    "Compute Xmeans on the dataset provided",
                    py::arg("data"), py::arg("nrow"), py::arg("ncol"),
                    py::arg("k"), py::arg("max_iters")=20, py::arg("nthread")=2,
                    py::arg("centers")=std::vector<double>(),
                    py::arg("init")="kmeanspp", py::arg("tolerance")=-1,
                    py::arg("dist_type")="eucl", py::arg("min_clust_size")=2
                )
            .def("fit", (kbase::cluster_t (Xmeans::*)(const std::string,
                            const size_t, const size_t, const unsigned,
                            const unsigned, const unsigned, std::vector<double>&,
                            const std::string, const double, const std::string,
                            const unsigned)) &Xmeans::fit,
                    "Compute Xmeans on the dataset provided",
                    py::arg("fn"), py::arg("nrow"), py::arg("ncol"),
                    py::arg("k"), py::arg("max_iters")=20, py::arg("nthread")=2,
                    py::arg("centers")=std::vector<double>(),
                    py::arg("init")="kmeanspp", py::arg("tolerance")=-1,
                    py::arg("dist_type")="eucl", py::arg("min_clust_size")=2
                    );

    // Gmeans
    py::class_<Gmeans>(m, "Gmeans")
            .def(py::init(), "Create an Gmeans clustering object")
            .def("fit", (kbase::cluster_t (Gmeans::*)(std::vector<double>&,
                            const size_t, const size_t, const unsigned,
                            const unsigned, const unsigned,
                            std::vector<double>&, const std::string,
                            const double, const std::string,
                            const unsigned, const short)) &Gmeans::fit,
                    "Compute Gmeans on the dataset provided",
                    py::arg("data"), py::arg("nrow"), py::arg("ncol"),
                    py::arg("k"), py::arg("max_iters")=20, py::arg("nthread")=2,
                    py::arg("centers")=std::vector<double>(),
                    py::arg("init")="kmeanspp", py::arg("tolerance")=-1,
                    py::arg("dist_type")="eucl", py::arg("min_clust_size")=2,
                    py::arg("strictness")=4
                )
            .def("fit", (kbase::cluster_t (Gmeans::*)(const std::string,
                            const size_t, const size_t, const unsigned,
                            const unsigned, const unsigned, std::vector<double>&,
                            const std::string, const double, const std::string,
                            const unsigned, const short)) &Gmeans::fit,
                    "Compute Gmeans on the dataset provided",
                    py::arg("fn"), py::arg("nrow"), py::arg("ncol"),
                    py::arg("k"), py::arg("max_iters")=20, py::arg("nthread")=2,
                    py::arg("centers")=std::vector<double>(),
                    py::arg("init")="kmeanspp", py::arg("tolerance")=-1,
                    py::arg("dist_type")="eucl", py::arg("min_clust_size")=2,
                    py::arg("strictness")=4
                );

    // Versioning information
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
