# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.string cimport string
import numpy as np

cdef extern from "kmeans_types.hpp" namespace "kpmeans::base":
    cdef cppclass kmeans_t:
        kmeans_t()
        kmeans_t(const size_t, const size_t, const size_t,
                 const size_t, const unsigned*,
                 const size_t*, const vector[double]&)
        const void _print() const
        const void write(const string dirname) const
        void set_params(const size_t nrow, const size_t ncol,
                const size_t iters, const size_t k)
        size_t nrow
        size_t ncol
        size_t iters
        size_t k
        vector[unsigned] assignments
        vector[size_t] assignment_count
        vector[double] centroids

cdef class Pykmeans_t:
    cdef kmeans_t c_kmeans_t      # hold a C++ instance which we're wrapping
    def __cinit__(self, const size_t nrow, const size_t ncol, const size_t iters,
             const size_t k, vector[unsigned]& assignments_buf,
             const vector[size_t]& assignment_count_buf,
             const vector[double]& centroids):
        self.c_kmeans_t = kmeans_t(nrow, ncol, iters, k, assignments_buf.data(),
                assignment_count_buf.data(), centroids)

    def _print(self):
        return self.c_kmeans_t._print()

    def write(self, const string dirname):
        return self.c_kmeans_t.write(dirname)

    def get_nrow(self):
        return self.c_kmeans_t.nrow

    def get_k(self):
        return self.c_kmeans_t.k

    def get_iters(self):
        return self.c_kmeans_t.iters

    def get_ncol(self):
        return self.c_kmeans_t.ncol

    def get_clusters(self):
        return self.c_kmeans_t.assignments

    def get_sizes(self):
        return self.c_kmeans_t.assignment_count

    def get_centroids(self):
        return self.c_kmeans_t.centroids

    def __richcmp__(self, other, int op):
        if op == 2: # 2 is __eq__
            if isinstance(other, self.__class__):
                return np.array_equal(self.get_clusters(),
                        other.get_clusters()) and\
                    np.array_equal(self.get_sizes(), other.get_sizes()) and\
                    np.array_equal(self.get_centroids(),
                            other.get_centroids()) and\
                    np.array_equal(self.get_iters(), other.get_iters())
            else:
                return False

    def set_params(self, const size_t nrow, const size_t ncol,
            const size_t iters, const size_t k):
        return self.c_kmeans_t.set_params(nrow, ncol, iters, k)
