#!/usr/bin/env python

# Copyright 2017 neurodata (http://neurodata.io/)
# Written by Disa Mhembere (disa@jhu.edu)
#
# This file is part of knor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Note these tests only work from the knorPy/test directory

import sys

__knor_version__ = "0.0.4"

PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    COMMON_INSTALL_LOC = \
        "/usr/local/lib/python2.7/dist-packages/knor-"+ \
            __knor_version__ + "-py2.7-linux-x86_64.egg"
elif PYTHON_VERSION == 3:
    COMMON_INSTALL_LOC = \
    "/usr/local/lib/python3.5/dist-packages/knor-" + \
        __knor_version__ + "-py3.5-linux-x86_64.egg"

if COMMON_INSTALL_LOC not in sys.path:
    sys.path.append(COMMON_INSTALL_LOC)

import knor
import numpy as np

FN = "../data/matrix_r50_c5_rrw.bin"

def dim_test_c_comp():
    data = np.random.random((10,3))
    km = knor.Kmeans(4)
    return km.fit(data)

def dim_test_c_im():
    data = np.random.random((10,3))
    centers = np.random.random((3,3))
    km = knor.Kmeans(3, centers=centers) # TODO: Infer k
    return km.fit(data)


def dexm_test_c_comp():
    km = knor.Kmeans(8)
    return km.fit(FN, 50, 5)

def dexm_test_c_im():
    k = 2
    centers = np.random.random((k,5))
    km = knor.Kmeans(k, centers=centers)
    return km.fit(FN, 50, 5)

def test_err():
    try:
        km = knor.Kmeans(7.2)
        return  km.fit(FN, nrow=50, ncol=5)
    except Exception as msg:
        print(("Successful test: CORRECTLY fails with message: {}".format(msg)))

print("\n\n*************** TEST 1/5 ******************\n\n")
print(dim_test_c_comp())
print("\n\n*************** TEST 2/5 ******************\n\n")
print(dim_test_c_im())
print("\n\n*************** TEST 3/5 ******************\n\n")
print(dexm_test_c_comp())
print("\n\n*************** TEST 4/5 ******************\n\n")
print(dexm_test_c_im())
print("\n\n*************** TEST 5/5 ******************\n\n")
test_err()
