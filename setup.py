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

from distutils.core import setup, Extension
from Cython.Build import cythonize
import sys, re
from exceptions import NotImplementedError
from Exceptions.runtime import UnsupportedError

_REPO_ISSUES_ = "https://github.com/flashxio/knorPy/issues"
_OS_SUPPORTED_ = {"linux":"linux", "mac":"darwin"}

patts = []
for opsys in _OS_SUPPORTED_.itervalues():
    patts.append(re.compile("(.*)("+opsys+")(.*)"))

raw_os = sys.platform.lower()

OS = None
for patt in patts:
    res = re.match(patt, raw_os)
    if res is not None:
        OS = res.groups()[1]
        break

if OS is None:
    raise UnsupportedError("Operating system {}\n." +\
            "Please post an issue at {}\n".format(raw_os, _REPO_ISSUES_))

elif OS == _OS_SUPPORTED_["linux"]:
    raise NotImplementedError("Linux OS support")
elif OS == _OS_SUPPORTED_["mac"]:
    setup(ext_modules = cythonize(Extension(
	"knor",                                # the extension name
	sources=["knor.pyx"],
	language="c++",
	extra_compile_args=["-std=c++11", "-O3",
        "-Wno-unused-function", "-I..","-I../libman",
        "-I../binding", "-I../libkcommon",
        "-I/usr/local/lib/python2.7/site-packages/numpy/core/include",
	"-DBOOST_LOG_DYN_LINK", "-I/usr/local/opt/boost/include",
	"-DBIND", "-DOSX"],
	extra_link_args=[
	"-L../libman", "-lman", "-L../libkcommon",
	"-lkcommon", "-lpthread", "-lboost_log-mt",
	"-lboost_system", "-L/usr/local/opt/boost/lib",
	])
	))
else:
    assert False, "Unsupported OS NOT correctly caught by knor"
