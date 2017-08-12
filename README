# knor for Python

[![Build
Status](https://travis-ci.org/flashxio/knorPy.svg?branch=master)](https://travis-ci.org/flashxio/knorPy)

These are Python 2.7+ (Including Python 3) bindings for the standalone
machine in-memory portion of the
`knor` (k-means NUMA Optimized Routines Library) as known as `knori`. See the
full C++ library for details: https://github.com/flashxio/knor

## Supported OSes

The `knor` python package has been tested on the following OSes:
- Mac OSX Sierra
- Ubuntu LTS 14.04 and 16.04
- Debian Linux 

## Python Dependencies

- Numpy: `pip install -U numpy`
- Cython: `pip install -U Cython`

## Mac Installation

```
pip install knor
```

## Linux installation

### Best Performance configuration

For the best performance make sure the `numa` system package is installed via

```
apt-get install -y libnuma-dbg libnuma-dev libnuma1
```

Then simply install the package via pip:

```
pip install knor
```

## Installation Errors

1.

```
  File “/Library/Python/2.7/site-packages/pip/utils/__init__.py”, line X, in
  call_subprocess
      % (command_desc, proc.returncode, cwd))
      InstallationError: Command “python setup.py egg_info” failed with error
      code 1 in /private/tmp/pip-build-vaASFl/knor/
```

### Solution

Update your `python` version to at least `2.7.10`

2.

```
    In file included from knor/cknor/libkcommon/clusters.cpp:23:
    knor/cknor/libkcommon/util.hpp:29:10: fatal error: ‘random’ file not found
    #include <random>
             ^
    1 error generated.
    error: command ‘/usr/bin/clang’ failed with exit status 1
```
### Solution
This usually occurs on Mac when your Xcode and Xcode command line tools are out of 
date. Update then to at least Version 8


3.

```
from Cython.Build import cythonize
ImportError: No module named Cython.Build
```

### Solution
Install Cython via `pip install -U cython`. It is necessary to bridge the gap between the underlying C++ and the Python bindings.

4.

```
unable to execute 'x86_64-linux-gnu-gcc': No such file or directory
  error: command 'x86_64-linux-gnu-gcc' failed with exit status 1
```

### Solution
Install the `gcc` compiler via `apt-get install build-essential` 

5.

```
 fatal error: Python.h: No such file or directory compilation terminated.
```

### Solution
Install the development headers for the version on Python you intend to install knor e.g

```
apt-get install python-dev  # for python2.x installs
apt-get install python3-dev  # for python3.x installs
```

## Documentation

```
import knor
help(knor.Kmeans)
```

## Example

```
import knor
import numpy as np
data = np.random.random((100, 10))
ret = knor.Kmeans(data, 5)
print(ret)
print(ret.get_sizes())
```

## The `knor_t` return object

The `knor_t` return object has `getter` methods for:

- `k`: The number of clusters requested
- `nrow`: The number of rows/samples in the dataset
- `ncol`: Then number of columns/features in the dataset
- `sizes`: The number of samples in each cluster/centroid
- `iters`: The number of iterations of k-means performed
- `centroids`: A `numpy ndarray` where each row is a cluster center
- `clusters`: An index for which cluster a sample was assigned

Each item can be accessed using the prefix `get_` followed by the name
of the attribute. E.g., `get_nrow()`, `get_sizes()`, etc.


