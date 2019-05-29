import sys
PYTHON_VERSION = sys.version_info[0]

if PYTHON_VERSION == 2:
    from knor import Kmeans
    from knor import KmeansPP
    from knor import cluster_t
else:
    from .knor import Kmeans
    from .knor import KmeansPP
    from .knor import cluster_t
