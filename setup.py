import os, sys, re
PYTHON_VERSION = sys.version_info[0]

if PYTHON_VERSION == 2:
    from exceptions import NotImplementedError
    from exceptions import RuntimeError

from glob import glob
from distutils.errors import DistutilsSetupError
from distutils.command.build_clib import build_clib
from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from utils import find_header_loc

_REPO_ISSUES_ = "https://github.com/flashxio/knorPy/issues"
_OS_SUPPORTED_ = {"linux":"linux", "mac":"darwin"}

patts = []
for opsys in _OS_SUPPORTED_.values():
    patts.append(re.compile("(.*)("+opsys+")(.*)"))

raw_os = sys.platform.lower()

OS = None
for patt in patts:
    res = re.match(patt, raw_os)
    if res is not None:
        OS = res.groups()[1]
        break

if OS not in list(_OS_SUPPORTED_.values()):
    raise RuntimeError("Operating system {}\n." +\
            "Please post an issue at {}\n".format(raw_os, _REPO_ISSUES_))


# Hack to stop -Wstrict-prototypes warning on linux
if OS == _OS_SUPPORTED_["linux"]:
    from distutils import sysconfig

    sysconfig._config_vars["OPT"] = \
            sysconfig._config_vars["OPT"].replace("-Wstrict-prototypes", "")
    sysconfig._config_vars["OPT"] = \
            sysconfig._config_vars["OPT"].replace("-O2", "-O3")
    (os.environ["OPT"],) = sysconfig.get_config_vars("OPT")

################################ End VarDecl ###################################

# For C++ libraries
libkcommon = ("kcommon",
        {"sources": glob(os.path.join("knor", "cknor", "libkcommon", "*.cpp"))})
libman = ("man",
        {"sources": glob(os.path.join("knor", "cknor", "libman", "*.cpp"))})

# Minimum libraries we will build
libraries = [libkcommon, libman]

if OS == _OS_SUPPORTED_["linux"]:
    libauto = ("auto",
            {"sources": glob(os.path.join("knor", "cknor", "libauto", "*.cpp"))})
    libraries.append(libauto)

# Add some sources + cython modules
sources = glob(os.path.join("knor", "cknor", "binding", "*.cpp"))
sources.append(os.path.join("knor", "knor.pyx"))

# Compile & link arguments
if OS == _OS_SUPPORTED_["linux"]:
    extra_compile_args = ["-std=c++11", "-fPIC",
            "-Wno-attributes", "-Wno-unused-variable",
            "-Wno-unused-function", "-I.","-Iknor/cknor/libman",
            "-Iknor/cknor/libauto",
            "-Iknor/cknor/binding", "-Iknor/cknor/libkcommon",
            "-fopenmp"]
    extra_compile_args.append("-I"+find_header_loc("numpy"))
    extra_compile_args.extend(["-DBIND", "-DUSE_NUMA"])

    extra_link_args=["-Lknor/cknor/libman", "-lman", "-Llknor/cknor/libauto",
            "-lauto",
            "-Lknor/cknor/libkcommon", "-lkcommon", "-lpthread",
            "-lnuma", "-fopenmp",
            ]

elif OS == _OS_SUPPORTED_["mac"]:
    extra_compile_args = ["-std=c++11",
            "-Wno-unused-function", "-I.","-Iknor/cknor/libman",
            "-Iknor/cknor/binding", "-Iknor/cknor/libkcommon"]

    extra_compile_args.append("-I"+find_header_loc("numpy"))

    if PYTHON_VERSION == 2:
        extra_compile_args.append("-I/usr/include/python2.7")
    else:
        extra_compile_args.append("-I/usr/include/python3.5")

    extra_compile_args.append("-DBIND")

    extra_link_args=["-Lknor/cknor/libman", "-lman",
            "-Lknor/cknor/libkcommon",
            "-lkcommon", "-lpthread",
            ]
else:
    raise RuntimeError("Unsupported OS {}".format(raw_os))

# Build cython modules
ext_modules = cythonize(Extension(
        "knor.knor",                                # the extension name
        sources=sources,
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args))

################################################################################

class knor_clib(build_clib, object):
    def initialize_options(self):
        super(knor_clib, self).initialize_options()
        if OS == _OS_SUPPORTED_["linux"]:
            self.include_dirs = [
                "knor/cknor/libman", "knor/cknor/libauto",
                "knor/cknor/binding", "knor/cknor/libkcommon",
                ]
            self.include_dirs.append(find_header_loc("numpy"))
            self.define = [
                    ("BIND", None), ("USE_NUMA", None)
                    ]
        elif OS == _OS_SUPPORTED_["mac"]:
            self.include_dirs = [
                "knor/cknor/libman", "knor/cknor/binding",
                "knor/cknor/libkcommon"
                ]
            self.include_dirs.append(find_header_loc("numpy"))

            self.define = [
                    ("BIND", None)
                    ]

        else:
            raise RuntimeError("Unsupported OS {}".format(raw_os))

    def build_libraries(self, libraries):
        for (lib_name, build_info) in libraries:
            sources = build_info.get("sources")
            if sources is None or not isinstance(sources, (list, tuple)):
                raise DistutilsSetupError(("in \"libraries\" option (library \"%s\"), " +
                       "\"sources\" must be present and must be " +
                       "a list of source filenames") % lib_name)
            sources = list(sources)

            print("building \"%s\" library" % lib_name)

            # First, compile the source code to object files in the library
            # directory.  (This should probably change to putting object
            # files in a temporary build directory.)
            macros = build_info.get("macros")
            include_dirs = build_info.get("include_dirs")

            # pass flasgs to compiler
            extra_preargs = ["-std=c++11", "-Wno-unused-function"]

            if OS == _OS_SUPPORTED_["linux"]:
                extra_preargs.append("-fopenmp")

            objects = self.compiler.compile(sources,
                                            output_dir=self.build_temp,
                                            macros=macros,
                                            include_dirs=include_dirs,
                                            debug=self.debug,
                                            extra_preargs=extra_preargs)

            # Now "link" the object files together into a static library.
            # (On Unix at least, this isn"t really linking -- it just
            # builds an archive.  Whatever.)
            self.compiler.create_static_lib(objects, lib_name,
                                            output_dir=self.build_clib,
                                            debug=self.debug)

################################################################################

# Run the setup
setup(
    name="knor",
    version="0.0.1a17",
    description="A fast parallel k-means library for Linux and Mac",
    long_description="The k-means NUMA Optimized Routine library or " +\
    "knor is a highly optimized and fast library for computing " +\
    "k-means in parallel with accelerations for Non-Uniform Memory " +\
    "Access (NUMA) architectures",
    url="https://github.com/flashxio/knor",
    author="Disa Mhembere",
    author_email="disa@jhu.edu",
    license="Apache License, Version 2.0",
    keywords="kmeans k-means parallel clustering machine-learning",
    install_requires=[
        "numpy",
        "Cython==0.23.5",
        "cython==0.23.5",
        ],
    package_dir = {"knor": "knor"},
    packages=["knor", "knor.Exceptions"],
    libraries =libraries,
    cmdclass = {"build_clib": knor_clib, "build_ext": build_ext},
    ext_modules = ext_modules,
    )
