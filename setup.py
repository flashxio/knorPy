import os, sys, re
PYTHON_VERSION = sys.version_info[0]

if PYTHON_VERSION == 2:
    from exceptions import NotImplementedError
    from exceptions import RuntimeError

from glob import glob
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_clib import build_clib
import setuptools

__version__ = '0.0.5'

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

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.

    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])

        # Add Specific build dependencies
        # opts.append("-I/usr/include/boost")

        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

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
sources.append(os.path.join("knor", "knor.cpp"))

# Compile & link arguments
if OS == _OS_SUPPORTED_["linux"]:
    extra_compile_args = ["-std=c++11", "-fPIC",
            "-Wno-attributes", "-Wno-unused-variable",
            "-Wno-unused-function", "-I.","-Iknor/cknor/libman",
            "-Iknor/cknor/libauto",
            "-Iknor/cknor/binding", "-Iknor/cknor/libkcommon",
            "-fopenmp"]
    # extra_compile_args.append("-I"+find_header_loc("numpy"))
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

    # extra_compile_args.append("-I"+find_header_loc("numpy"))

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
ext_modules = [
        Extension(
        "knor.knor",                                # the extension name
        sources=sources,
        language="c++",
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True), # TODO include li
            ".", "knor/cknor/libman",
            "knor/cknor/binding", "knor/cknor/libkcommon"
        ],

        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args),
]

################################################################################

class knor_clib(build_clib, object):
    def initialize_options(self):
        super(knor_clib, self).initialize_options()
        if OS == _OS_SUPPORTED_["linux"]:
            self.include_dirs = [
                "knor/cknor/libman", "knor/cknor/libauto",
                "knor/cknor/binding", "knor/cknor/libkcommon",
                ]
            # self.include_dirs.append(find_header_loc("numpy"))
            self.define = [
                    ("BIND", None), ("USE_NUMA", None)
                    ]
        elif OS == _OS_SUPPORTED_["mac"]:
            self.include_dirs = [
                "knor/cknor/libman", "knor/cknor/binding",
                "knor/cknor/libkcommon"
                ]
            # self.include_dirs.append(find_header_loc("numpy"))

            self.define = [
                    ("BIND", None)
                    ]

        else:
            raise RuntimeError("Unsupported OS {}".format(raw_os))

    def build_libraries(self, libraries):
        for (lib_name, build_info) in libraries:
            sources = build_info.get("sources")
            if sources is None or not isinstance(sources, (list, tuple)):
                raise RuntimeError(("in \"libraries\" option (library \"%s\"), " +
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
    version=__version__,
    author="Disa Mhembere",
    author_email="disa@cs.jhu.edu",
    description="A fast parallel clustering library for Linux and Mac",
    long_description="The k* NUMA Optimized Routine library or " +\
    "knor is a highly optimized and fast library for computing " +\
    "clustering in parallel with accelerations for Non-Uniform Memory " +\
    "Access (NUMA) architectures",
    url="https://github.com/flashxio/knor",
    license="Apache License, Version 2.0",
    keywords="Parallel clustering machine-learning",
    install_requires=[
        "setuptools",
        "numpy",
        "pybind11",
        ],
    package_dir = {"knor": "knor"},
    packages=["knor", "knor.Exceptions"],
    libraries =libraries,
    cmdclass = {"build_clib": knor_clib, "build_ext": BuildExt},
    ext_modules = ext_modules,
    zip_safe=False,
    )
