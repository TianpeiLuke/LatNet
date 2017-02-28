import os
from os.path import join
#from distutils.core import setup, Extension

#from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

from sklearn._build_utils import get_blas_info
from numpy.distutils.command import build_src



# a bit of monkeypatching ...
import Cython.Compiler.Main
build_src.Pyrex = Cython
build_src.have_pyrex = True
def have_pyrex():
    import sys
    try:
        import Cython.Compiler.Main
        sys.modules['Pyrex'] = Cython
        sys.modules['Pyrex.Compiler'] = Cython.Compiler
        sys.modules['Pyrex.Compiler.Main'] = Cython.Compiler.Main
        return True
    except ImportError:
        return False
build_src.have_pyrex = have_pyrex

##########################
# BEGIN additionnal code #
##########################
from numpy.distutils.misc_util import appendpath
from numpy.distutils import log
from os.path import join as pjoin, dirname
from distutils.dep_util import newer_group
from distutils.errors import DistutilsError


def generate_a_pyrex_source(self, base, ext_name, source, extension):
    ''' Monkey patch for numpy build_src.build_src method
    Uses Cython instead of Pyrex.
    Assumes Cython is present
    '''
    if self.inplace:
        target_dir = dirname(base)
    else:
        target_dir = appendpath(self.build_src, dirname(base))
    target_file = pjoin(target_dir, ext_name + '.c')
    depends = [source] + extension.depends
    if self.force or newer_group(depends, target_file, 'newer'):
        import Cython.Compiler.Main
        log.info("cythonc:> %s" % (target_file))
        self.mkpath(target_dir)
        options = Cython.Compiler.Main.CompilationOptions(
            defaults=Cython.Compiler.Main.default_options,
            include_path=extension.include_dirs,
            output_file=target_file)
        cython_result = Cython.Compiler.Main.compile(source, options=options)
        if cython_result.num_errors != 0:
            raise DistutilsError("%d errors while compiling %r with Cython" % (cython_result.num_errors, source))
    return target_file

build_src.build_src.generate_a_pyrex_source = generate_a_pyrex_source
########################
# END additionnal code #
########################


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('whatever', parent_package, top_path)

    cblas_libs, blas_info = get_blas_info()

    if os.name == 'posix':
        cblas_libs.append('m')

    config.add_extension('cd_fast_fixed',
                         sources=['cd_fast_fixed.pyx'],
                         include_dirs=[#join('..', 'src', 'cblas'),
                                       numpy.get_include(),
                                       blas_info.pop('include_dirs', [])],
                         libraries=cblas_libs,
                         extra_compile_args=blas_info.pop('extra_compile_args',
                                                          []),
                         **blas_info)

    config.add_extension('cd_fast_adaptive', sources=['cd_fast_adaptive.pyx'],
                         libraries=cblas_libs,
                         include_dirs=[#join('..', 'src', 'cblas'),
                                       numpy.get_include(),
                                       blas_info.pop('include_dirs', [])],
                         extra_compile_args=blas_info.pop('extra_compile_args',
                                                          []), **blas_info)

#    config.add_extension('test',
#                         sources=['test.pyx'],
#                         include_dirs=numpy.get_include())

    # add other directories
    #config.add_subpackage('tests')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration().todict())
