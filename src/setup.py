import os
from os.path import join

import numpy

from sklearn._build_utils import get_blas_info

#def configuration(parent_package='', top_path=None):
#    from numpy.distutils.misc_util import Configuration
#
#    config = Configuration('LatNet.src', parent_package, top_path)
#
#    cblas_libs, blas_info = get_blas_info()
#    if os.name == 'posix':
#        cblas_libs.append('m')
#
#    config.add_extension('cd_fast_fixed', sources=['cd_fast_fixed.pyx'],
#                         libraries=cblas_libs, #'openblas', #cblas_libs,
#                         include_dirs= #join('..', 'src', 'cblas'),
#                                        #numpy.get_include(),
#                                        blas_info['library_dirs'])
#                                        #os.curdir])
#                                        #blas_info.pop('include_dirs', [])],
#                                       #extra_compile_args=blas_info.pop('extra_compile_args', []), 
#                                       #**blas_info)
#
#    config.add_extension('cd_fast_adaptive', sources=['cd_fast_adaptive.pyx'],
#                         libraries=cblas_libs,#'openblas',
#                         include_dirs=#[join('..', 'src', 'cblas'),
#                                       # numpy.get_include(),
#                                        blas_info['library_dirs'])#])
#                                        #blas_info.pop('include_dirs', [])],
#                         #extra_compile_args=blas_info.pop('extra_compile_args',[]), 
#                                       #**blas_info)
##    config.add_extension('sgd_fast',
##                         sources=['sgd_fast.pyx'],
##                         include_dirs=[join('..', 'src', 'cblas'),
##                                       numpy.get_include(),
##                                       blas_info.pop('include_dirs', [])],
##                         libraries=cblas_libs,
##                         extra_compile_args=blas_info.pop('extra_compile_args',
##                                                          []),
##                         **blas_info)
##
##    config.add_extension('sag_fast',
##                         sources=['sag_fast.pyx'],
##                         include_dirs=numpy.get_include())
##
##    # add other directories
##    config.add_subpackage('tests')
#    from Cython.Build import cythonize
#    config.ext_modules[0] = cythonize(config.ext_modules[0])
#    config.ext_modules[1] = cythonize(config.ext_modules[1])
#
#    return config
#
#if __name__ == '__main__':
#    from numpy.distutils.core import setup
#    #print(**configuration(top_path='').todict())
#    setup(**configuration(top_path='').todict())


if __name__ == '__main__':
    from distutils.core import setup  
    from distutils.extension import Extension  
    from Cython.Build import cythonize  
    #from Cython.Distutils import build_ext

    cblas_libs, blas_info = get_blas_info()
    if os.name == 'posix':
        cblas_libs.append('m')

    extensions = Extension(name="cd_fast_adaptive",
                      sources= ["cd_fast_adaptive.pyx"],
                      libraries=cblas_libs,
                      library_dirs=[numpy.get_include(),
                                    blas_info['library_dirs'][0]],
                      include_dirs=blas_info['library_dirs'],
                      define_macros=blas_info['define_macros'],
                      language=blas_info['language'])

    setup( name = 'cd_fast_adaptive',
           description='the lasso with feature-wise adaptive regularizer',
           #cmdclass = {'build_ext': build_ext},
           ext_modules = cythonize(extensions), #, working=os.curdir),
          )
