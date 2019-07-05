from distutils.core import setup, Extension

eculidmodule = Extension('eculid',
        runtime_library_dirs = ['.'],
        extra_objects = ['../lib/libeculid.so'],
        sources = ['eculidmodule.cpp'])

setup (name = 'eculid',
        version = '1.0',
        description = 'eculid algorithm',
        ext_modules = [eculidmodule])
