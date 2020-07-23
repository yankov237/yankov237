from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("pi_series_omega_control.pyx")
)

