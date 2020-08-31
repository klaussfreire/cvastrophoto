import sys
import os.path

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

VERSION = "0.1.0"

version_path = os.path.join(os.path.dirname(__file__), 'cvastrophoto', '_version.py')
if not os.path.exists(version_path):
    with open(version_path, "w") as version_file:
        pass
with open(version_path, "r+") as version_file:
    version_content = "__version__ = %r" % (VERSION,)
    if version_file.read() != version_content:
        version_file.seek(0)
        version_file.write(version_content)
        version_file.flush()
        version_file.truncate()

with open(os.path.join(os.path.dirname(__file__), 'README.rst')) as readme_file:
    readme = readme_file.read()

import re
with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as requirements_file:
    requirements = list(filter(bool, [ r.strip() for r in requirements_file ]))

extra = {}

packages = [
    "cvastrophoto",
]

if cythonize is not None:
    extra['ext_modules'] = cythonize([
        Extension(
            'cvastrophoto.rops.denoise.diffusion',
            sources=['cvastrophoto/rops/denoise/diffusion.py']
        ),
    ])


setup(
    name = "cvastrophoto",
    version = VERSION,
    description = "Computational Astrophotography Tools",
    author = "Claudio Freire",
    author_email = "klaussfreire@gmail.com",
    url = "https://github.com/klaussfreire/cvastrophoto/",
    license = "LGPLv3",
    long_description = readme,
    packages = packages,

    tests_require = 'nose',
    test_suite = 'tests',

    install_requires = requirements,

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries",
        "Operating System :: OS Independent",
    ],
    **extra
)

