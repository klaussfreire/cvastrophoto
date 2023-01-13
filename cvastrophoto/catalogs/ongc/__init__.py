from __future__ import absolute_import

try:
    from pyongc import ongc
except ImportError:
    pass

if ongc.__version__ < '1.':
    from .ongc0 import OpenNGC0Catalog as OpenNGCCatalog
else:
    from .ongc1 import OpenNGC1Catalog as OpenNGCCatalog

