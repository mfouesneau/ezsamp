# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Defines constants used in `astropy.vo.samp`.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import inspect

__all__ = ['SAMP_STATUS_OK', 'SAMP_STATUS_WARNING', 'SAMP_STATUS_ERROR',
           'SAFE_MTYPES', 'SAMP_ICON']

__profile_version__ = "1.3"


# Some default files
localpath = '/'.join(os.path.abspath(inspect.getfile(inspect.currentframe())).split('/')[:-1])

#: General constant for samp.ok status string
SAMP_STATUS_OK = "samp.ok"
#: General constant for samp.warning status string
SAMP_STATUS_WARNING = "samp.warning"
#: General constant for samp.error status string
SAMP_STATUS_ERROR = "samp.error"

SAFE_MTYPES = ["samp.app.*", "samp.msg.progress", "table.*", "image.*",
               "coord.*", "spectrum.*", "bibcode.*", "voresource.*"]

with open(localpath + '/data/astropy_icon.png', 'rb') as f:
    SAMP_ICON = f.read()


try:
    import ssl
except ImportError:
    SSL_SUPPORT = False
else:
    SSL_SUPPORT = True
    del ssl
