"""Alias of ``evaluation.eval_core.records`` (moved there to be shared by all pipelines).

The module object is the same, so attribute access and ``unittest.mock.patch`` on
this path keep working.
"""

import sys

from evaluation.eval_core import records as _module

sys.modules[__name__] = _module
