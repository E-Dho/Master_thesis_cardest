"""Alias of ``evaluation.eval_core.adapters.base`` (moved there to be shared by all pipelines).

The module object is the same, so attribute access and ``unittest.mock.patch`` on
this path keep working.
"""

import sys

from evaluation.eval_core.adapters import base as _module

sys.modules[__name__] = _module
