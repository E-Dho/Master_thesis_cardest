"""Compatibility alias for DistJoin commit 09908b9.

That upstream commit renamed ``AQP_estimator.py`` to ``estimator.py`` without
updating its imports.  Keeping the alias outside the pinned checkout lets the
evaluation use the exact source revision without editing it.
"""

from estimator import *  # noqa: F401,F403

