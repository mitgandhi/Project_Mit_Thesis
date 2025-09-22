"""Tools for building consolidated plot summaries.

The modules inside this package depend on each other (for example
``run_unified_batch`` relies on ``unified_piston_plots`` and the contact
pressure helpers).  Having an ``__init__`` file ensures the package can be
imported reliably without modifying ``sys.path`` at runtime.
"""

from __future__ import annotations

__all__ = []

