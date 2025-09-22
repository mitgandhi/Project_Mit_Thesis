"""Utilities for analysing optimization simulation results.

This package groups together a collection of analysis scripts that share
common helper functions defined in :mod:`plots_optimization.T1`.  Historically
the scripts imported ``T1`` by mutating ``sys.path`` at runtime which made the
code sensitive to the current working directory.  The package is now
explicitly defined so the modules can be imported using the standard Python
package machinery.
"""

from __future__ import annotations

__all__ = []

