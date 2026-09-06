"""clara -- CLARA / PAM k-medoids clustering, from scratch, pure Python.

No third-party dependencies are used anywhere in this package; only the Python
standard library (``math``, ``random``, ``csv``, ``argparse`` ...).

Public API is filled in as the package grows:
- :mod:`clara.distance`  -- dissimilarity functions  (available now)
- :class:`clara.KMedoids` -- Partitioning Around Medoids
- :class:`clara.CLARA`    -- Clustering LARge Applications
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
