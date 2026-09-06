"""clara -- CLARA / PAM k-medoids clustering, from scratch, pure Python.

No third-party dependencies are used anywhere in this package; only the Python
standard library (``math``, ``random``, ``csv``, ``argparse`` ...).

Public API:
- :class:`clara.CLARA`     -- Clustering LARge Applications (sampling + PAM)
- :class:`clara.KMedoids`  -- Partitioning Around Medoids (BUILD + SWAP)
- :mod:`clara.distance`    -- dissimilarity functions
- :mod:`clara.metrics`     -- silhouette score, total / average cost
- :mod:`clara.dataset`     -- CSV loading and synthetic blobs

Command line:  ``python -m clara path/to/data.csv -k 4``
"""

from __future__ import annotations

from .clara import CLARA
from .kmedoids import KMedoids

__version__ = "0.1.0"
__all__ = ["CLARA", "KMedoids", "__version__"]
