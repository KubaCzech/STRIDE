"""Backwards-compatibility bridge for legacy 'src' imports.

DEPRECATION NOTICE:
Importing from 'src' is deprecated and will be removed in a future release.
Please use:
    import stride
    from stride.drift import BinaryErrorDriftDescriptor
    from stride.models import MODELS
    from stride.datasets import DATASETS
"""

import warnings

warnings.warn(
    "Importing from 'src' is deprecated. Please update imports to 'stride'.",
    DeprecationWarning,
    stacklevel=2,
)

from stride import *  # noqa: F401, F403
