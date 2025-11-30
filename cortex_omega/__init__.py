try:
    from ._version import version as __version__
except ImportError:
    # Fallback for local development if package isn't installed
    __version__ = "0.0.0-dev"

from .api.client import Cortex
from .core.errors import EpistemicVoidError
