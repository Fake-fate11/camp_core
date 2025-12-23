try:
    from .base import *  # noqa: F401,F403
except ImportError:
    pass

from .linear_head import LinearMappingHead

__all__ = [
    "LinearMappingHead",
]

