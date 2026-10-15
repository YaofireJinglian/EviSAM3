try:
    import mmcv
except ImportError:
    mmcv = None

from .version import __version__, version_info

__all__ = ['__version__', 'version_info']
