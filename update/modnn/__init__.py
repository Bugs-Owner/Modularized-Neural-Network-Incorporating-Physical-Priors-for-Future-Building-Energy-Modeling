__version__ = "3.2.0"
__changelog__ = """
Version 1.0.1:
- Fixed import errors caused by missing submodules in PyPI build.
Version 1.0.2:
- Remove ylim when plot for better visualtion.
"""

from modnn.Config import get_config, describe
from modnn.utils import Mod

__all__ = ["get_config", "describe", "Mod"]
