
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from . import analysis
from . import audio
from . import video
from . import plotting
