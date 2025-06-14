"""JAX Music Source Separation Toolkit (JAXMSST)

A JAX-based toolkit for music source separation using transformer models.
"""

__version__ = "0.1.0"
__author__ = "JAXMSST Team"
__email__ = ""
__description__ = "JAX Music Source Separation Toolkit"

# Import main modules
from . import infer
from . import train
from . import convert
from . import webui
from . import bsr_dataset
from . import make_dataset
from . import multihost_dataloading
from . import profiling

__all__ = [
    "infer",
    "train", 
    "convert",
    "webui",
    "bsr_dataset",
    "make_dataset",
    "multihost_dataloading",
    "profiling",
]