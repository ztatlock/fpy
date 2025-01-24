"""
Frontend for the FPy language.

This module contains a parser, syntax checker, and type checking
for the FPy language.
"""

from .decorator import fpy
from .fpyast import set_default_formatter
from .formatter import Formatter

set_default_formatter(Formatter())
