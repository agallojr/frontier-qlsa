"""
Frontier QLSA - Quantum Linear System Algorithm solver
This package provides a namespace wrapper around the root-level modules.
"""

# Re-export modules from root level
import sys
from pathlib import Path

# Add parent directory to path to import root-level modules
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Import and re-export the main modules
from circuit_HHL import *
from func_HeleShaw import *
from func_matrix_vector import *
from func_qc import *
from solver import *

# Clean up namespace
del sys, Path, _root
