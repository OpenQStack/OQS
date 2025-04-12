
"""
OpenQStack
==========

A modular, extensible platform for learning, simulating, and prototyping
quantum error correction. Includes visualizations, noise models, and
education-friendly interfaces—think of it as the Arduino of quantum control.
"""

# --- Metadata ---
__version__ = "0.1.0"
__author__ = "Jaebum Eric Kim"
__email__ = "erickim1492@gmail.com"
__license__ = "MIT"
__url__ = "https://github.com/yourusername/openqstack"

# --- Top-level Imports ---
from .qec import BitFlipCode, ErrorChannel, BitFlipChannel, DepolarizingChannel, PhaseFlipChannel
from .visualize import show_state, plot_probabilities, plot_bloch

# --- Public API ---
__all__ = [
    "BitFlipCode",
    "ErrorChannel",
    "BitFlipChannel",
    "DepolarizingChannel",
    "PhaseFlipChannel",
    "show_state",
    "plot_probabilities",
    "plot_bloch",
]
