"""Make `src/` importable without requiring an editable install.

`pip install -e .` is still the recommended workflow (see README), but this
keeps `pytest` working out of the box for anyone who just clones the repo
and installs requirements.txt.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
