import sys
from pathlib import Path

# Keeps `pytest` working from a bare checkout with no install step.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
