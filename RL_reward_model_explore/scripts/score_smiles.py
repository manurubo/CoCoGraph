#!/usr/bin/env python3
"""Entry point for scoring SMILES."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference import main

if __name__ == "__main__":
    main()
