"""Make repository-local imports work when running test runners as scripts."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
EXTERNAL_USE_CASES_DIR = ROOT.parent / "motiflets_use_cases"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
