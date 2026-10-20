"""Gera todas as figuras em figures/ (rode a partir da raiz do repositório)."""

import pathlib
import runpy
import sys

here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(here))
sys.path.insert(0, str(here.parent))

for script in sorted(here.glob("fig*.py")):
    print(f"\n=== {script.name} ===")
    runpy.run_path(str(script), run_name="__main__")
