from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APPS = ["cargo", "jpetstore", "daytrader"]


def run(*args: str) -> None:
    print(f"\n$ python {' '.join(args)}")
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


if __name__ == "__main__":
    run("mvaa/tests/RQ1/bc_penalty_analysis.py")

    for app in APPS:
        run("mvaa/tests/RQ1/data_penalty_analysis.py", "--app", app)

    run("mvaa/tests/RQ1/decomposition_space.py")

    run("mvaa/tests/RQ1/discriminative_power.py")

    run("mvaa/tests/RQ2/partial_spearman.py")

    run("mvaa/tests/RQ3/questionnaire_results.py")

    run("mvaa/tests/graphics/graphics.py")

    print("\nDone. Outputs written to results/RQ1, results/RQ2, results/RQ3, results/graphics.")
