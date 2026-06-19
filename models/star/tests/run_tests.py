import subprocess
import sys
import os
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
TEST_FILES = [
    "test_head_first.py",
    "test_head_first_classifier.py",
    "test_number_ner.py",
    "test_star_scorer.py",
]


def run_test(filename: str) -> int:
    path = TEST_DIR / filename
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    print(f"\n===== {filename} =====", flush=True)
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(TEST_DIR.parents[2]),
        env=env,
    )
    return result.returncode


def main() -> int:
    failed = []
    for filename in TEST_FILES:
        code = run_test(filename)
        if code != 0:
            failed.append((filename, code))

    print("\n===== SUMMARY =====", flush=True)
    if not failed:
        print("All STAR tests passed.", flush=True)
        return 0

    for filename, code in failed:
        print(f"{filename} failed with exit code {code}.", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
