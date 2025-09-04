#!/usr/bin/env python3
"""
Development script to run all code quality checks
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return True if successful"""
    print(f"\n=== {description} ===")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def main() -> int:
    """Run all quality checks"""
    print("🔍 Running code quality checks...")

    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)

    checks = [
        (["uv", "run", "black", "--check", "."], "Black formatting check"),
        (["uv", "run", "isort", "--check-only", "."], "Import sorting check"),
        (["uv", "run", "flake8", "--max-line-length=88", "."], "Flake8 linting"),
        (["uv", "run", "mypy", "."], "Type checking"),
    ]

    passed = 0
    total = len(checks)

    for cmd, desc in checks:
        if run_command(cmd, desc):
            passed += 1
            print(f"✅ {desc} passed")
        else:
            print(f"❌ {desc} failed")

    print("\n=== Summary ===")
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("🎉 All checks passed!")
        return 0
    else:
        print("💥 Some checks failed!")
        return 1


if __name__ == "__main__":
    import os

    sys.exit(main())
