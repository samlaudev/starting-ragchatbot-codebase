#!/usr/bin/env python3
"""
Development script to format code automatically
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
    """Format code automatically"""
    print("🎨 Formatting code...")

    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    commands = [
        (["uv", "run", "black", "."], "Black formatting"),
        (["uv", "run", "isort", "."], "Import sorting"),
    ]

    for cmd, desc in commands:
        if run_command(cmd, desc):
            print(f"✅ {desc} completed")
        else:
            print(f"❌ {desc} failed")
            return 1

    print("\n🎉 Code formatting completed!")
    return 0


if __name__ == "__main__":
    import os

    sys.exit(main())
