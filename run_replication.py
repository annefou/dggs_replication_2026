#!/usr/bin/env python3
"""
Run the ORIGINAL benchmark code from the paper authors.

This script is a thin wrapper around the benchmark code from:
https://github.com/manaakiwhenua/dggsBenchmarks/releases/tag/v1.1.1

The original code is cloned into /app/original_benchmarks by the Dockerfile.
This wrapper script ensures we run the EXACT same code the paper authors used,
which is the essence of reproducibility.

Paper: Law & Ardo (2024)
"Using a discrete global grid system for a scalable, interoperable, 
and reproducible system of land-use mapping"
https://doi.org/10.1080/20964471.2024.2429847
"""

import os
import sys
import subprocess
from pathlib import Path

# Path to the original benchmark code
ORIGINAL_BENCHMARKS_DIR = Path("/app/original_benchmarks")


def list_original_files():
    """List what's in the original benchmark repository."""
    print("=" * 60)
    print("ORIGINAL BENCHMARK CODE (from paper authors)")
    print("=" * 60)
    
    if not ORIGINAL_BENCHMARKS_DIR.exists():
        print(f"ERROR: {ORIGINAL_BENCHMARKS_DIR} not found!")
        print("The original benchmark repo should be cloned by the Dockerfile.")
        return False
    
    print(f"\nContents of {ORIGINAL_BENCHMARKS_DIR}:")
    for item in sorted(ORIGINAL_BENCHMARKS_DIR.iterdir()):
        if item.is_dir():
            print(f"  📁 {item.name}/")
            # Show first-level contents of directories
            for subitem in sorted(item.iterdir())[:5]:
                print(f"      {subitem.name}")
            if len(list(item.iterdir())) > 5:
                print(f"      ... and {len(list(item.iterdir())) - 5} more")
        else:
            print(f"  📄 {item.name}")
    
    # Check for common benchmark script names
    print("\n" + "=" * 60)
    print("LOOKING FOR BENCHMARK SCRIPTS")
    print("=" * 60)
    
    possible_scripts = [
        "benchmark.py",
        "run_benchmark.py", 
        "run_benchmarks.py",
        "main.py",
        "vector_benchmark.py",
        "raster_benchmark.py",
    ]
    
    found_scripts = []
    for root, dirs, files in os.walk(ORIGINAL_BENCHMARKS_DIR):
        for f in files:
            if f.endswith('.py'):
                full_path = Path(root) / f
                print(f"  Found Python file: {full_path.relative_to(ORIGINAL_BENCHMARKS_DIR)}")
                found_scripts.append(full_path)
    
    return found_scripts


def show_readme():
    """Show the README from the original repo if it exists."""
    readme_paths = [
        ORIGINAL_BENCHMARKS_DIR / "README.md",
        ORIGINAL_BENCHMARKS_DIR / "readme.md",
        ORIGINAL_BENCHMARKS_DIR / "README.rst",
        ORIGINAL_BENCHMARKS_DIR / "README",
    ]
    
    for readme in readme_paths:
        if readme.exists():
            print("\n" + "=" * 60)
            print(f"README ({readme.name})")
            print("=" * 60)
            with open(readme) as f:
                content = f.read()
                # Show first 2000 characters
                if len(content) > 2000:
                    print(content[:2000])
                    print(f"\n... (truncated, {len(content)} total characters)")
                else:
                    print(content)
            return True
    
    print("\nNo README found in original benchmark repo")
    return False


def run_original_benchmark(script_name: str = None):
    """
    Attempt to run the original benchmark code.
    
    If script_name is not provided, will try to auto-detect the main script.
    """
    os.chdir(ORIGINAL_BENCHMARKS_DIR)
    
    # Add the original benchmarks dir to Python path
    sys.path.insert(0, str(ORIGINAL_BENCHMARKS_DIR))
    
    if script_name:
        script_path = ORIGINAL_BENCHMARKS_DIR / script_name
        if script_path.exists():
            print(f"\nRunning: {script_path}")
            result = subprocess.run([sys.executable, str(script_path)], 
                                    capture_output=False)
            return result.returncode
        else:
            print(f"ERROR: Script not found: {script_path}")
            return 1
    
    # Auto-detect main script
    for possible in ["benchmark.py", "run_benchmark.py", "main.py"]:
        script_path = ORIGINAL_BENCHMARKS_DIR / possible
        if script_path.exists():
            print(f"\nAuto-detected main script: {possible}")
            print(f"Running: {script_path}")
            result = subprocess.run([sys.executable, str(script_path)], 
                                    capture_output=False)
            return result.returncode
    
    print("\nCould not auto-detect main benchmark script.")
    print("Please specify the script name manually.")
    return 1


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run the original DGGS benchmark code from the paper authors"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List files in the original benchmark repository"
    )
    parser.add_argument(
        "--readme", "-r",
        action="store_true", 
        help="Show the README from the original repo"
    )
    parser.add_argument(
        "--run", "-R",
        nargs="?",
        const="auto",
        metavar="SCRIPT",
        help="Run a benchmark script (or auto-detect if not specified)"
    )
    parser.add_argument(
        "--explore", "-e",
        action="store_true",
        help="Explore the original repo (list files + show README)"
    )
    
    args = parser.parse_args()
    
    # Default: explore
    if not any([args.list, args.readme, args.run, args.explore]):
        args.explore = True
    
    if args.explore or args.list:
        scripts = list_original_files()
    
    if args.explore or args.readme:
        show_readme()
    
    if args.run:
        if args.run == "auto":
            return run_original_benchmark()
        else:
            return run_original_benchmark(args.run)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
