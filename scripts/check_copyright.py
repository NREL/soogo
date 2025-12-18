#!/usr/bin/env python3
"""Check that Python files have proper copyright headers."""

# Copyright (c) 2025 Alliance for Energy Innovation, LLC

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

__authors__ = ["Weslley S. Pereira"]

import sys
import re
from pathlib import Path


def has_copyright_header(file_path: Path) -> bool:
    """Check if a Python file has a copyright header.

    Args:
        file_path: Path to the Python file to check

    Returns:
        True if the file has a copyright header, False otherwise
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

    # Look for copyright notice in the first 50 lines
    lines = content.split("\n")[:50]
    first_section = "\n".join(lines)

    # Check for copyright notice
    copyright_pattern = r"Copyright \(c\) \d{4}"
    has_copyright = bool(
        re.search(copyright_pattern, first_section, re.IGNORECASE)
    )

    # Check for GPL license mention
    has_license = "GNU General Public License" in first_section

    return has_copyright and has_license


def should_check_file(file_path: Path) -> bool:
    """Determine if a file should be checked for copyright headers.

    Args:
        file_path: Path to the file

    Returns:
        True if the file should be checked, False otherwise
    """
    # Skip __init__.py files that are just imports
    if file_path.name == "__init__.py":
        try:
            content = file_path.read_text(encoding="utf-8")
            # If it's very short or just has imports, skip it
            lines = [
                line.strip()
                for line in content.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]
            if len(lines) <= 3:
                return False
        except Exception:
            pass

    # Skip test files in certain cases (optional - you can remove this)
    # if 'test' in file_path.name:
    #     return False

    return True


def main():
    """Main function to check copyright headers in staged files."""
    # Get files to check from command line arguments
    files_to_check = sys.argv[1:]

    if not files_to_check:
        print("No files to check")
        return 0

    failed_files = []

    for file_path_str in files_to_check:
        file_path = Path(file_path_str)

        # Only check Python files
        if file_path.suffix != ".py":
            continue

        # Skip files in certain directories
        if any(part.startswith(".") for part in file_path.parts):
            continue

        if not should_check_file(file_path):
            continue

        if not has_copyright_header(file_path):
            failed_files.append(file_path)

    if failed_files:
        print("\n❌ The following files are missing copyright headers:")
        for file_path in failed_files:
            print(f"  - {file_path}")
        print("\nPlease add a copyright header to these files.")
        print("See .copyright_header_template.txt for the required format.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
