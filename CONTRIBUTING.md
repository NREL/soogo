# Contributing guidelines

Thank you for investing your precious time to contribute to this project. Please read the following guidelines before contributing.

## Code style

This project uses [Ruff](https://docs.astral.sh/ruff/) to enforce code style. Ruff is a wrapper around [Black](https://black.readthedocs.io/en/stable/) and [Flake8](https://flake8.pycqa.org/en/latest/). To use Ruff, run `ruff check` at the root of this repository to see if there are formatting fixes to be performed. Then run `ruff format` to format the code. Run `ruff --help` to see the available options.

## Pre-commit hooks

This project uses [pre-commit](https://pre-commit.com/) to automatically check code quality before commits. To set up pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

The pre-commit hooks will automatically:

- Check for copyright headers in Python files
- Remove trailing whitespace
- Fix end-of-file issues
- Validate YAML files
- Check for large files
- Run Ruff for code formatting

## Copyright headers

All Python source files must include a copyright header at the top of the file. The header should follow this format:

```python
"""Module docstring."""

# Copyright (c) 2025 Alliance for Sustainable Energy, LLC

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

__authors__ = ["Your Name"]
```

For files based on existing algorithms from academic papers, include the original authors in the `__credits__` variable at [soogo/\_\_init\_\_.py](soogo/__init__.py).
