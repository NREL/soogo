"""Test suite to ensure all example files run successfully."""

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

import sys
import subprocess
import pytest
import json
from pathlib import Path
from contextlib import contextmanager


class TestExamples:
    """Test class to verify that all example files run successfully."""

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        cls.examples_dir = Path(__file__).parent.parent / "examples"
        cls.legacy_opt_dir = cls.examples_dir / "legacy_opt"

        # Ensure we're in the right directory
        assert cls.examples_dir.exists(), (
            f"Examples directory not found: {cls.examples_dir}"
        )

    @contextmanager
    def _temporary_kernel(self, base_name: str = "soogo-test"):
        """Context manager for temporary kernel creation and cleanup."""
        import time

        # Create unique kernel name
        kernel_name = f"{base_name}-{int(time.time())}-{id(self)}"

        try:
            # Create kernel
            cmd = [
                sys.executable,
                "-m",
                "ipykernel",
                "install",
                "--user",
                f"--name={kernel_name}",
                f"--display-name=Python ({kernel_name})",
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            yield kernel_name

        finally:
            # Always clean up
            try:
                cmd = [
                    sys.executable,
                    "-m",
                    "jupyter",
                    "kernelspec",
                    "remove",
                    kernel_name,
                    "-f",
                ]
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError:
                print(f"Warning: Could not remove kernel {kernel_name}")

    def _run_notebook(self, notebook_path: Path, timeout: int = 300) -> bool:
        """Run a Jupyter notebook and check if it executes successfully.

        :param notebook_path: Path to the notebook file
        :param timeout: Maximum execution time in seconds
        :return: True if notebook runs successfully
        """
        with self._temporary_kernel() as kernel_name:
            try:
                # Use nbconvert to execute the notebook
                cmd = [
                    sys.executable,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "--inplace",
                    f"--ExecutePreprocessor.timeout={timeout}",
                    f"--ExecutePreprocessor.kernel_name={kernel_name}",
                    str(notebook_path),
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=self.examples_dir,
                    timeout=timeout + 60,  # Add buffer time
                )

                if result.returncode != 0:
                    print(f"STDOUT: {result.stdout}")
                    print(f"STDERR: {result.stderr}")
                    return False

                return True

            except subprocess.TimeoutExpired:
                print(
                    f"Notebook {notebook_path} timed out after {timeout} seconds"
                )
                return False
            except Exception as e:
                print(f"Error running notebook {notebook_path}: {e}")
                return False

    def _run_python_script(
        self, script_path: Path, timeout: int = 300
    ) -> bool:
        """Run a Python script and check if it executes successfully.

        :param script_path: Path to the Python script
        :param timeout: Maximum execution time in seconds
        :return: True if script runs successfully
        """
        try:
            cmd = [sys.executable, str(script_path)]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=script_path.parent,
                timeout=timeout,
            )

            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return False

            return True

        except subprocess.TimeoutExpired:
            print(f"Script {script_path} timed out after {timeout} seconds")
            return False
        except Exception as e:
            print(f"Error running script {script_path}: {e}")
            return False

    @contextmanager
    def _backup_and_restore_notebook(self, notebook_path: Path):
        """Context manager to backup and restore notebook after execution."""
        import shutil

        backup_path = notebook_path.with_suffix(".ipynb.backup")

        try:
            # Create backup
            shutil.copy2(notebook_path, backup_path)
            yield
        finally:
            # Restore from backup
            if backup_path.exists():
                shutil.move(backup_path, notebook_path)

    def test_gosac_notebook(self):
        """Test that gosac.ipynb runs successfully."""
        notebook_path = self.examples_dir / "gosac.ipynb"

        with self._backup_and_restore_notebook(notebook_path):
            success = self._run_notebook(notebook_path, timeout=180)
            assert success, f"Failed to run {notebook_path}"

    def test_sampling_notebook(self):
        """Test that sampling.ipynb runs successfully."""
        notebook_path = self.examples_dir / "sampling.ipynb"

        with self._backup_and_restore_notebook(notebook_path):
            success = self._run_notebook(notebook_path, timeout=120)
            assert success, f"Failed to run {notebook_path}"

    def test_socemo_notebook(self):
        """Test that socemo.ipynb runs successfully."""
        notebook_path = self.examples_dir / "socemo.ipynb"

        with self._backup_and_restore_notebook(notebook_path):
            success = self._run_notebook(notebook_path, timeout=240)
            assert success, f"Failed to run {notebook_path}"

    def test_optimization_notebook(self):
        """Test that optimization.ipynb runs successfully."""
        notebook_path = self.examples_dir / "optimization.ipynb"

        with self._backup_and_restore_notebook(notebook_path):
            success = self._run_notebook(notebook_path, timeout=180)
            assert success, f"Failed to run {notebook_path}"

    def test_optimization_program_1(self):
        """Test that optimization_program_1.py runs successfully."""
        script_path = self.legacy_opt_dir / "optimization_program_1.py"

        # Check if the script exists
        if not script_path.exists():
            pytest.skip(f"Script not found: {script_path}")

        success = self._run_python_script(script_path, timeout=300)
        assert success, f"Failed to run {script_path}"

    def test_optimization_program_2(self):
        """Test that optimization_program_2.py runs successfully."""
        script_path = self.legacy_opt_dir / "optimization_program_2.py"

        # Check if the script exists
        if not script_path.exists():
            pytest.skip(f"Script not found: {script_path}")

        success = self._run_python_script(script_path, timeout=300)
        assert success, f"Failed to run {script_path}"

    @pytest.mark.parametrize(
        "notebook_name",
        [
            "gosac.ipynb",
            "sampling.ipynb",
            "socemo.ipynb",
            "optimization.ipynb",
            "batch_sampling.ipynb",
            "optimization_edges.ipynb",
        ],
    )
    def test_notebook_syntax(self, notebook_name):
        """Test that notebooks have valid JSON syntax."""
        notebook_path = self.examples_dir / notebook_name

        if not notebook_path.exists():
            pytest.skip(f"Notebook not found: {notebook_path}")

        try:
            with open(notebook_path, "r") as f:
                json.load(f)
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON in {notebook_name}: {e}")

    @pytest.mark.parametrize(
        "script_name",
        ["optimization_program_1.py", "optimization_program_2.py"],
    )
    def test_script_syntax(self, script_name):
        """Test that Python scripts have valid syntax."""
        script_path = self.legacy_opt_dir / script_name

        if not script_path.exists():
            pytest.skip(f"Script not found: {script_path}")

        try:
            with open(script_path, "r") as f:
                compile(f.read(), script_path, "exec")
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {script_name}: {e}")

    def test_examples_import_soogo(self):
        """Test that key modules can be imported (basic smoke test)."""
        try:
            # Test basic imports - just import without assigning
            import soogo  # noqa: F401
            from soogo import gosac, socemo, surrogate_optimization  # noqa: F401
            from soogo import RbfModel  # noqa: F401
            from soogo import acquisition, sampling  # noqa: F401
            # Note: GaussianProcess import might fail in some environments
        except ImportError as e:
            pytest.fail(f"Failed to import required modules: {e}")


if __name__ == "__main__":
    # Allow running the test directly
    pytest.main([__file__, "-v"])
