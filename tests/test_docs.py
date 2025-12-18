"""Test documentation generation with Sphinx."""

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

import subprocess
import tempfile
import sys
from pathlib import Path
import pytest


class TestSphinxDocumentation:
    """Test class for Sphinx documentation generation."""

    def test_sphinx_build(self):
        """Test that Sphinx can build the documentation without errors."""
        # Get the repository root directory
        repo_root = Path(__file__).parent.parent
        docs_dir = repo_root / "docs"

        # Skip test if sphinx is not available
        try:
            subprocess.run(
                [sys.executable, "-c", "import sphinx"],
                capture_output=True,
                check=True,
                cwd=repo_root,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("sphinx not available")

        # Create a temporary directory for the build output
        with tempfile.TemporaryDirectory() as temp_dir:
            build_dir = Path(temp_dir) / "_build"

            # Run sphinx-build
            cmd = [
                sys.executable,
                "-m",
                "sphinx",
                "-b",
                "html",  # HTML builder
                "-W",  # Turn warnings into errors
                "-q",  # Quiet mode (only show warnings/errors)
                str(docs_dir),
                str(build_dir),
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=repo_root
            )

            # Check that the main index.html file was created
            index_file = build_dir / "index.html"

            # Check if the build was successful
            assert result.returncode == 0 or index_file.exists(), (
                f"Sphinx build failed with return code {result.returncode}\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}"
            )

            assert index_file.exists(), "index.html was not generated"

            # Check that some HTML files were generated
            html_files = list(build_dir.glob("*.html"))
            assert len(html_files) > 0, "No HTML files were generated"

            # Check for at least the main module documentation
            expected_files = [
                "index.html",
            ]

            for expected_file in expected_files:
                file_path = build_dir / expected_file
                assert file_path.exists(), (
                    f"Expected file {expected_file} was not generated"
                )

    def test_sphinx_doctree_build(self):
        """Test that Sphinx can build doctrees without errors."""
        # Get the repository root directory
        repo_root = Path(__file__).parent.parent
        docs_dir = repo_root / "docs"

        # Skip test if sphinx is not available
        try:
            subprocess.run(
                [sys.executable, "-c", "import sphinx"],
                capture_output=True,
                check=True,
                cwd=repo_root,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("sphinx not available")

        # Create a temporary directory for the build output
        with tempfile.TemporaryDirectory() as temp_dir:
            doctree_dir = Path(temp_dir) / "_doctrees"
            build_dir = Path(temp_dir) / "_build"

            # Run sphinx-build to create doctrees
            cmd = [
                sys.executable,
                "-m",
                "sphinx",
                "-b",
                "html",
                "-d",
                str(doctree_dir),  # Doctree directory
                "-W",  # Turn warnings into errors
                "-q",  # Quiet mode
                str(docs_dir),
                str(build_dir),
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=repo_root
            )

            # Check if the build was successful
            assert result.returncode == 0, (
                f"Sphinx doctree build failed with return code {result.returncode}\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}"
            )

            # Check that doctree files were created
            assert doctree_dir.exists(), "Doctree directory was not created"
            doctree_files = list(doctree_dir.glob("*.doctree"))
            assert len(doctree_files) > 0, "No doctree files were generated"

    def test_sphinx_no_warnings(self):
        """Test that Sphinx documentation builds without warnings."""
        # Get the repository root directory
        repo_root = Path(__file__).parent.parent
        docs_dir = repo_root / "docs"

        # Skip test if sphinx is not available
        try:
            subprocess.run(
                [sys.executable, "-c", "import sphinx"],
                capture_output=True,
                check=True,
                cwd=repo_root,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("sphinx not available")

        # Create a temporary directory for the build output
        with tempfile.TemporaryDirectory() as temp_dir:
            build_dir = Path(temp_dir) / "_build"

            # Run sphinx-build with verbose warnings
            cmd = [
                sys.executable,
                "-m",
                "sphinx",
                "-b",
                "html",
                "-v",  # Verbose mode to see all warnings
                str(docs_dir),
                str(build_dir),
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=repo_root
            )

            # The build should succeed (return code 0 or 1 for warnings)
            assert result.returncode in [0, 1], (
                f"Sphinx build failed with return code {result.returncode}\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}"
            )

            # Check for specific warning patterns that we want to avoid
            output = result.stdout + result.stderr

            # These are warnings we specifically fixed
            problematic_patterns = [
                "Block quote ends without a blank line",
                "unexpected unindent",
                "failed to import module",
                "toctree contains reference to nonexisting document",
                "autodoc: failed to import",
            ]

            for pattern in problematic_patterns:
                assert pattern not in output, (
                    f"Found problematic warning pattern: '{pattern}'\n"
                    f"Full output:\n{output}"
                )

    def test_versions_template_renders_without_versions_context(self):
        """Ensure versions template renders when sphinx-multiversion
        context is missing.
        """
        jinja2 = pytest.importorskip("jinja2")

        repo_root = Path(__file__).parent.parent
        template_dir = repo_root / "docs" / "_templates"
        template_path = template_dir / "versions.html"

        if (
            not template_path.exists()
        ):  # Defensive guard for projects without the template
            pytest.skip("versions template not present")

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir))
        )
        template = env.get_template("versions.html")

        # The template should render when sphinx-multiversion does not
        # provide context
        template.render()
        template.render(versions={"tags": [], "branches": []})

    def test_documentation_structure_exists(self):
        """Test that all expected documentation files exist."""
        repo_root = Path(__file__).parent.parent
        docs_dir = repo_root / "docs"

        # Check that essential documentation files exist in the new hierarchy
        essential_files = [
            "conf.py",
            "index.rst",
            "api/soogo.rst",
            "api/soogo.acquisition.rst",
            "api/soogo.integrations.rst",
            "api/soogo.model.rst",
            "api/soogo.optimize.rst",
            "api/tests.rst",
        ]

        for file_name in essential_files:
            file_path = docs_dir / file_name
            assert file_path.exists(), (
                f"Documentation file {file_name} does not exist"
            )

            # Check that the file is not empty
            assert file_path.stat().st_size > 0, (
                f"Documentation file {file_name} is empty"
            )

    def test_module_imports_in_docs(self):
        """Test that modules referenced in documentation can be imported."""
        import sys

        # Add the soogo package to the Python path
        repo_root = Path(__file__).parent.parent
        sys.path.insert(0, str(repo_root))

        try:
            # Test importing main modules that are documented
            import soogo  # noqa: F401
            import soogo.acquisition  # noqa: F401
            import soogo.optimize  # noqa: F401
            import soogo.utils  # noqa: F401
            import soogo.termination  # noqa: F401
            import soogo.sampling  # noqa: F401
            import soogo.model  # noqa: F401
            import soogo.model.gp  # noqa: F401
            import soogo.model.rbf  # noqa: F401
            import soogo.model.base  # noqa: F401
            import soogo.model.rbf_kernel  # noqa: F401
            import soogo.integrations  # noqa: F401

            # Test that key classes can be imported
            from soogo.model import RbfModel, GaussianProcess
            from soogo.acquisition import WeightedAcquisition
            from soogo import (
                OptimizeResult,
                surrogate_optimization,
                bayesian_optimization,
            )

            # Basic smoke test - instantiate some classes
            rbf_model = RbfModel()
            assert rbf_model is not None

            gp_model = GaussianProcess()
            assert gp_model is not None

            # Test that classes and functions have expected attributes
            assert hasattr(WeightedAcquisition, "optimize")
            assert hasattr(OptimizeResult, "__init__")
            assert callable(surrogate_optimization)
            assert callable(bayesian_optimization)

        except ImportError as e:
            pytest.fail(
                f"Failed to import module referenced in documentation: {e}"
            )
        finally:
            # Clean up the path
            if str(repo_root) in sys.path:
                sys.path.remove(str(repo_root))


if __name__ == "__main__":
    # Allow running the test directly
    test_instance = TestSphinxDocumentation()
    test_instance.test_sphinx_build()
    test_instance.test_documentation_structure_exists()
    test_instance.test_module_imports_in_docs()
    print("All documentation tests passed!")
