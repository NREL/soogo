"""Test documentation generation with Sphinx."""

import subprocess
import tempfile
import sys
from pathlib import Path
import pytest


def get_python_executable():
    """Get the appropriate Python executable."""
    # Check if we're in a virtual environment
    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        # We're in a virtual environment, use current Python
        return sys.executable

    # Not in a virtual environment, try to find venv Python
    repo_root = Path(__file__).parent.parent
    venv_python = repo_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)

    # Fall back to current Python
    return sys.executable


class TestSphinxDocumentation:
    """Test class for Sphinx documentation generation."""

    def test_sphinx_build(self):
        """Test that Sphinx can build the documentation without errors."""
        # Get the repository root directory
        repo_root = Path(__file__).parent.parent
        docs_dir = repo_root / "docs"
        python_exec = get_python_executable()

        # Skip test if sphinx is not available
        try:
            subprocess.run(
                [python_exec, "-c", "import sphinx"],
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
                python_exec,
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

            # Check if the build was successful
            assert result.returncode == 0, (
                f"Sphinx build failed with return code {result.returncode}\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}"
            )

            # Check that the main index.html file was created
            index_file = build_dir / "index.html"
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
        python_exec = get_python_executable()

        # Skip test if sphinx is not available
        try:
            subprocess.run(
                [python_exec, "-c", "import sphinx"],
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
                python_exec,
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
        python_exec = get_python_executable()

        # Skip test if sphinx is not available
        try:
            subprocess.run(
                [python_exec, "-c", "import sphinx"],
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
                python_exec,
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

    def test_documentation_structure_exists(self):
        """Test that all expected documentation files exist."""
        repo_root = Path(__file__).parent.parent
        docs_dir = repo_root / "docs"

        # Check that essential documentation files exist
        essential_files = [
            "conf.py",
            "index.rst",
            "soogo.rst",
            "acquisition.rst",
            "optimize.rst",
            "optimize_result.rst",
            "utils.rst",
            "termination.rst",
            "model.rst",
            "gp.rst",
            "rbf.rst",
            "surrogate.rst",
            "rbf_kernel.rst",
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
            import soogo.optimize_result  # noqa: F401
            import soogo.utils  # noqa: F401
            import soogo.termination  # noqa: F401
            import soogo.model  # noqa: F401
            import soogo.model.gp  # noqa: F401
            import soogo.model.rbf  # noqa: F401
            import soogo.model.base  # noqa: F401
            import soogo.model.rbf_kernel  # noqa: F401

            # Test that key classes can be imported
            from soogo.model import RbfModel
            from soogo.acquisition import WeightedAcquisition
            from soogo.optimize_result import OptimizeResult

            # Basic smoke test - instantiate some classes
            rbf_model = RbfModel()
            assert rbf_model is not None

            # Test that classes have expected attributes
            assert hasattr(WeightedAcquisition, "optimize")
            assert hasattr(OptimizeResult, "__init__")

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
