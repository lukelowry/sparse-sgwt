# -*- coding: utf-8 -*-
"""
Sparse Spectral Graph Wavelet Transform (SGWT)
----------------------------------------------
Author: Luke Lowery (lukel@tamu.edu)
File: sgwt/tests/run_tests.py
Description: Convenience script to run tests from a source checkout.
"""
import pytest
import sys
import os

def run_all_tests():
    """
    Discovers and runs all tests, returning True if all tests pass.

    This function is designed to be called when running this script directly
    from a source checkout (e.g., `python -m sgwt.tests.run_tests`). It
    ensures the project's root directory is in `sys.path` so that the `sgwt`
    package can be imported correctly without being installed. It then
    invokes pytest, which handles test discovery based on `pytest.ini`.

    Returns:
        bool: True if all tests passed, False otherwise.
    """
    # Get the directory where this script is located (sgwt/tests)
    test_dir = os.path.abspath(os.path.dirname(__file__))
    # The project root is two levels up from sgwt/tests
    root_dir = os.path.abspath(os.path.join(test_dir, '..', '..'))

    # Ensure the project root is in the path for correct module discovery
    # when running from a source checkout.
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    
    # Run pytest. Pytest will automatically discover and run tests based on
    # `pytest.ini`. Any arguments passed to this script are forwarded to pytest.
    args = sys.argv[1:]
    retcode = pytest.main(args)
    
    return retcode == pytest.ExitCode.OK

if __name__ == "__main__":
    # This allows running tests via `python -m sgwt.tests.run_tests`
    if not run_all_tests():
        sys.exit("Tests failed.")
