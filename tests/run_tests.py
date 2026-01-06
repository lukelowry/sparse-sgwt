# -*- coding: utf-8 -*-
"""
Sparse Spectral Graph Wavelet Transform (SGWT)
----------------------------------------------
Author: Luke Lowery (lukel@tamu.edu)
File: tests/run_tests.py
Description: Master test script to discover and run all validation tests using pytest.
"""
import pytest
import sys
import os

def run_all_tests():
    """
    Discovers and runs all tests in the 'tests' directory using pytest.
    
    Ensures the project root is in the Python path for correct module imports.
    """
    # Get the directory where this script is located
    test_dir = os.path.abspath(os.path.dirname(__file__))
    root_dir = os.path.abspath(os.path.join(test_dir, '..'))

    # Ensure both root and tests are in path for imports
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    if test_dir not in sys.path:
        sys.path.insert(0, test_dir)
    
    # Run pytest. Pytest will automatically discover and run tests.
    # The exit code is 0 if all tests pass.
    # We pass the test directory to be explicit.
    # Pytest will discover tests based on `pytest.ini` or default discovery.
    # Any arguments passed to this script will be forwarded to pytest.
    args = sys.argv[1:]
    retcode = pytest.main(args)
    
    return retcode == pytest.ExitCode.OK

if __name__ == "__main__":
    if not run_all_tests():
        sys.exit(1)