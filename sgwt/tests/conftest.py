# -*- coding: utf-8 -*-
"""
pytest configuration file for the sgwt test suite.

This file customizes pytest's behavior to provide cleaner test reports.
"""

def pytest_collection_modifyitems(config, items):
    """
    Hook to modify test items after collection.

    This removes the test file path from the node ID of each test,
    resulting in a cleaner, less cluttered report. For example:
    
    FROM: sgwt/tests/test_cholesky.py::TestCholesky::test_...
    TO:   TestCholesky::test_...
    """
    for item in items:
        # Split on the first '::' and take the part after it.
        if '::' in item.nodeid:
            item._nodeid = item.nodeid.split('::', 1)[1]