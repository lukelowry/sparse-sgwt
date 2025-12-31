# -*- coding: utf-8 -*-
"""
Sparse Spectral Graph Wavelet Transform (SGWT)
----------------------------------------------
Author: Luke Lowery (lukel@tamu.edu)
File: tests/run_tests.py
Description: Master test script to discover and run all validation tests.
"""
import unittest
import sys
import os

class BriefColoredResult(unittest.TextTestResult):
    def startTest(self, test):
        super().startTest(test)

    def _print_status(self, color_code, status, test):
        # Handle discovery errors or standard tests
        test_str = str(test)
        if "ModuleImportError" in test_str or "_FailedTest" in test_str:
            desc = f"Load Error: {test_str.split('.')[-1].split(' ')[0]}"
        else:
            desc = test.shortDescription() or test_str.split(' ')[0]
        self.stream.writeln(f"[\033[{color_code}m{status:^7}\033[0m] {desc}")

    def addSuccess(self, test):
        super().addSuccess(test)
        self._print_status("92", "PASSED", test)

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._print_status("91", "FAILED", test)

    def addError(self, test, err):
        super().addError(test, err)
        self._print_status("91", "ERROR", test)

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._print_status("93", "SKIPPED", test)

def run_all_tests():
    # Get the directory where this script is located
    test_dir = os.path.abspath(os.path.dirname(__file__))
    root_dir = os.path.abspath(os.path.join(test_dir, '..'))

    # Ensure both root and tests are in path for imports
    if root_dir not in sys.path: sys.path.insert(0, root_dir)
    if test_dir not in sys.path: sys.path.insert(0, test_dir)
    
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern='test_*.py')
    
    if suite.countTestCases() == 0:
        print(f"Error: No tests found in {test_dir}")
        return False

    # Use our custom result class for colored, descriptive output
    runner = unittest.TextTestRunner(resultclass=BriefColoredResult, verbosity=0)
    result = runner.run(suite)

    # Summary
    passed  = result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
    failed  = len(result.failures) + len(result.errors)
    skipped = len(result.skipped)

    print("\n" + "═"*45)
    print(f"║ {'TEST RESULTS SUMMARY':^41} ║")
    print("╠" + "═"*43 + "╣")
    print(f"║ Total Tests Run: {result.testsRun:>24} ║")
    print(f"║ \033[92mPassed:          {passed:>24}\033[0m ║")
    print(f"║ \033[91mFailed/Errors:   {failed:>24}\033[0m ║")
    print(f"║ \033[93mSkipped:         {skipped:>24}\033[0m ║")
    print("╠" + "═"*43 + "╣")
    
    status_text = "PASSED" if result.wasSuccessful() else "FAILED"
    color = "92" if result.wasSuccessful() else "91"
    print(f"║ OVERALL STATUS:      \033[{color}m{status_text:>20}\033[0m ║")
    print("╚" + "═"*43 + "╝\n")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(not success)