Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[0.3.5] - 2026-01-10
--------------------

Changed
~~~~~~~
- Expanded test suite to full code and branch coverage, including all edge cases and defensive branches.
- Tests to do cover KLU wrapper, because it is not used in this version.

[0.3.4] - 2026-01-06
--------------------

Changed
~~~~~~~
- Migrated the entire test suite from `unittest` to `pytest` for a more modern, readable, and feature-rich testing framework. The tests are now included as an installable sub-package `sgwt.tests`.

Fixed
~~~~~
- Resolved an issue where `DyConvolve.addbranch` would cause an unhandled error for negative branch weights. It now raises a descriptive `ValueError`.

[0.3.3] - 2026-01-05
------------

Added
~~~~~
- New documentation section for Chebyshev approximation examples and benchmarks.
- Lazy loading for built-in resources (Laplacians, Signals, Kernels) to improve import time and reduce memory usage.

Fixed
~~~~~
- Offset ``d_a`` of Kernel Fitted approximation correctly implemented
- Documentation typos referring to ``VFKern`` instead of ``VFKernel``.

[0.3.2] - 2026-01-03
--------------------

Added
~~~~~
- ``CHANGELOG.rst`` to track project history.
- ``CITATION.cff`` for easier citation.

Changed
~~~~~~~
- Renamed internal ``io`` module to ``util`` for better semantic clarity.
- Renamed ``VFKern`` to ``VFKernel`` for clarity and consistency.
- Improved ``README.rst`` layout, usage example, and author links.
- Enhanced documentation for the data library (``library_data.rst``) with code snippets.
- Corrected and clarified kernel JSON format documentation (``library_json.rst``).
- Standardized API and documentation for consistency (e.g., `n_vertices`, `np.ndarray` links, default parameter formatting).
- Switched documentation parser from `numpydoc` to `sphinx.ext.napoleon` for improved styling and stability.

Fixed
~~~~~
- Updated documentation and packaging to reflect Windows-only compatibility due to the pre-compiled CHOLMOD ``.dll``.

[0.3.1] - 2026-01-01
--------------------

Added
~~~~~
- First stable public release of the ``sgwt`` package.
- Comprehensive documentation, usage examples, and testing suite.