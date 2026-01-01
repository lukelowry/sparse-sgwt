import os
import sys
sys.path.insert(0, os.path.abspath('..'))
#import sgwt

project = 'Sparse SGWT'
copyright = '2024, Luke Lowery'
author = 'Luke Lowery'
release = '0.3.0'
version = '0.3.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.inheritance_diagram',
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
}

exclude_patterns = ['_build']


html_theme_options = {
    "navigation_depth": 2,
}

# Mock imports if necessary (e.g. if C-extensions/DLLs cannot be built on RTD servers)
# autodoc_mock_imports = ['ctypes'] 