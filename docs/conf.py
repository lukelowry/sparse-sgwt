import sgwt
import importlib.metadata as importlib_metadata

extensions = [
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.inheritance_diagram"
]

extensions.append("sphinx.ext.autodoc")
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "member-order": "groupwise",
}

# Better API formatting
autoclass_content = "both"        # Include __init__ docstring in class description
autodoc_typehints = "description" # Move type hints to parameter descriptions
add_module_names = False          # Don't show full module path (e.g. sgwt.static.Convolve -> Convolve)

extensions.append("sphinx.ext.intersphinx")
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
}

extensions.append("sphinx_copybutton")

extensions.append("numpydoc")
numpydoc_show_class_members = False
numpydoc_use_plots = True  # Add the plot directive whenever mpl is imported.
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    # Common aliases
    "np": "numpy",
    "csc_matrix": "scipy.sparse.csc_matrix",

    # Your project's types
    "VFKern": "sgwt.io.VFKern",

    # Python built-ins and typing module
    "optional": ":py:obj:`~typing.Optional`",
    "union": ":py:obj:`~typing.Union`",
    "list": ":py:class:`list`",
    "dict": ":py:class:`dict`",
    "bool": ":py:class:`bool`",
    "int": ":py:class:`int`",
    "float": ":py:class:`float`",
}


exclude_patterns = ["_build"]
source_suffix = ".rst"
master_doc = "index"

project = "Sparse SGWT"
copyright = "2024, Luke Lowery"
author = "Luke Lowery"
version = importlib_metadata.version("sgwt")
release = version

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 2,
}

autodoc_mock_imports = ["ctypes"]