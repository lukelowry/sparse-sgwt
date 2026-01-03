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
autodoc_typehints = "description"   # Show type hints in signatures, not parameter descriptions
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

# -- Workaround for numpydoc and autodoc_typehints ---------------------------
# This is a workaround to prevent numpydoc from rendering the 'Parameters'
# section, which would be redundant when `autodoc_typehints = 'description'`
# is used. Numpydoc will still process other sections like 'Returns' and
# 'Examples'.
# See: https://github.com/numpy/numpydoc/issues/215
def supress_numpydoc_parameters(app, what, name, obj, options, lines):
    if what not in ('function', 'method', 'class', 'attribute'):
        return

    # Find the start of the 'Parameters' section
    try:
        param_start_index = lines.index('Parameters')
    except ValueError:
        return

    # Check for the underline '----------'
    if len(lines) <= param_start_index + 1 or not lines[param_start_index + 1].strip().startswith('---'):
        return

    # Find the end of the 'Parameters' section
    param_end_index = len(lines)
    known_sections = ('Returns', 'Yields', 'Receives', 'Other Parameters', 'Attributes', 'Methods', 'See Also', 'Notes', 'Warnings', 'References', 'Examples')
    
    for i in range(param_start_index + 2, len(lines)):
        line_stripped = lines[i].strip()
        if line_stripped in known_sections and len(lines) > i + 1 and lines[i+1].strip().startswith(('-', '=')):
            param_end_index = i
            break

    del lines[param_start_index:param_end_index]

def setup(app):
    """Register the Sphinx hook."""
    app.connect('autodoc-process-docstring', supress_numpydoc_parameters)