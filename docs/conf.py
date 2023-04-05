# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

import os
import sys
import datetime

sys.path.insert(0, os.path.abspath('.'))
path = os.path.abspath(os.path.dirname(__file__))


# -- Project information -----------------------------------------------------

project = 'prtools'
copyright = '2023, Andy Kee'
author = 'Andy Kee'
copyright = f'{datetime.date.today().year} Andy Kee'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

html_logo = '_static/logo.png'


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
html_theme_options = {
    "logo": {
        "text": "prtools"
    },
    "github_url": "https://github.com/andykee/prtools",
    "navbar_end": ["navbar-icon-links"],  # ["theme-switcher", "version-switcher", "navbar-icon-links"]
    "navbar_persistent": [],
    "footer_items": ["copyright"]
}

# ["search-field.html", "sidebar-nav-bs.html", "sidebar-ethical-ads.html"]
html_sidebars = {
    "**": ["search-field.html", "sidebar-ethical-ads.html"]
}

#html_theme_options = {
#    "navbar_end": ["navbar-icon-links.html",],
#}

html_show_sourcelink = False

html_scaled_image_link = False

# if true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = True

autodoc_default_options = {
    'member-order': 'alphabetical',
    'exclude-members': '__init__, __weakref__, __dict__, __module__',
    'undoc-members': False
}

autosummary_generate = True

