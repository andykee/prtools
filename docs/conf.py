# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

import os
import sys
import datetime

sys.path.insert(0, os.path.abspath('../'))
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
              'sphinx.ext.viewcode',
              'sphinx_remove_toctrees',
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
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "article_header_start": [],
    "secondary_sidebar_items": [],
    "footer_start": ["copyright"],
    "footer_end": [],
    "show_prev_next": False,
    
}

html_sidebars = {
  "index": ["search-field.html"],
  "generated/*": ["search-field.html", "sidebar-nav.html"]
}

html_show_sourcelink = False

html_scaled_image_link = False

# if true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

autodoc_default_options = {
    'member-order': 'alphabetical',
    'exclude-members': '__init__, __weakref__, __dict__, __module__',
    'undoc-members': False
}

autosummary_generate = True

remove_from_toctrees = ["generated/*"]