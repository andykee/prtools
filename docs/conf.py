# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

import os
import re
import sys
import datetime

sys.path.insert(0, os.path.abspath('../'))
path = os.path.abspath(os.path.dirname(__file__))

with open(os.path.normpath(os.path.join(path, '..', 'prtools', '__init__.py'))) as f:
    version = release = re.search("__version__ = '(.*?)'", f.read()).group(1)

# -- Project information -----------------------------------------------------

project = 'prtools'
author = 'Andy Kee'
copyright = f'{datetime.date.today().year} Andy Kee'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_remove_toctrees',
    'matplotlib.sphinxext.plot_directive'
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

html_logo = '_static/logo.svg'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['custom.css']
html_theme_options = {
    "github_url": "https://github.com/andykee/prtools",
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    "article_header_start": [],
    "secondary_sidebar_items": [],
    "footer_start": ["copyright"],
    "footer_end": [],
    "show_prev_next": False,
    "pygment_light_style": "tango",
    "pygment_dark_style": "nord",
    "favicons": [
      {
         "rel": "icon",
         "sizes": "16x16",
         "href": "favicon/favicon-16x16.png",
      },
      {
         "rel": "icon",
         "sizes": "32x32",
         "href": "favicon/favicon-32x32.png",
      },
      {
         "rel": "apple-touch-icon",
         "sizes": "180x180",
         "href": "favicon/apple-touch-icon-180x180.png",
         "color": "#000000",
      },
    ]
}

html_sidebars = {
  "index": ["sidebar-nav.html"], # also see sidebar-empty.html
  "generated/*": ["sidebar-nav.html"]
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


# -- Plot config -------------------------------------------------------------
dpi = 144

plot_rcparams = {}  # noqa
plot_rcparams['font.size'] = 12*72/dpi  # 12 pt
plot_rcparams['axes.titlesize'] = 14*72/dpi  # 14 pt
plot_rcparams['axes.labelsize'] = 12*72/dpi  # 12 pt
plot_rcparams['axes.linewidth'] = 0.5
plot_rcparams['lines.linewidth'] = 1
plot_rcparams['lines.markersize'] = 2
plot_rcparams['xtick.major.width'] = 0.5
plot_rcparams['xtick.major.size'] = 2
plot_rcparams['ytick.major.width'] = 0.5
plot_rcparams['ytick.major.size'] = 2
plot_rcparams['grid.linewidth'] = 0.5
plot_rcparams['xtick.labelsize'] = 12*72/dpi  # 12 pt
plot_rcparams['ytick.labelsize'] = 12*72/dpi  # 12 pt
plot_rcparams['legend.fontsize'] = 12*72/dpi  # 12 pt
plot_rcparams['figure.figsize'] = (2.5, 2.5)
plot_rcparams['figure.subplot.wspace'] = 0.2
plot_rcparams['figure.subplot.hspace'] = 0.2
plot_rcparams['savefig.bbox'] = 'tight'
plot_rcparams['savefig.transparent'] = True

plot_apply_rcparams = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [('png', dpi*2)]
plot_pre_code = """
import prtools
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(12345)
"""