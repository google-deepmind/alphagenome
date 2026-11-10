# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration file for the Sphinx documentation builder."""

#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import importlib.metadata
import inspect
import os
import sys

# The package is installed by Readthedocs before sphinx building.
import alphagenome
from IPython.lib.lexers import IPython3Lexer  # pylint: disable=g-importing-member

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '_ext'))

# -- Project information -----------------------------------------------------

project = 'alphagenome'
project_info = importlib.metadata.metadata(project)
author = project_info['Author']
copyright = f'2024, {author}'  # pylint: disable=redefined-builtin
version = project_info['Version']
repository_url = f'https://github.com/google-deepmind/{project}'


# The full version, including alpha/beta/rc tags
release = version
# Warn if links are broken
nitpicky = True

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_nb',
    'sphinx_design',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinxcontrib.bibtex',
    'sphinx_autodoc_typehints',
    'sphinx.ext.mathjax',
    'IPython.sphinxext.ipython_console_highlighting',
    'sphinx.ext.coverage',
    'sphinx_copybutton',
    'sphinx_reredirects',
    'sphinx.ext.linkcode',
    'autolink',
]

autosummary_generate = True
autodoc_member_order = 'groupwise'
default_role = 'literal'
bibtex_reference_style = 'author_year'
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
myst_heading_anchors = 6  # Create heading anchors for h1-h6
autodoc_mock_imports = [
    'absl',
]
bibtex_bibfiles = ['refs.bib']

myst_enable_extensions = [
    'amsmath',
    'colon_fence',
    'deflist',
    'dollarmath',
    'html_image',
    'html_admonition',
    'attrs_inline',
    'attrs_block',
]

# TODO(b/372225132): Resolve showing notebook output without executing.
# TODO(b/372226231): Resolve not modifying notebook when building docs.
myst_url_schemes = ['http', 'https', 'mailto']
nb_output_stderr = 'remove'
nb_execution_mode = 'off'
nb_merge_streams = True
typehints_defaults = 'braces'
_PUBLIC_MODULE_ALIASES = {
    'pandas.core.frame': 'pandas',
    'pandas.core.series': 'pandas',
    'anndata._core.anndata': 'anndata',
}


def typehints_fixup_module_name(module):
  return _PUBLIC_MODULE_ALIASES.get(module, module)


suppress_warnings = ['config.cache']

source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'anndata': ('https://anndata.readthedocs.io/en/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'protos']

# -- Options for autodoc -----------------------------------------------------

autodoc_default_options = {
    'member-order': 'bysource',
    'special-members': True,
    'exclude-members': '__repr__, __str__, __weakref__, __lt__',
}
# Append custom `__init__` docstrings to the class docstring.
autoclass_content = 'both'
# jaxtyping annotations (e.g. `Float32[ndarray, 'S A']`) have no API page.
nitpick_ignore_regex = [
    ('py:class', r'jaxtyping\..*'),
    # Protobuf messages and wrappers that appear in signatures are not part of
    # the documented API.
    ('py:class', r'alphagenome\.protos\..*'),
    ('py:class', r'google\.protobuf\..*'),
    # Private helper types, for example genome._FastaExtractorType.
    ('py:class', r'alphagenome\..*\._\w+'),
]


# -- Source code links -----------------------------------------------------


def linkcode_resolve(domain, info):
  """Resolve a GitHub URL corresponding to Python object."""
  if domain != 'py':
    return None

  try:
    mod = sys.modules[info['module']]
  except ImportError:
    return None

  obj = mod
  try:
    for attr in info['fullname'].split('.'):
      obj = getattr(obj, attr)
  except AttributeError:
    return None
  else:
    obj = inspect.unwrap(obj)

  try:
    filename = inspect.getsourcefile(obj)
  except TypeError:
    return None

  try:
    source, lineno = inspect.getsourcelines(obj)
  except OSError:
    return None

  path = os.path.relpath(filename, start=os.path.dirname(alphagenome.__file__))
  return (
      f'{repository_url}/tree/main/src/{project}/'
      f'{path}#L{lineno}-L{lineno + len(source) - 1}'
  )


_HIDDEN_MEMBERS = frozenset({
    'from_proto',
    'to_proto',
    'from_pyranges_dict',
    'to_pyranges_dict',
    'pad_inplace',
    'resize_inplace',
})


def _skip_member(app, what, name, obj, skip, options):
  """Hides serialization helpers and builtin C methods on IntEnum classes."""
  del app, what, skip, options
  if name in _HIDDEN_MEMBERS:
    return True
  if inspect.isroutine(obj) and not getattr(obj, '__module__', None):
    return True
  return None


def _mark_unlisted_nosearch(app, env):
  """Excludes unlisted AlphaGenome 1 notebooks from the search index."""
  del app
  for docname in env.toctree_includes.get(
      'tutorials/alphagenome1_notebooks', ()
  ):
    env.metadata[docname]['nosearch'] = True


def setup(app):
  app.add_lexer('python', IPython3Lexer)
  app.connect('autodoc-skip-member', _skip_member)
  app.connect('env-check-consistency', _mark_unlisted_nosearch)


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = 'sphinx_book_theme'
html_title = 'AlphaGenome'
pygments_style = 'default'
html_theme_options = {
    'repository_url': repository_url,
    'repository_branch': 'main',
    'use_repository_button': True,
    'launch_buttons': {
        'colab_url': 'https://colab.research.google.com',
    },
    'max_navbar_depth': 5,
    'show_toc_level': 2,
    'show_prev_next': False,
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    (
        'https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500'
        '&family=Google+Sans+Text:wght@400;500&family=Roboto+Mono:wght@400;500'
        '&display=swap'
    ),
    'custom.css',
]

# Redirects for moved pages, handled by sphinx_reredirects. Maps old docnames
# (without .html extension) to targets relative to the old page's directory.
redirects = {
    'colabs/batch_variant_scoring': 'v1/batch_variant_scoring.html',
    'colabs/deriving_PSI_values': 'v1/deriving_PSI_values.html',
    'colabs/essential_commands': 'v1/essential_commands.html',
    'colabs/example_analysis_workflow': 'v1/example_analysis_workflow.html',
    'colabs/haplotype_workaround': 'v1/haplotype_workaround.html',
    'colabs/quick_start': 'v1/quick_start.html',
    'colabs/splicing_variant_scoring': 'v1/splicing_variant_scoring.html',
    'colabs/tissue_ontology_mapping': 'v1/tissue_ontology_mapping.html',
    'colabs/variant_scoring_ui': 'v1/variant_scoring_ui.html',
    'colabs/visualization_modality_tour': 'v1/visualization_modality_tour.html',
    'visualization_library_basics': 'visualization_basics.html',
}
