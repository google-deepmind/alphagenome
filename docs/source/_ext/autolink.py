# Copyright 2026 Google LLC.
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

"""Sphinx extension that links inline code in tutorial notebooks to the API.

Notebook markdown uses plain backticks (e.g. `genome.Variant`) so that it reads
cleanly in Colab and on GitHub. When the docs are built, this extension wraps
each inline literal that names a documented Python object in a cross-reference,
so the rendered site links it to the generated API page.

Matching rules:

*   Only documents under `colabs/` are processed.
*   Dotted names are matched against every dotted suffix of the documented
    objects, e.g. `genome.Variant` -> `alphagenome.data.genome.Variant`. A
    trailing `()` is ignored.
*   Bare names (no dot) are only linked when they are CamelCase and resolve to
    exactly one documented class, so tokens like `ATAC` are left alone.
*   If a name matches both a v1 and a v2 object, notebooks under `colabs/v1/`
    link to the v1 object and all other notebooks link to the v2 object.
*   Literals inside existing links and headings are left untouched.
*   The tutorials' `dna_model` variable is treated as `DnaClient`, and a few
    common third-party names (e.g. `AnnData`, `pandas.DataFrame`) link to their
    own docs through intersphinx.
"""

import collections
from typing import Any

from docutils import nodes
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.transforms.post_transforms import SphinxPostTransform

_PACKAGE_PREFIX = 'alphagenome.'
_V1_DOC_PREFIX = 'colabs/v1/'
_V2_MARKER = '.v2.'

# Variable names used throughout the tutorials, mapped to the class they hold.
_VARIABLE_ALIASES = {'dna_model': 'DnaClient'}

# Third-party names resolved through intersphinx rather than the local API.
_EXTERNAL_TARGETS = {
    'AnnData': 'anndata.AnnData',
    'anndata.AnnData': 'anndata.AnnData',
    'pandas.DataFrame': 'pandas.DataFrame',
    'pd.DataFrame': 'pandas.DataFrame',
    'numpy.ndarray': 'numpy.ndarray',
    'np.ndarray': 'numpy.ndarray',
    'matplotlib.figure.Figure': 'matplotlib.figure.Figure',
}


def _suffix_index(app: Sphinx) -> dict[str, set[tuple[str, str]]]:
  """Maps every dotted suffix of a documented object to (fullname, objtype)."""
  index = getattr(app, '_autolink_index', None)
  if index is not None:
    return index
  index = collections.defaultdict(set)
  for fullname, entry in app.env.get_domain('py').objects.items():
    if not fullname.startswith(_PACKAGE_PREFIX):
      continue
    parts = fullname.split('.')
    for i in range(len(parts)):
      index['.'.join(parts[i:])].add((fullname, entry.objtype))
  app._autolink_index = index  # pylint: disable=protected-access
  return index


def _is_camel_case(text: str) -> bool:
  return text[:1].isupper() and any(c.islower() for c in text)


def _pick(
    text: str, candidates: set[tuple[str, str]], is_v2_doc: bool
) -> str | None:
  """Returns the single best target for `text`, or None if unresolved."""
  if '.' not in text:
    if not _is_camel_case(text):
      return None
    candidates = {c for c in candidates if c[1] == 'class'}
  if len(candidates) > 1:
    candidates = {c for c in candidates if (_V2_MARKER in c[0]) == is_v2_doc}
  if len(candidates) != 1:
    return None
  return next(iter(candidates))[0]


class AutolinkLiterals(SphinxPostTransform):
  """Wraps inline literals in notebooks in cross-references to the API."""

  # Runs before Sphinx's ReferencesResolver (priority 10), which then resolves
  # the pending_xref nodes like any other `{py:obj}` reference.
  default_priority = 5

  def run(self, **kwargs: Any) -> None:
    del kwargs
    docname = self.env.docname
    if not docname.startswith('colabs/'):
      return
    is_v2_doc = not docname.startswith(_V1_DOC_PREFIX)
    index = _suffix_index(self.app)
    for literal in list(self.document.findall(nodes.literal)):
      parent = literal.parent
      if isinstance(
          parent, (nodes.reference, addnodes.pending_xref, nodes.title)
      ):
        continue
      text = literal.astext().strip().removesuffix('()')
      head, _, rest = text.partition('.')
      if rest and head in _VARIABLE_ALIASES:
        text = f'{_VARIABLE_ALIASES[head]}.{rest}'
      if text in _EXTERNAL_TARGETS:
        target, reftype = _EXTERNAL_TARGETS[text], 'class'
      else:
        target = _pick(text, index.get(text, set()), is_v2_doc)
        reftype = 'obj'
      if target is None:
        continue
      xref = addnodes.pending_xref(
          '',
          refdomain='py',
          reftype=reftype,
          reftarget=target,
          refdoc=docname,
          refexplicit=True,
          refwarn=False,
      )
      xref += literal.deepcopy()
      literal.replace_self(xref)


def setup(app: Sphinx) -> dict[str, Any]:
  app.add_post_transform(AutolinkLiterals)
  return {'parallel_read_safe': True, 'parallel_write_safe': True}
