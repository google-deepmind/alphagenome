{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
{%- if not methods %}
   :members:
{%- endif %}

{% block methods %}
{%- set visible_methods = methods | reject('equalto', '__init__') | list %}
{% if visible_methods %}
.. rubric:: Methods

.. autosummary::
   :toctree: .
{% for item in visible_methods %}
   ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
