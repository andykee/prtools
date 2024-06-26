{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% if attributes %}
.. rubric:: {{ _('Attributes') }}

.. autosummary::
    :toctree:
    {% for item in attributes %}
    {%- if not item.startswith('_') %}
    ~{{ name }}.{{ item }}
    {%- endif -%}
    {%- endfor %}

{% endif %}

{% if methods | reject("equalto", "__init__") | list %}
.. rubric:: {{ _('Methods') }}

.. autosummary::
    :toctree:
    {% for item in methods %}
    {%- if not item.startswith('_') or item in ['__call__'] %}
    ~{{ name }}.{{ item }}
    {%- endif -%}
    {%- endfor %}

{% endif %}