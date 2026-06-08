TimeFeatures
============

|addon| is an Orange add-on for constructing time-based features,
generating dependency graphs between variables, and saving datasets to a
database.

Build the documentation locally with:

.. code-block:: bash

   python -m pip install -e ".[docs]"
   python -m sphinx -b html docs docs/build/html

The add-on also ships a local HTML build for Orange's internal Help panel.
The widget help action resolves these local pages through the
``orange.canvas.help`` entry point; it does not need an internet URL.

.. _widgets:

Widgets
-------

.. toctree::
   :maxdepth: 1

   widgets/time-feature-constructor
   widgets/variable-dependency-graph
   widgets/save-to-db

Workflow
--------

A typical workflow uses **Time Features Constructor** to add derived
variables, sends the generated variable-definition table to **Variable
Dependency Graph**, and optionally sends the transformed data to **Save to
DB**.
