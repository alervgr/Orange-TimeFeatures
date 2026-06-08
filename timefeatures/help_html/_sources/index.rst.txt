TimeFeatures
============

|addon| is an `Orange3 <https://orangedatamining.com/>`_ add-on for
time-series feature engineering. It ships four widgets that work
together to define, visualise, persist and re-load derived variables
built on top of an existing dataset.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Widget
     - What it does
   * - :doc:`widgets/time-feature-constructor`
     - Defines new variables from existing ones using Python-style
       expressions and time-window functions (``shift``, ``sum``,
       ``mean``, ``min``, ``max``, ``sd``, ``count``).
   * - :doc:`widgets/variable-dependency-graph`
     - Builds a directed, **weighted** dependency graph from the
       resulting variable / expression table. Edge weights summarise
       how far back or forward in time each variable looks.
   * - :doc:`widgets/save-to-db`
     - Persists the resulting dataset to a SQL database (PostgreSQL
       or MySQL), with full SQL-injection defences and an optional
       completion email.
   * - :doc:`widgets/load-from-db`
     - Lists the datasets previously stored by Save to DB and pulls
       the chosen one back into Orange, optionally marking the class
       column directly so no Select Columns widget is needed.

.. _workflow:

Workflow
--------

A typical pipeline:

.. code-block:: text

   File → Time Features Constructor → ┬→ <downstream models>
                                      └→ Variable Dependency Graph
                                              ↓
                                      Network Explorer

The **Time Features Constructor** outputs both the transformed data
(top arrow) and the variable / expression definition table (bottom
arrow). Feed the latter into the **Variable Dependency Graph** to
visualise the dependencies, and send the data into **Save to DB** if
you want to keep it in a database.

Getting started
---------------

.. toctree::
   :maxdepth: 2

   installation

.. _widgets:

Widgets
-------

.. toctree::
   :maxdepth: 1

   widgets/time-feature-constructor
   widgets/variable-dependency-graph
   widgets/save-to-db
   widgets/load-from-db

Project
-------

.. toctree::
   :maxdepth: 1

   changes

Building this documentation
---------------------------

.. code-block:: bash

   pip install -e ".[docs]"
   python -m sphinx -b html docs docs/build/html

The HTML build is also bundled with the wheel so Orange's in-app help
panel can resolve every widget's *Help* action without internet
access.

Indices
-------

* :ref:`genindex`
* :ref:`search`
