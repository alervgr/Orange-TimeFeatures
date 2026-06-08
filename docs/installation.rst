Installation
============

|addon| is an Orange3 add-on, so it requires a working Orange3
installation in the same Python environment.

From PyPI
---------

.. code-block:: bash

   pip install TimeFeatures

This pulls in every runtime dependency, including the precompiled
``psycopg2-binary`` wheel so the **Save to DB** widget works on macOS,
Linux and Windows without a C toolchain.

From source
-----------

.. code-block:: bash

   git clone https://github.com/alervgr/Orange-TimeFeatures.git
   cd Orange-TimeFeatures
   pip install -e .

The ``-e`` flag installs in *editable* mode so changes to the source
tree are picked up on the next Orange restart without reinstalling.

Inside Orange
-------------

|addon| also installs through Orange's GUI:

1. Open Orange.
2. Go to **Options → Add-ons…**.
3. Tick **TimeFeatures** in the list and confirm.
4. Restart Orange. The new widgets appear under the **Time-Features**
   toolbox section.

.. image:: https://github.com/alervgr/Orange-TimeFeatures/blob/main/imgs/installation.png?raw=true
   :alt: TimeFeatures in Orange's Add-on installer.

Anaconda
--------

The same ``pip install TimeFeatures`` command works inside a Conda
environment as long as Orange3 is installed there.

Dependencies
------------

|addon| declares the following runtime dependencies (see
``setup.py``):

.. list-table::
   :header-rows: 1

   * - Package
     - Minimum version
     - Used by
   * - ``numpy``
     - 1.22.4
     - All widgets.
   * - ``scipy``
     - 1.7.3
     - Sparse adjacency matrices in the dependency graph.
   * - ``PyQt5``
     - 5.15.6
     - Widget UI.
   * - ``AnyQt``
     - 0.2.0
     - Qt abstraction used by Orange.
   * - ``SQLAlchemy``
     - 1.4.0
     - Dialect-agnostic SQL toolkit used by **Save to DB**.
   * - ``psycopg2-binary``
     - 2.9.9
     - PostgreSQL driver for **Save to DB**.
   * - ``PyMySQL``
     - 1.0.0
     - MySQL driver for **Save to DB**.
   * - ``Orange3-Network``
     - 1.8.0
     - ``Network`` type produced by **Variable Dependency Graph**.

Orange3 itself is intentionally left out of the requirements list — the
host application provides it.

Running tests
-------------

The test suite uses ``unittest`` and lives under
``timefeatures/widgets/tests/``. From a checkout:

.. code-block:: bash

   pip install -e .
   python -m unittest discover -v timefeatures/widgets/tests

The widget-level tests inherit from ``Orange.widgets.tests.base.WidgetTest``
and therefore require Qt; a virtual display (``Xvfb``) is enough on a
headless CI worker.
