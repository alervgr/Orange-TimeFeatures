Variable Dependency Graph
=========================

Builds a directed network from a variable-definition table. Each edge means
that one variable expression references another variable.

Inputs
------

.. list-table::
   :header-rows: 1

   * - Signal
     - Type
     - Description
   * - Variable Definitions
     - ``Orange.data.Table``
     - Table with ``Variable`` and ``Expression`` columns, usually produced
       by **Time Features Constructor**.

Outputs
-------

.. list-table::
   :header-rows: 1

   * - Signal
     - Type
     - Description
   * - Network
     - ``orangecontrib.network.Network``
     - Directed dependency graph.

How It Works
------------

Rows without an expression are treated as original variables. Rows with an
expression are treated as derived variables. The widget scans each
expression for references to known variable names and creates one directed
edge per dependency.

The output network stores node metadata:

.. list-table::
   :header-rows: 1

   * - Meta Variable
     - Description
   * - ``var_name``
     - Variable name.
   * - ``var_type``
     - ``Derived`` for generated variables or ``Original`` for source
       variables.

Controls
--------

``Generate`` rebuilds the graph from the current configuration table. The
widget also generates automatically when valid input arrives.

Input Requirements
------------------

The input table must contain the first two columns named exactly
``Variable`` and ``Expression``. If these columns are missing or renamed,
the widget reports an error and sends no network.
