Variable Dependency Graph
=========================

The **Variable Dependency Graph** widget builds and visualizes a directed network graph from a variable-definition table. Each directed edge in the graph indicates that one variable's expression references another variable, making it easy to trace feature dependencies.

Inputs
------

.. list-table::
   :header-rows: 1

   * - Signal
     - Type
     - Description
   * - Variable Definitions
     - ``Orange.data.Table``
     - A table containing ``Variable`` and ``Expression`` columns. This is usually produced by the **Time Features Constructor** widget's second output.

Outputs
-------

.. list-table::
   :header-rows: 1

   * - Signal
     - Type
     - Description
   * - Network
     - ``orangecontrib.network.Network``
     - A directed dependency graph representing how variables relate to one another.

How It Works
------------

The widget processes the input table row by row:

- Rows without an expression are treated as **original variables** (source features).
- Rows with an expression are treated as **derived variables** (calculated features).

The widget scans each expression for references to known variable names and creates one directed edge per dependency (from the referenced variable to the derived variable).

The output network stores metadata for each node, which can be used for styling in downstream network visualization widgets:

.. list-table::
   :header-rows: 1

   * - Meta Variable
     - Description
   * - ``var_name``
     - The name of the variable.
   * - ``var_type``
     - Categorized as ``Derived`` for generated variables or ``Original`` for source variables.

Controls
--------

- **Generate**: Rebuilds the graph from the current configuration table. The widget also generates the graph automatically when valid new input arrives.

Input Requirements
------------------

The input table **must** contain the first two columns named exactly ``Variable`` and ``Expression``. If these columns are missing or have been renamed, the widget will report an error and will not send any network output.

Usage Example
-------------

1. Connect the **Variable Definitions** output of a **Time Features Constructor** widget to the input of the **Variable Dependency Graph**.
2. Connect the **Network** output of this widget to the **Network Explorer** widget (from the Orange Network add-on).
3. In the **Network Explorer**, you can color the nodes by ``var_type`` to visually distinguish your original dataset features from your newly constructed time features, and see exactly how they depend on each other.
