Variable Dependency Graph
=========================

The **Variable Dependency Graph** widget builds a directed network whose
nodes are variables and whose edges encode the dependencies declared in
their expressions. Edge weights reflect the time-window size used by the
expression, making it easy to spot which derived features have the
widest temporal footprint.

Inputs
------

.. list-table::
   :header-rows: 1

   * - Signal
     - Type
     - Description
   * - Variable Definitions
     - ``Orange.data.Table``
     - A configuration table containing the ``Variable`` and
       ``Expression`` columns produced by **Time Features Constructor**
       (second output).

Outputs
-------

.. list-table::
   :header-rows: 1

   * - Signal
     - Type
     - Description
   * - Network
     - ``orangecontrib.network.Network``
     - A directed weighted graph. ``network.edges[0].edges`` is the
       sparse adjacency matrix; its non-zero values carry the edge
       weights described below.

How It Works
------------

The widget processes the input table row by row:

- A row whose ``Expression`` cell is empty / ``NaN`` / ``?`` is treated
  as an **original variable** (source feature, ``var_type = Original``).
- A row with a valid expression is treated as a **derived variable**
  (``var_type = Derived``).

For every derived variable, the widget scans its expression for
references to known variable names. Every reference produces one
directed edge **source → dependency** (i.e. an edge ``X1 → X2`` means
"X1's expression references X2").

Edge Weights
------------

Each edge carries a numeric weight that summarises **how far back or
forward in time** the source variable looks at its dependency.

Rule
~~~~

For an edge ``Xᵢ → Xⱼ``:

- If ``Xⱼ`` appears inside one or more **temporal calls** in ``Xᵢ``'s
  expression (``shift``, ``sum``, ``mean``, ``count``, ``min``, ``max``,
  ``sd``), the weight is the **maximum absolute integer argument**
  across all such calls.
- If ``Xⱼ`` only appears outside temporal calls (e.g. ``Xⱼ + 1``), the
  weight defaults to ``1``.

Examples
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 15 35

   * - Expression on the source
     - Edge weight
     - Notes
   * - ``shift(X2, -20)``
     - ``20``
     - One temporal call, single argument.
   * - ``sum(X2, -5, 10)``
     - ``10``
     - ``max(abs(-5), abs(10))`` — three-argument family.
   * - ``X2 + 1``
     - ``1``
     - Plain reference, no temporal window.
   * - ``shift(X2, -5) + X2``
     - ``5``
     - Mixed usage; the temporal occurrence wins.
   * - ``shift(X2, -3) + mean(X2, -7, 7)``
     - ``7``
     - Maximum across all temporal calls.
   * - ``shift(X2, -3) + mean(X3, -10, 10)``
     - ``3`` / ``10``
     - Per-dependency: edges ``→ X2`` and ``→ X3`` get independent
       weights.

Node Metadata
-------------

The output network exposes per-node metadata in ``network.nodes`` for
downstream styling:

.. list-table::
   :header-rows: 1

   * - Meta
     - Type
     - Values
   * - ``var_name``
     - String
     - Sanitised variable name (spaces and hyphens become ``_``).
   * - ``var_type``
     - Discrete
     - ``Derived`` (has an expression) or ``Original`` (source feature).

Controls
--------

- **Generate** — rebuilds the graph from the current configuration
  table. The widget also auto-regenerates whenever a valid input
  arrives.

Input Requirements
------------------

The first two columns of the input table **must** be named exactly
``Variable`` and ``Expression``. Any other shape is rejected with an
explicit error message; no network output is sent.

Usage Example
-------------

1. Connect the **Variable Definitions** output of **Time Features
   Constructor** to the input of this widget.
2. Connect the **Network** output to the **Network Explorer** widget
   from the *Orange Network* add-on.
3. In **Network Explorer**:

   - Color the nodes by ``var_type`` to separate derived from original
     features.
   - Optionally map the edge thickness to the edge weights (the
     ``data`` array of the sparse adjacency matrix) to visualise which
     dependencies span the widest temporal windows.

Implementation notes
--------------------

- Variable name lookup is **O(1)** via a precomputed ``name → index``
  map, so the full graph build is linear in the number of references.
- The detection regex uses word boundaries (``\b``), so ``X1`` will not
  match inside ``X10``.
- Sanitisation maps spaces and hyphens to underscores so the names line
  up with how the **Time Features Constructor** rewrites them inside
  expressions.
