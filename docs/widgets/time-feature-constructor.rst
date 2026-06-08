Time Features Constructor
=========================

Constructs new numeric variables from the input data using Python-style
expressions and time-window functions.

Inputs
------

.. list-table::
   :header-rows: 1

   * - Signal
     - Type
     - Description
   * - Data
     - ``Orange.data.Table``
     - Source table used to evaluate expressions.
   * - Variable Definitions
     - ``Orange.data.Table``
     - Optional configuration table with ``Variable`` and ``Expression``
       columns.

Outputs
-------

.. list-table::
   :header-rows: 1

   * - Signal
     - Type
     - Description
   * - Data
     - ``Orange.data.Table``
     - Input table transformed with the generated variables.
   * - Variable Definitions
     - ``Orange.data.Table``
     - Configuration table describing original and generated variables.

Controls
--------

``New`` creates a new numeric variable definition. ``Remove`` deletes the
selected definition, and ``Reset`` clears the widget state. ``Send``
evaluates the current definitions and sends the transformed data.

The editor contains a variable name, a meta-variable toggle, an expression
field, a selector for source variables, a selector for standard functions,
and a selector for time functions.

Expressions
-----------

Expressions can reference variables from the original input domain. Variable
names are sanitized like Python identifiers: spaces and punctuation become
underscores, and names that start with a digit get a leading underscore.

Examples:

.. code-block:: python

   age + 1
   log(price)
   shift(age, -20)
   mean(temperature, -2, 2)

Standard Python built-ins such as ``abs``, ``int``, ``float`` and ``pow``
are available, together with public functions from Python's ``math`` module
and selected random/nan helpers.

Time Functions
--------------

.. list-table::
   :header-rows: 1

   * - Function
     - Description
   * - ``shift(var, offset)``
     - Value of ``var`` at the row shifted by ``offset``. Out-of-range rows
       become missing values.
   * - ``sum(var, start, end)``
     - Sum of non-missing values in the inclusive offset window.
   * - ``mean(var, start, end)``
     - Mean of non-missing values in the inclusive offset window.
   * - ``count(var, start, end)``
     - Number of non-missing values in the inclusive offset window.
   * - ``min(var, start, end)``
     - Minimum non-missing value in the inclusive offset window.
   * - ``max(var, start, end)``
     - Maximum non-missing value in the inclusive offset window.
   * - ``sd(var, start, end)``
     - Standard deviation in the inclusive offset window.

Notes
-----

The widget re-evaluates definitions from the original input table. This
prevents repeated sends from accumulating previously generated variables as
new source columns.
