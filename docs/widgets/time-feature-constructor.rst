Time Features Constructor
=========================

The **Time Features Constructor** constructs new numeric variables from the input data using Python-style expressions and time-window functions. It is an essential tool for time-series feature engineering, allowing you to easily compute rolling statistics, lagged variables, and custom mathematical transformations.

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
     - Optional configuration table containing predefined ``Variable`` and ``Expression`` columns.

Outputs
-------

.. list-table::
   :header-rows: 1

   * - Signal
     - Type
     - Description
   * - Data
     - ``Orange.data.Table``
     - The transformed input table, appended with the newly generated variables.
   * - Variable Definitions
     - ``Orange.data.Table``
     - A configuration table describing both the original and generated variables. This can be passed to other widgets like the **Variable Dependency Graph**.

Description
-----------

The widget interface consists of an editor where you can define new variables. 

- **New** creates a new numeric variable definition. 
- **Remove** deletes the selected definition.
- **Reset** clears the widget state back to default. 
- **Send** evaluates the current definitions and sends the transformed data to the output.

The editor contains a variable name field, a meta-variable toggle, an expression field, a selector for source variables, a selector for standard functions, and a selector for time functions.

Expressions
-----------

Expressions can reference variables from the original input domain. Variable names are sanitized to be valid Python identifiers: spaces and punctuation become underscores, and names that start with a digit receive a leading underscore.

**Examples:**

.. code-block:: python

   age + 1
   log(price)
   shift(age, -20)
   mean(temperature, -2, 2)

Standard Python built-ins such as ``abs``, ``int``, ``float`` and ``pow`` are available. Additionally, public functions from Python's ``math`` module and selected random/nan helpers can be used directly in your expressions.

Time Functions
--------------

The widget provides specialized time-window functions for sequential data.

.. list-table::
   :header-rows: 1

   * - Function
     - Description
   * - ``shift(var, offset)``
     - Value of ``var`` at the row shifted by ``offset``. Out-of-range rows become missing values.
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

Usage Example
-------------

If you have a dataset with a ``temperature`` variable recorded daily, you can create a 3-day rolling average feature by adding a new variable with the expression:

.. code-block:: python

   mean(temperature, -2, 0)

This will compute the mean of the temperature for the current day and the two preceding days.

Notes
-----

The widget re-evaluates definitions from the original input table on every change. This prevents repeated sends from accumulating previously generated variables as new source columns, ensuring a clean and reproducible transformation pipeline.
