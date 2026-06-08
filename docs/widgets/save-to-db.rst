Save to DB
==========

Saves an Orange data table into a SQL database supported by Orange's SQL
backend layer.

Inputs
------

.. list-table::
   :header-rows: 1

   * - Signal
     - Type
     - Description
   * - Data
     - ``Orange.data.Table``
     - Dataset to upload.

Outputs
-------

This widget has no output signals.

Controls
--------

The widget shows the detected class type, row count and column count for
the input table. Connection fields are inherited from Orange's SQL base
widget. The TimeFeatures-specific fields are:

.. list-table::
   :header-rows: 1

   * - Field
     - Description
   * - Table name
     - Destination table name. It must start with a letter or underscore and
       contain only letters, digits and underscores, up to 63 characters.
   * - Email
     - Optional notification address used after a successful upload.

``Save`` creates or updates the metadata table named ``datasets`` and then
creates the destination table for the uploaded data.

Database Behavior
-----------------

The widget maps Orange variable types to SQL column types:

.. list-table::
   :header-rows: 1

   * - Orange Variable
     - SQL Type
   * - ``DiscreteVariable``
     - ``VARCHAR``
   * - ``ContinuousVariable``
     - ``FLOAT(10)``
   * - ``TimeVariable``
     - ``TIMESTAMP``
   * - ``StringVariable``
     - ``VARCHAR``

Validation
----------

Before saving, the widget checks that a table name is present, that the
table name is SQL-safe, that host and database fields are present, and that
the optional email field has a valid email format.
