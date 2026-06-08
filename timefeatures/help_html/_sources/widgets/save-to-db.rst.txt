Save to DB
==========

The **Save to DB** widget saves an Orange data table into a SQL database. It relies on Orange's SQL backend layer to seamlessly upload your datasets for persistent storage and further querying.

Inputs
------

.. list-table::
   :header-rows: 1

   * - Signal
     - Type
     - Description
   * - Data
     - ``Orange.data.Table``
     - The dataset you wish to upload to the database.

Outputs
-------

This widget has no output signals. It acts as an endpoint in your data pipeline.

Controls
--------

The widget displays the detected class type, row count, and column count for the input table. Connection fields (such as Host, Port, Database, User, and Password) are inherited from Orange's SQL base widget.

The TimeFeatures-specific fields are:

.. list-table::
   :header-rows: 1

   * - Field
     - Description
   * - Table name
     - The destination table name in the database. It must start with a letter or underscore and contain only letters, digits, and underscores, up to 63 characters.
   * - Email
     - An optional notification email address used to send an alert after a successful data upload.

Clicking **Save** creates or updates a metadata table named ``datasets`` and then creates the destination table for the uploaded data.

Database Behavior
-----------------

When uploading data, the widget automatically maps Orange variable types to appropriate SQL column types:

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

Before attempting to save the data, the widget performs several validation checks:

- Verifies that a table name is provided.
- Ensures the table name is SQL-safe (avoids SQL injection or syntax errors).
- Checks that the required host and database connection fields are present.
- Validates the optional email field to ensure it has a correct email format.

Usage Example
-------------

1. Connect a **File** widget (or any widget outputting data) to the **Save to DB** widget.
2. Enter your database connection details (Host, Database, Username, Password).
3. Specify a valid **Table name** (e.g., ``my_time_series_data``).
4. Optionally, provide an **Email** to receive a notification upon completion.
5. Click **Save** to upload your dataset.
