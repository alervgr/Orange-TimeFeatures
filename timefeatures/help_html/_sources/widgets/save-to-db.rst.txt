Save to DB
==========

The **Save to DB** widget persists an Orange data table to a SQL
database. It supports two dialects out of the box:

- **PostgreSQL** — via Orange's built-in SQL backend layer
  (``psycopg2``).
- **MySQL** — via a lightweight ``pymysql`` wrapper shipped with
  |addon|.

The dialect is selected with the top combo box of the connection
panel; the same host / port / database / user / password fields apply
to both.

Inputs
------

.. list-table::
   :header-rows: 1

   * - Signal
     - Type
     - Description
   * - Data
     - ``Orange.data.Table``
     - The dataset to upload.

Outputs
-------

This widget has no output signals — it is a pipeline sink.

Controls
--------

The widget shows the detected target type, row count and column count
of the input. Connection fields (host, port, database, user, password)
come from Orange's SQL base widget.

TimeFeatures-specific fields:

.. list-table::
   :header-rows: 1

   * - Field
     - Description
   * - Database type
     - Combo box at the top of the connection box. Pick **PostgreSQL**
       or **MySQL**.
   * - Table name
     - Destination table name. Validated against PostgreSQL identifier
       rules (see *Validation* below); MySQL accepts a superset, so
       the same rule is safe on both.
   * - Email
     - Optional notification address. A summary email is sent once the
       upload finishes, including the table name, row / column counts,
       and elapsed time.

Clicking **Save** creates (if missing) a metadata table named
``datasets`` and then the destination table, and finally inserts every
row of the input.

Type Mapping
------------

Column types are picked per dialect so the table works on either
backend without truncation surprises:

.. list-table::
   :header-rows: 1

   * - Orange Variable
     - PostgreSQL
     - MySQL
   * - ``DiscreteVariable``
     - ``VARCHAR(255)``
     - ``VARCHAR(255)``
   * - ``ContinuousVariable``
     - ``DOUBLE PRECISION``
     - ``DOUBLE``
   * - ``TimeVariable``
     - ``TIMESTAMP``
     - ``DATETIME`` (MySQL's ``TIMESTAMP`` is capped at 2038).
   * - ``StringVariable``
     - ``TEXT``
     - ``TEXT``

Identifiers (table and column names) are quoted with the dialect's
native syntax — ``"name"`` on PostgreSQL, ``\`name\``` on MySQL — and
any internal occurrence of the quote character is doubled to avoid
breakouts.

Validation
----------

Before touching the database the widget enforces:

- **Table name present** — required.
- **Table name well-formed** — must match
  ``^[A-Za-z_][A-Za-z0-9_]{0,62}$`` (PostgreSQL identifier rules:
  starts with a letter or underscore, only letters / digits /
  underscores, max 63 characters).
- **Connection fields present** — host and database.
- **Email well-formed** — only when the optional field is filled.

Security
--------

The widget defends against SQL injection at two layers:

- **Parametrised queries** — every value inserted into the database
  uses parameter binding (``execute_sql_query(query, params=...)``).
  Row contents and metadata values are never concatenated into the SQL
  string.
- **Identifier escaping** — table names and column names that appear
  in DDL (``CREATE TABLE`` / ``INSERT INTO``) are wrapped with
  PostgreSQL's standard quoted-identifier syntax (``"name"`` with any
  internal ``"`` doubled). The whitelist regex above further blocks
  payloads from ever reaching the escape step.

Usage Example
-------------

1. Connect a **File** widget (or any data-producing widget) to **Save
   to DB**.
2. Fill in the database connection details (host, database, username,
   password).
3. Enter a valid table name, e.g. ``my_time_series_data``.
4. *(Optional)* Enter an email for the completion notification.
5. Click **Save**. The widget shows a progress bar while inserting and
   reports any backend error.

Requirements
------------

Both database drivers ship as part of |addon|'s install:

.. code-block:: text

   psycopg2-binary>=2.9.9   # PostgreSQL
   PyMySQL>=1.0.0           # MySQL

``psycopg2-binary`` is the binary distribution, so the widget runs on
macOS, Linux and Windows without a C compiler or the ``libpq``
development headers. ``PyMySQL`` is pure Python and has no native
build step. For production deployments you may swap either driver for
its source-distribution counterpart (``psycopg2`` or ``mysqlclient``)
in your environment.
