Changelog
=========

Unreleased
----------

**Load from DB (new widget)**

- Lists every dataset registered in the ``datasets`` metadata table
  via SQLAlchemy and pulls the selected one back into Orange as an
  ``Orange.data.Table`` using ``pandas.read_sql`` +
  ``Orange.data.pandas_compat.table_from_frame``.
- Offers a Class column combo populated from the dataset's columns,
  pre-selected with the user's persisted choice or the ``class_name``
  recorded by Save to DB. The output ``Table`` already exposes the
  chosen column as ``domain.class_var``, so no Select Columns widget
  is needed downstream.
- Same dialect selector (PostgreSQL / MySQL), connection-status
  label and ``QThread`` worker pattern as Save to DB. Both the
  metadata listing and the actual table read happen off the GUI
  thread.
- Workflow-persisted settings: ``selected_dataset`` and
  ``selected_class`` are ``Setting(..., schema_only=True)``, restored
  as soon as the connection comes back up on reload.

**Variable Dependency Graph**

- *New:* edges now carry **numeric weights** equal to the largest
  ``|argument|`` among the temporal calls (``shift``, ``sum``, ``mean``,
  ``count``, ``min``, ``max``, ``sd``) in the source expression that
  reference the dependency. Plain (non-temporal) references default to
  weight ``1``. The weights live in ``network.edges[0].edges.data``
  and are consumable by any downstream Network widget.
- *New:* the graph is now **directed** (``DirectedEdges``). Previously
  the sparse matrix was passed straight to ``Network`` which auto-wrapped
  it as ``UndirectedEdges`` — a latent bug since A→B is not the same as
  B→A in a dependency graph.
- *New:* nodes carry an extra ``expression`` meta (literal expression
  text, empty for original variables) so the Network Explorer can show
  the formula as the node label.
- *New:* ``Warning.no_derived`` fires when every input row is an
  original variable — usually a sign the user wired the data output
  of Time Features Constructor instead of the variable-definitions
  one.
- Refactor: flattened the ``from_row_col`` / ``grafo`` decorator pattern
  into a single, documented ``build_dependency_network`` function.
- Performance: O(n³) → O(n²) by precomputing a ``name → index`` map and
  using a ``set`` to dedupe dependencies.
- Bugfix: an ``IndexError`` was raised when the input domain had a
  single column. The widget now checks ``len(domain) >= 2`` before
  reading the second column.
- Removed 65 lines of dead commented-out code.

**Time Features Constructor**

- *New:* **chained descriptors**. A descriptor can now reference
  another derived descriptor in its expression (e.g. ``X2 :=
  shift(X1, -1)`` with ``X1`` itself defined a few rows above). The
  widget topologically sorts the descriptors and cascades the
  transforms — each step runs against the table state produced by
  the previous one, so ``X2`` sees ``X1`` as a regular column.
  Cycles (e.g. ``X1 := X2 + 1`` together with ``X2 := X1 + 1``) raise
  a *Circular dependency between descriptors: X1, X2* error. Errors
  during evaluation are reported per-descriptor so the failing row is
  obvious.
- *Fixed (critical):* time-window functions used to lose context every
  5 000 rows because Orange chunks tables during ``transform``. The
  widget now caches the full source per ``FeatureFunc`` and returns the
  right slice for each chunk, so ``shift(x, -20)`` is correct across
  chunk boundaries on datasets of any size.
- *Fixed:* "Variables to generate" was being cleared after every Send
  and on every input change, so workflow save / reload lost the
  definitions. Storage moved from ``ContextSetting`` to
  ``Setting(..., schema_only=True)`` (matching upstream Orange v4) and
  the editor is now restored from the persisted descriptors on every
  input. The legacy context handler stays around to migrate old
  workflows.
- Each **Send** re-transforms the *original* input table instead of the
  cumulative output, removing the implicit "consume on Send" semantics
  and the duplicate-name errors that came with it.
- ``eval`` hardened: ``__builtins__`` replaced with an empty dict in
  the expression evaluator. Only the curated whitelist
  (safe builtins + ``math`` + selected ``random`` / ``numpy``
  helpers) is exposed.
- Refactor: ``modificar_expression`` collapsed from 7 near-identical
  loops to a single regex-driven pass.
- Bugfix: ``FeatureEditor.editorData`` returned the variable name as
  the expression.

**Save to DB**

- *New:* **write mode** selector with three options — *Create new*
  (default, fail if the target exists), *Overwrite* (drop and
  recreate the table and its ``datasets`` row), *Append* (keep
  existing rows and add the new ones). Re-running a workflow no
  longer breaks. The persisted ``write_mode`` Setting defaults to
  ``"create"`` so old workflows keep their previous behaviour. After
  the upload, the widget runs ``SELECT COUNT(*)`` and rewrites the
  ``datasets`` row with the actual total, so the registry stays
  accurate across appends.
- *New:* **MySQL support**. The connection panel now exposes a
  database-type selector (PostgreSQL / MySQL). Per-dialect column
  types and identifier quoting (``"name"`` vs ``\`name\```) live in
  a ``_Dialect`` abstraction.
- *Performance:* uploads now go through
  ``Orange.data.pandas_compat.table_to_frame`` +
  ``DataFrame.to_sql(method='multi', chunksize=1000)`` over a
  SQLAlchemy engine. The old row-per-INSERT loop is gone — typical
  speedups are 50-100×, especially over the network. Identifier
  quoting and per-column DDL types are emitted by the SQLAlchemy
  dialect (``DOUBLE_PRECISION`` on PostgreSQL, ``DOUBLE`` on MySQL,
  ``DATETIME`` instead of ``TIMESTAMP`` on MySQL to dodge the 2038
  cap, ``VARCHAR(255)`` / ``TEXT`` everywhere else).
- *UX:* the upload runs in a background ``QThread`` (``_UploadWorker``),
  so the canvas stays interactive on long writes. A status label under
  the connection box reports *Not connected* / *Connected to … : …* /
  *Uploading rows N/M…* / *Upload completed in Xs* / *Upload failed:
  …* with colour cues. **Save**, **Connect** and the form fields are
  disabled while the worker runs and re-enabled on success or
  failure; the widget aborts the thread cleanly on close.
- *Fixed (critical):* SQL injection in the metadata ``INSERT``. The
  query is now fully parametrised.
- Added an identifier whitelist
  (``^[A-Za-z_][A-Za-z0-9_]{0,62}$``) and a ``quote_ident`` helper that
  wraps identifiers with PostgreSQL-standard double quotes (with any
  internal ``"`` doubled). Used in every ``CREATE TABLE`` / ``INSERT
  INTO`` that touches user-supplied names.
- Replaced ``psycopg2`` with ``psycopg2-binary`` in the install
  requirements so installation works on macOS, Linux and Windows
  without a C compiler.
- ``setup.py`` now declares ``install_requires``; ``pip install`` was
  previously not pulling in any runtime dependency.

Testing
-------

The widget suite grew to **92 tests**, covering:

- Unit tests for ``modificar_expression``, the time-window helpers,
  ``_sanitize_name`` / ``_expression_or_none`` and ``_temporal_weights``.
- A regression test for the 5 000-row chunk bug
  (``shift(x, -20)`` over a 12 000-row table).
- ``eval``-safety tests asserting that ``__import__`` / ``open`` raise
  ``NameError`` while ``sqrt`` / ``abs`` still resolve.
- Widget-level tests (via ``Orange.widgets.tests.base.WidgetTest``) for
  descriptor persistence, including an end-to-end
  ``settingsHandler.pack_data`` / ``stored_settings=`` round-trip that
  mirrors what Orange does when saving and reopening a ``.ows`` file.
- Edge-weight tests covering single calls, three-argument families,
  mixed temporal / non-temporal references, max across multiple calls,
  and per-dependency independence.
