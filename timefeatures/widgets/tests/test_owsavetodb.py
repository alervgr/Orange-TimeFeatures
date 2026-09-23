"""Tests para timefeatures.widgets.owsavetodb.

Cubre las utilidades puras y el ``_UploadWorker`` completo. El worker se
prueba siempre sobre SQLite y, además, contra PostgreSQL y MySQL reales
cuando se definen estas variables de entorno (URLs de SQLAlchemy que
apunten a bases de datos desechables):

    TIMEFEATURES_TEST_POSTGRES_URL=postgresql://user:pass@localhost:5432/db
    TIMEFEATURES_TEST_MYSQL_URL=mysql://user:pass@localhost:3306/db

El envío de email no se prueba (está desactivado).
"""
import os
import tempfile
import threading
import time
import unittest
from datetime import datetime
from unittest import mock

import numpy as np
import Orange
from Orange.widgets.tests.base import WidgetTest

from timefeatures.widgets import owsavetodb
from timefeatures.widgets.owsavetodb import (
    TABLE_NAME_REGEX,
    _DIALECTS,
    _Dialect,
    _MySQLDialect,
    _PostgresDialect,
    _UploadWorker,
    _WRITE_MODE_KEYS,
    _create_sqlalchemy_engine,
    _dataframe_for_sql_export,
    _iter_dataframe_chunks,
    _pandas_if_exists,
    _sql_export_variables,
    owsavetodb as OWSaveToDB,
    quote_ident,
)


class TestTableNameRegex(unittest.TestCase):
    """Whitelist para el nombre de tabla provisto por el usuario."""

    def test_accepts_valid_names(self):
        for name in ["a", "abc", "_x", "table_1", "A1B2", "_" * 5,
                     "a" * 63]:  # 63 = límite de PostgreSQL
            with self.subTest(name=name):
                self.assertIsNotNone(TABLE_NAME_REGEX.match(name))

    def test_rejects_leading_digit(self):
        self.assertIsNone(TABLE_NAME_REGEX.match("1abc"))

    def test_rejects_special_chars(self):
        for name in ["a-b", "a b", "a.b", "a;b", "a'b", 'a"b', "a/b", "a*"]:
            with self.subTest(name=name):
                self.assertIsNone(TABLE_NAME_REGEX.match(name))

    def test_rejects_empty(self):
        self.assertIsNone(TABLE_NAME_REGEX.match(""))

    def test_rejects_too_long(self):
        # PostgreSQL: 63 caracteres es el máximo
        self.assertIsNone(TABLE_NAME_REGEX.match("a" * 64))

    def test_rejects_sql_injection_attempts(self):
        for name in [
            "'); DROP TABLE x;--",
            "x'; SELECT 1",
            "x OR 1=1",
            "x; --",
            "x` UNION SELECT",
        ]:
            with self.subTest(name=name):
                self.assertIsNone(TABLE_NAME_REGEX.match(name))


class TestQuoteIdent(unittest.TestCase):
    """Escapado de identificadores estilo PostgreSQL: envolver en "..." y
    doblar cualquier `"` interno."""

    def test_wraps_with_double_quotes(self):
        self.assertEqual(quote_ident("col"), '"col"')

    def test_escapes_internal_quote(self):
        self.assertEqual(quote_ident('a"b'), '"a""b"')

    def test_preserves_spaces_and_punctuation(self):
        # Espacios y `;` no rompen el quoting (PostgreSQL los acepta dentro
        # de un identificador entrecomillado).
        self.assertEqual(quote_ident("a b"), '"a b"')
        self.assertEqual(quote_ident("foo;bar"), '"foo;bar"')

    def test_handles_non_string_input(self):
        # str() coerciona enteros (defensa por si llega un valor no-string).
        self.assertEqual(quote_ident(42), '"42"')

    def test_prevents_quote_breakout(self):
        # Aunque el atacante meta varias comillas, el resultado siempre
        # empieza y acaba con `"` y los `"` internos van doblados.
        raw = '"; DROP TABLE foo; --'
        quoted = quote_ident(raw)
        self.assertTrue(quoted.startswith('"'))
        self.assertTrue(quoted.endswith('"'))
        body = quoted[1:-1]
        # cada `"` original aparece como `""` en el cuerpo
        self.assertEqual(body.count('"'), 2 * raw.count('"'))


# --------------------------------------------------------------------- #
#  Dialect abstraction
# --------------------------------------------------------------------- #
class TestPostgresDialect(unittest.TestCase):
    def setUp(self):
        self.dialect = _PostgresDialect()

    def test_quote_ident_matches_module_helper(self):
        # The legacy `quote_ident` is PostgreSQL-style; the dialect must
        # produce the same output for the same input.
        for name in ("col", 'a"b', "table 1", "foo;bar"):
            self.assertEqual(self.dialect.quote_ident(name), quote_ident(name))

    def test_column_types(self):
        self.assertEqual(
            self.dialect.column_type(Orange.data.DiscreteVariable("d")),
            "VARCHAR(255)",
        )
        self.assertEqual(
            self.dialect.column_type(Orange.data.ContinuousVariable("c")),
            "DOUBLE PRECISION",
        )
        self.assertEqual(
            self.dialect.column_type(Orange.data.TimeVariable("t")),
            "TIMESTAMP",
        )
        self.assertEqual(
            self.dialect.column_type(Orange.data.StringVariable("s")),
            "TEXT",
        )


class TestMySQLDialect(unittest.TestCase):
    def setUp(self):
        self.dialect = _MySQLDialect()

    def test_quote_ident_uses_backticks(self):
        self.assertEqual(self.dialect.quote_ident("col"), "`col`")

    def test_quote_ident_escapes_internal_backtick(self):
        # MySQL doubles an internal backtick to escape it.
        self.assertEqual(self.dialect.quote_ident("a`b"), "`a``b`")

    def test_quote_ident_keeps_double_quotes_literal(self):
        # MySQL treats `"` as a string delimiter (depending on sql_mode),
        # but inside a backtick-quoted identifier it's just data. The
        # dialect should not transform it.
        self.assertEqual(self.dialect.quote_ident('a"b'), '`a"b`')

    def test_column_types(self):
        self.assertEqual(
            self.dialect.column_type(Orange.data.DiscreteVariable("d")),
            "VARCHAR(255)",
        )
        # DOUBLE (not FLOAT(10)) — MySQL's FLOAT(M,D) syntax means
        # precision/scale, which is not what we want.
        self.assertEqual(
            self.dialect.column_type(Orange.data.ContinuousVariable("c")),
            "DOUBLE",
        )
        # DATETIME (not TIMESTAMP) — MySQL's TIMESTAMP is limited to
        # 1970-2038.
        self.assertEqual(
            self.dialect.column_type(Orange.data.TimeVariable("t")),
            "DATETIME",
        )
        self.assertEqual(
            self.dialect.column_type(Orange.data.StringVariable("s")),
            "TEXT",
        )

    def test_unknown_variable_falls_back_to_text(self):
        class _Mystery:
            name = "x"
        self.assertEqual(self.dialect.column_type(_Mystery()), "TEXT")


class TestDialectsRegistry(unittest.TestCase):
    def test_default_registry_has_both(self):
        self.assertIn("PostgreSQL", _DIALECTS)
        self.assertIn("MySQL", _DIALECTS)

    def test_postgres_backend_factory_returns_orange_backend_or_none(self):
        # On a clean install Orange's PostgreSQL backend is present, but if
        # not we expect None instead of an exception.
        factory = _DIALECTS["PostgreSQL"].backend_factory()
        if factory is not None:
            self.assertEqual(factory.display_name, "PostgreSQL")

    def test_mysql_backend_factory_returns_our_wrapper(self):
        from timefeatures.widgets.owsavetodb import _MySQLBackend
        factory = _DIALECTS["MySQL"].backend_factory()
        self.assertIs(factory, _MySQLBackend)


class TestDataFrameExport(unittest.TestCase):
    def test_uses_legacy_column_order_with_class_and_metas(self):
        attrs = [
            Orange.data.ContinuousVariable("a"),
            Orange.data.ContinuousVariable("b"),
        ]
        class_var = Orange.data.DiscreteVariable("c", values=["x", "y"])
        metas = [
            Orange.data.StringVariable("m1"),
            Orange.data.StringVariable("m2"),
        ]
        domain = Orange.data.Domain(attrs, class_var, metas)
        table = Orange.data.Table.from_numpy(
            domain,
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([0, 1]),
            np.array([["foo", "bar"], ["baz", "qux"]], dtype=object),
        )

        variables = _sql_export_variables(table)
        frame, _ = _dataframe_for_sql_export(table)

        self.assertEqual([v.name for v in variables], ["c", "m2", "m1", "a", "b"])
        self.assertEqual(list(frame.columns), ["c", "m2", "m1", "a", "b"])
        self.assertEqual(frame.iloc[0].tolist(), ["x", "bar", "foo", 1.0, 2.0])

    def test_without_class_keeps_metas_before_attributes(self):
        attrs = [Orange.data.ContinuousVariable("a")]
        metas = [Orange.data.StringVariable("m1")]
        domain = Orange.data.Domain(attrs, metas=metas)
        table = Orange.data.Table.from_numpy(
            domain,
            np.array([[1.0]]),
            None,
            np.array([["foo"]], dtype=object),
        )

        frame, _ = _dataframe_for_sql_export(table)

        self.assertEqual(list(frame.columns), ["m1", "a"])
        self.assertEqual(frame.iloc[0].tolist(), ["foo", 1.0])

    def test_dataframe_chunks_include_all_rows(self):
        import pandas as pd

        frame = pd.DataFrame({"x": range(5)})
        chunks = list(_iter_dataframe_chunks(frame, chunksize=2))

        self.assertEqual([len(chunk) for chunk in chunks], [2, 2, 1])
        self.assertEqual(
            [value for chunk in chunks for value in chunk["x"].tolist()],
            [0, 1, 2, 3, 4],
        )

    def test_dataframe_chunks_yield_empty_frame(self):
        import pandas as pd

        frame = pd.DataFrame({"x": []})
        chunks = list(_iter_dataframe_chunks(frame, chunksize=2))

        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].empty)


# --------------------------------------------------------------------- #
#  Write-mode wiring
# --------------------------------------------------------------------- #
class TestPandasIfExists(unittest.TestCase):
    def test_create_first_chunk_fails_then_appends(self):
        # ``create`` keeps the original guard: fail on the first chunk if
        # the target table already exists; from chunk 1 onwards we
        # always append (the table we just created).
        self.assertEqual(_pandas_if_exists("create", 0), "fail")
        self.assertEqual(_pandas_if_exists("create", 1), "append")
        self.assertEqual(_pandas_if_exists("create", 42), "append")

    def test_overwrite_uses_same_pattern_as_create(self):
        # Overwrite uploads into a fresh staging table, so the chunks
        # themselves use create-style semantics.
        self.assertEqual(_pandas_if_exists("overwrite", 0), "fail")
        self.assertEqual(_pandas_if_exists("overwrite", 5), "append")

    def test_append_always_appends(self):
        # ``append`` must never raise: pandas creates the table if it
        # doesn't exist and appends if it does.
        for i in range(0, 10):
            self.assertEqual(_pandas_if_exists("append", i), "append")


class TestWriteModeKeys(unittest.TestCase):
    def test_known_modes(self):
        # The Setting persists these literal strings, so a typo here
        # would silently fall back to "create" in the widget.
        self.assertEqual(
            set(_WRITE_MODE_KEYS), {"create", "overwrite", "append"},
        )


class TestReplaceTableSql(unittest.TestCase):
    def test_postgres_drops_then_renames_inside_transaction(self):
        self.assertEqual(
            _PostgresDialect().replace_table_sql("stg", "t", True),
            ['DROP TABLE "t"', 'ALTER TABLE "stg" RENAME TO "t"'],
        )
        self.assertEqual(
            _PostgresDialect().replace_table_sql("stg", "t", False),
            ['ALTER TABLE "stg" RENAME TO "t"'],
        )

    def test_mysql_swaps_both_names_in_one_statement(self):
        # DROP + RENAME would auto-commit separately in MySQL; the
        # multi-table RENAME TABLE is atomic.
        self.assertEqual(
            _MySQLDialect().replace_table_sql("stg", "t", True),
            ["RENAME TABLE `t` TO `stg_old`, `stg` TO `t`",
             "DROP TABLE `stg_old`"],
        )
        self.assertEqual(
            _MySQLDialect().replace_table_sql("stg", "t", False),
            ["RENAME TABLE `stg` TO `t`"],
        )


# --------------------------------------------------------------------- #
#  _UploadWorker contra una base de datos real
# --------------------------------------------------------------------- #
TEST_TABLE = "tf_upload_test"


class _SQLiteDialect(_Dialect):
    """Dialecto mínimo para ejecutar el worker sobre SQLite. Usa el
    intercambio por defecto (``DROP TABLE`` + ``ALTER TABLE ... RENAME``)."""
    name = "SQLite"
    sqlalchemy_drivername = "sqlite"
    quote_ident = staticmethod(quote_ident)


def _params_from_url(url):
    from sqlalchemy.engine import make_url
    url = make_url(url)
    return {
        "host": url.host, "port": url.port, "database": url.database,
        "username": url.username, "password": url.password,
    }


def sqlite_params(path):
    return {
        "host": None, "port": None, "database": path,
        "username": None, "password": None,
    }


def numeric_table(n_rows, offset=0):
    domain = Orange.data.Domain([Orange.data.ContinuousVariable("x")])
    return Orange.data.Table.from_numpy(
        domain, (np.arange(n_rows, dtype=float) + offset).reshape(-1, 1)
    )


def make_upload_worker(dialect, connection_params, table, mode="create",
                       table_name=TEST_TABLE):
    return _UploadWorker(
        table=table,
        dialect=dialect,
        connection_params=connection_params,
        metadata={
            "table_name": table_name,
            "params": {
                "dataset_name": table_name,
                "created_at": datetime.now(),
                "row_count": len(table),
                "col_count": len(table.domain),
                "target_type": "None",
                "class_name": None,
            },
        },
        email_params={"mail": ""},
        write_mode=mode,
    )


# --------------------------------------------------------------------- #
#  Utilidades para probar los hilos de los widgets
# --------------------------------------------------------------------- #
class Gate:
    """Detiene un worker al empezar ``run()`` hasta que la prueba lo
    suelta, para poder actuar mientras la operación está en curso."""

    def __init__(self):
        self.reached = threading.Event()
        self.release = threading.Event()


def gated(worker_cls, gate):
    class Gated(worker_cls):
        def run(self):
            gate.reached.set()
            # Con un tope: si el código bajo prueba bloquea el hilo
            # principal, la prueba falla en vez de colgarse.
            gate.release.wait(5)
            super().run()
    return Gated


def wait_until(condition, timeout=5.0):
    """Procesa eventos de Qt hasta que ``condition()`` sea cierta."""
    from AnyQt.QtWidgets import QApplication
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


def thread_finished(thread):
    try:
        return thread.isFinished()
    except RuntimeError:  # objeto Qt ya destruido tras terminar
        return True


class _UploadWorkerCases:
    """Casos comunes a todos los motores. Cada subclase define ``dialect``
    y ``connection_params``. Con 2 500 filas y bloques de 1 000, una
    subida tiene tres bloques y se puede interrumpir a mitad."""

    dialect = None
    connection_params = None

    def setUp(self):
        from sqlalchemy import text
        self.text = text
        self.engine = _create_sqlalchemy_engine(
            self.dialect, **self.connection_params
        )
        self._cleanup()
        self._baseline = self._tables() - {"datasets"}

    def tearDown(self):
        self._cleanup()
        # Cualquier tabla nueva que quede (p. ej. una de preparación)
        # se elimina para no contaminar la siguiente prueba.
        qi = self.dialect.quote_ident
        with self.engine.begin() as connection:
            for name in self._tables() - self._baseline - {"datasets"}:
                connection.execute(self.text(f"DROP TABLE {qi(name)}"))
        self.engine.dispose()

    # --- helpers ------------------------------------------------------ #
    def _tables(self):
        from sqlalchemy import inspect
        return set(inspect(self.engine).get_table_names())

    def _cleanup(self):
        qi = self.dialect.quote_ident
        with self.engine.begin() as connection:
            connection.execute(
                self.text(f"DROP TABLE IF EXISTS {qi(TEST_TABLE)}")
            )
        self._delete_metadata_row()

    def _delete_metadata_row(self):
        if "datasets" not in self._tables():
            return
        qi = self.dialect.quote_ident
        with self.engine.begin() as connection:
            connection.execute(
                self.text(
                    f"DELETE FROM {qi('datasets')} WHERE {qi('name')} = :n"
                ),
                {"n": TEST_TABLE},
            )

    def _rows(self):
        qi = self.dialect.quote_ident
        with self.engine.connect() as connection:
            result = connection.execute(self.text(
                f"SELECT {qi('x')} FROM {qi(TEST_TABLE)} ORDER BY {qi('x')}"
            ))
            return [int(value) for (value,) in result]

    def _registered_rows(self):
        if "datasets" not in self._tables():
            return None
        qi = self.dialect.quote_ident
        with self.engine.connect() as connection:
            row = connection.execute(
                self.text(
                    f"SELECT {qi('rows')} FROM {qi('datasets')} "
                    f"WHERE {qi('name')} = :n"
                ),
                {"n": TEST_TABLE},
            ).first()
        return None if row is None else int(row[0])

    def assertNoLeftoverTables(self):
        extra = self._tables() - self._baseline - {"datasets", TEST_TABLE}
        self.assertEqual(extra, set())

    def _upload(self, mode, n_rows, offset=0, fail_on_chunk=None,
                cancel_on_chunk=None):
        """Ejecuta el worker de forma síncrona y devuelve
        ``{"finished": segundos}`` o ``{"failed": mensaje}``."""
        worker = make_upload_worker(
            self.dialect, self.connection_params,
            numeric_table(n_rows, offset), mode,
        )
        result = {}
        worker.finished.connect(lambda t: result.setdefault("finished", t))
        worker.failed.connect(lambda m: result.setdefault("failed", m))
        if cancel_on_chunk is not None:
            # Pulsar Cancel mientras se sube ese bloque: el worker lo
            # detecta al empezar el siguiente.
            marker = f"Uploading rows {cancel_on_chunk + 1}/"
            worker.status_changed.connect(
                lambda msg: setattr(worker, "is_cancelled", True)
                if msg.startswith(marker) else None
            )

        real_chunks = owsavetodb._iter_dataframe_chunks

        def chunks(frame, *args, **kwargs):
            for index, chunk in enumerate(real_chunks(frame, *args, **kwargs)):
                if index == fail_on_chunk:
                    raise RuntimeError("simulated failure mid-upload")
                yield chunk

        with mock.patch.object(owsavetodb, "_iter_dataframe_chunks", chunks):
            worker.run()
        return result

    # --- create ------------------------------------------------------- #
    def test_create_uploads_rows_and_registers_them(self):
        self.assertIn("finished", self._upload("create", 2500))
        self.assertEqual(self._rows(), list(range(2500)))
        self.assertEqual(self._registered_rows(), 2500)
        self.assertNoLeftoverTables()

    def test_create_refuses_table_without_metadata_row(self):
        self._upload("create", 10)
        self._delete_metadata_row()
        result = self._upload("create", 5, offset=100)
        self.assertIn("already exists", result["failed"])
        self.assertEqual(self._rows(), list(range(10)))

    def test_create_cancelled_mid_upload_leaves_nothing(self):
        result = self._upload("create", 2500, cancel_on_chunk=1)
        self.assertIn("cancelled", result["failed"])
        self.assertNotIn(TEST_TABLE, self._tables())
        self.assertIsNone(self._registered_rows())
        self.assertNoLeftoverTables()
        # Un nuevo intento con el mismo nombre no choca con restos.
        self.assertIn("finished", self._upload("create", 10))

    # --- overwrite ---------------------------------------------------- #
    def test_overwrite_replaces_rows_and_metadata(self):
        self._upload("create", 2500)
        self.assertIn("finished", self._upload("overwrite", 10, offset=5000))
        self.assertEqual(self._rows(), list(range(5000, 5010)))
        self.assertEqual(self._registered_rows(), 10)
        self.assertNoLeftoverTables()

    def test_overwrite_failure_mid_upload_keeps_previous_table(self):
        self._upload("create", 10)
        result = self._upload("overwrite", 2500, offset=5000, fail_on_chunk=1)
        self.assertIn("simulated failure", result["failed"])
        self.assertEqual(self._rows(), list(range(10)))
        self.assertEqual(self._registered_rows(), 10)
        self.assertNoLeftoverTables()

    def test_overwrite_cancelled_mid_upload_keeps_previous_table(self):
        self._upload("create", 10)
        result = self._upload("overwrite", 2500, offset=5000, cancel_on_chunk=1)
        self.assertIn("cancelled", result["failed"])
        self.assertEqual(self._rows(), list(range(10)))
        self.assertEqual(self._registered_rows(), 10)
        self.assertNoLeftoverTables()

    # --- append ------------------------------------------------------- #
    def test_append_adds_rows_and_updates_count(self):
        self._upload("create", 10)
        self.assertIn("finished", self._upload("append", 5, offset=100))
        self.assertEqual(self._rows(), list(range(10)) + list(range(100, 105)))
        self.assertEqual(self._registered_rows(), 15)

    def test_append_failure_mid_upload_keeps_existing_rows(self):
        self._upload("create", 10)
        result = self._upload("append", 2500, offset=5000, fail_on_chunk=1)
        self.assertIn("simulated failure", result["failed"])
        self.assertEqual(self._rows(), list(range(10)))
        self.assertEqual(self._registered_rows(), 10)

    def test_append_into_missing_table_creates_it(self):
        self.assertIn("finished", self._upload("append", 10))
        self.assertEqual(self._rows(), list(range(10)))
        self.assertEqual(self._registered_rows(), 10)
        self.assertNoLeftoverTables()

    def test_append_failure_into_missing_table_leaves_nothing(self):
        result = self._upload("append", 2500, fail_on_chunk=1)
        self.assertIn("simulated failure", result["failed"])
        self.assertNotIn(TEST_TABLE, self._tables())
        self.assertNoLeftoverTables()


class TestUploadWorkerSQLite(_UploadWorkerCases, unittest.TestCase):
    dialect = _SQLiteDialect()

    def setUp(self):
        handle, self._path = tempfile.mkstemp(suffix=".sqlite")
        os.close(handle)
        self.connection_params = sqlite_params(self._path)
        super().setUp()

    def tearDown(self):
        super().tearDown()
        os.remove(self._path)


@unittest.skipUnless(os.environ.get("TIMEFEATURES_TEST_POSTGRES_URL"),
                     "TIMEFEATURES_TEST_POSTGRES_URL not set")
class TestUploadWorkerPostgres(_UploadWorkerCases, unittest.TestCase):
    dialect = _DIALECTS["PostgreSQL"]
    connection_params = _params_from_url(
        os.environ.get("TIMEFEATURES_TEST_POSTGRES_URL") or "postgresql://"
    )


@unittest.skipUnless(os.environ.get("TIMEFEATURES_TEST_MYSQL_URL"),
                     "TIMEFEATURES_TEST_MYSQL_URL not set")
class TestUploadWorkerMySQL(_UploadWorkerCases, unittest.TestCase):
    dialect = _DIALECTS["MySQL"]
    connection_params = _params_from_url(
        os.environ.get("TIMEFEATURES_TEST_MYSQL_URL") or "mysql://"
    )


# --------------------------------------------------------------------- #
#  Widget: hilo de subida
# --------------------------------------------------------------------- #
class TestSaveToDbUploadThread(WidgetTest):
    @unittest.skip("widget layout exceeds 800px; out of scope")
    def test_minimum_size(self):
        pass

    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".sqlite")
        os.close(handle)
        self.addCleanup(os.remove, self.path)
        self.dialect = _SQLiteDialect()
        patcher = mock.patch.dict(owsavetodb._DIALECTS, {"SQLite": self.dialect})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.gate = Gate()
        self.addCleanup(self.gate.release.set)
        patcher = mock.patch.object(
            owsavetodb, "_UploadWorker", gated(_UploadWorker, self.gate)
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        self.widget = self.create_widget(OWSaveToDB)
        self.send_signal(self.widget.Inputs.data, numeric_table(3000))
        self.widget.selected_backend = "SQLite"
        params = sqlite_params(self.path)
        for attr in ("host", "port", "database", "username", "password"):
            setattr(self.widget, attr, params[attr])

    def _start_upload(self):
        # ``_check_db_settings`` relee los campos de texto, que no
        # admiten una ruta de SQLite; los parámetros ya están puestos.
        with mock.patch.object(self.widget, "_check_db_settings"):
            self.widget._start_upload(TEST_TABLE, None)
        self.assertTrue(wait_until(self.gate.reached.is_set))

    def test_upload_completes(self):
        self._start_upload()
        self.gate.release.set()
        self.assertTrue(wait_until(lambda: not self.widget._uploading))
        self.assertIn("completed", self.widget.connection_status_label.text())
        # El hilo avisa al terminar y el widget suelta sus referencias.
        self.assertTrue(wait_until(lambda: self.widget._upload_thread is None))
        self.assertIsNone(self.widget._upload_worker)

    def test_closing_during_upload_does_not_block_and_cancels_it(self):
        self._start_upload()
        worker, thread = self.widget._upload_worker, self.widget._upload_thread

        started = time.monotonic()
        self.widget.onDeleteWidget()
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertTrue(worker.is_cancelled)

        self.gate.release.set()
        self.assertTrue(wait_until(lambda: thread_finished(thread)))
        # La subida cancelada se deshace: no queda la tabla.
        from sqlalchemy import inspect
        engine = _create_sqlalchemy_engine(
            self.dialect, **sqlite_params(self.path)
        )
        self.assertNotIn(TEST_TABLE, inspect(engine).get_table_names())
        engine.dispose()


if __name__ == "__main__":
    unittest.main()
