"""Tests para timefeatures.widgets.owsavetodb (helpers de saneamiento SQL).

Sólo cubre las utilidades puras; la conexión real a Postgres y el envío
de email no se testean aquí (requieren entorno con servidor).
"""
import unittest

import numpy as np
import Orange

from timefeatures.widgets.owsavetodb import (
    TABLE_NAME_REGEX,
    _DIALECTS,
    _MySQLDialect,
    _PostgresDialect,
    _dataframe_for_sql_export,
    _iter_dataframe_chunks,
    _sql_export_variables,
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


if __name__ == "__main__":
    unittest.main()
