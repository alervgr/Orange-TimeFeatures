"""Tests para timefeatures.widgets.owsavetodb (helpers de saneamiento SQL).

Sólo cubre las utilidades puras; la conexión real a Postgres y el envío
de email no se testean aquí (requieren entorno con servidor).
"""
import unittest

from timefeatures.widgets.owsavetodb import (
    TABLE_NAME_REGEX,
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


if __name__ == "__main__":
    unittest.main()
