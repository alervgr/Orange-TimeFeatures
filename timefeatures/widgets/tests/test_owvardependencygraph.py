"""Tests para timefeatures.widgets.owvardependencygraph."""
import unittest

import numpy as np

import Orange
from Orange.data import DiscreteVariable, Domain, Instance, Table

from timefeatures.widgets.owvardependencygraph import (
    _expression_or_none,
    _sanitize_name,
    build_dependency_network,
)


def make_config(rows):
    """Construye una tabla de configuración (Variable, Expression).

    ``rows`` es una lista de ``(nombre, expresión_o_None)`` igual que la que
    produce el Time Features Constructor.
    """
    names = [n for n, _ in rows]
    unique_exprs = sorted({e for _, e in rows if e is not None})
    var_col = DiscreteVariable("Variable", values=names)
    expr_col = DiscreteVariable("Expression", values=unique_exprs or ["_"])
    domain = Domain([var_col, expr_col])
    instances = [Instance(domain, [n, e]) for n, e in rows]
    return Table.from_list(domain, instances)


# --------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------- #
class TestSanitizeName(unittest.TestCase):
    def test_replaces_spaces_and_hyphens(self):
        self.assertEqual(_sanitize_name("a b-c"), "a_b_c")

    def test_passthrough(self):
        self.assertEqual(_sanitize_name("X_1"), "X_1")

    def test_coerces_non_string(self):
        self.assertEqual(_sanitize_name(42), "42")


class TestExpressionOrNone(unittest.TestCase):
    def test_returns_text_for_valid_expr(self):
        self.assertEqual(_expression_or_none("shift(x,-1)"), "shift(x,-1)")

    def test_nan_returns_none(self):
        self.assertIsNone(_expression_or_none(float("nan")))

    def test_empty_returns_none(self):
        self.assertIsNone(_expression_or_none(""))

    def test_question_mark_returns_none(self):
        # Orange representa valores faltantes como "?" al imprimirlos.
        self.assertIsNone(_expression_or_none("?"))

    def test_none_string_returns_none(self):
        self.assertIsNone(_expression_or_none("None"))


# --------------------------------------------------------------------- #
#  build_dependency_network
# --------------------------------------------------------------------- #
class TestBuildDependencyNetwork(unittest.TestCase):
    def _names(self, network):
        return [str(v) for v in network.nodes[:, "var_name"].metas.ravel()]

    def _types(self, network):
        # var_type: 0 = Derived, 1 = Original
        return [int(v) for v in network.nodes[:, "var_type"].metas.ravel()]

    def _edges(self, network):
        """Devuelve un set de pares (src_idx, dst_idx)."""
        m = network.edges[0].edges.tocoo()
        return set(zip(m.row.tolist(), m.col.tolist()))

    def test_simple_chain(self):
        # X1 → X2 → X3 (cada uno depende del siguiente).
        # X3 es original (sin expresión).
        table = make_config([
            ("X1", "shift(X2,-1)"),
            ("X2", "shift(X3,-1)"),
            ("X3", None),
        ])
        net = build_dependency_network(table)
        self.assertEqual(self._names(net), ["X1", "X2", "X3"])
        # 0=Derived, 1=Original
        self.assertEqual(self._types(net), [0, 0, 1])
        self.assertEqual(self._edges(net), {(0, 1), (1, 2)})

    def test_self_loop(self):
        # X1 := shift(X1, -1) → arista X1 → X1
        table = make_config([
            ("X1", "shift(X1,-1)"),
            ("X2", None),
        ])
        net = build_dependency_network(table)
        self.assertEqual(self._edges(net), {(0, 0)})

    def test_no_duplicate_edges(self):
        # X1 := X2 + X2 * 2 referencia X2 dos veces → una sola arista.
        table = make_config([
            ("X1", "X2 + X2 * 2"),
            ("X2", None),
        ])
        net = build_dependency_network(table)
        self.assertEqual(self._edges(net), {(0, 1)})

    def test_word_boundaries_no_substring_match(self):
        # "X1" no debe matchear dentro de "X10".
        table = make_config([
            ("X1", None),
            ("X10", "shift(X1,-1)"),
        ])
        net = build_dependency_network(table)
        # X10 depende de X1, no de sí mismo.
        self.assertEqual(self._edges(net), {(1, 0)})

    def test_variable_name_with_space_or_hyphen(self):
        # El sanitizado convierte " " y "-" a "_". La expresión usa el
        # nombre saneado.
        table = make_config([
            ("var one", "shift(var_two,-1)"),
            ("var-two", None),
        ])
        net = build_dependency_network(table)
        self.assertEqual(self._names(net), ["var_one", "var_two"])
        self.assertEqual(self._edges(net), {(0, 1)})

    def test_no_dependencies(self):
        # Variables sin expresión → todas originales, cero aristas.
        table = make_config([
            ("X1", None),
            ("X2", None),
        ])
        net = build_dependency_network(table)
        self.assertEqual(self._types(net), [1, 1])
        self.assertEqual(self._edges(net), set())

    def test_dependency_to_unknown_variable_is_ignored(self):
        # "foo" no es ninguna variable de la tabla → no se crea arista.
        table = make_config([
            ("X1", "foo + 1"),
            ("X2", None),
        ])
        net = build_dependency_network(table)
        self.assertEqual(self._edges(net), set())


if __name__ == "__main__":
    unittest.main()
