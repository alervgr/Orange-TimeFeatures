"""Tests para timefeatures.widgets.owvardependencygraph."""
import unittest

import numpy as np

import Orange
from Orange.data import DiscreteVariable, Domain, Instance, Table

from timefeatures.widgets.owvardependencygraph import (
    _expression_or_none,
    _sanitize_name,
    _temporal_weights,
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
        # Igual que en el constructor: un nombre que empieza por dígito se
        # escribe con ``_`` delante en las expresiones.
        self.assertEqual(_sanitize_name(42), "_42")

    def test_matches_constructor_for_punctuation(self):
        self.assertEqual(_sanitize_name("temp (C)"), "temp__C_")


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

    def _weighted_edges(self, network):
        """Devuelve un dict {(src, dst): weight}."""
        m = network.edges[0].edges.tocoo()
        return {
            (int(r), int(c)): float(w)
            for r, c, w in zip(m.row, m.col, m.data)
        }

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

    def test_variable_name_with_punctuation_or_leading_digit(self):
        # Mismo saneado que el constructor: cualquier carácter no
        # alfanumérico pasa a "_" y un dígito inicial recibe "_" delante.
        table = make_config([
            ("X1", "shift(temp__C_, -3) + _1abc"),
            ("temp (C)", None),
            ("1abc", None),
        ])
        net = build_dependency_network(table)
        self.assertEqual(self._names(net), ["X1", "temp__C_", "_1abc"])
        self.assertEqual(
            self._weighted_edges(net), {(0, 1): 3.0, (0, 2): 1.0}
        )

    def test_no_dependencies(self):
        # Variables sin expresión → todas originales, cero aristas.
        table = make_config([
            ("X1", None),
            ("X2", None),
        ])
        net = build_dependency_network(table)
        self.assertEqual(self._types(net), [1, 1])
        self.assertEqual(self._edges(net), set())

    def test_names_inside_string_literals_are_not_references(self):
        table = make_config([
            ("X1", 'str(X2) + "X3"'),
            ("X2", None),
            ("X3", None),
        ])
        self.assertEqual(self._edges(build_dependency_network(table)), {(0, 1)})

    def test_called_functions_are_not_references(self):
        # Una columna llamada "sum" no depende de cada uso de sum(...).
        table = make_config([
            ("X1", "sum(X2, -1, 1)"),
            ("sum", None),
            ("X2", None),
        ])
        self.assertEqual(self._edges(build_dependency_network(table)), {(0, 2)})

    def test_names_bound_in_comprehensions_are_not_references(self):
        table = make_config([
            ("X1", "len([X3 for X3 in range(int(X2))])"),
            ("X2", None),
            ("X3", None),
        ])
        self.assertEqual(self._edges(build_dependency_network(table)), {(0, 1)})

    def test_unicode_names_are_normalised_like_python(self):
        # Python normaliza los identificadores con NFKC al analizar la
        # expresión: la ligadura "ﬁ" se lee como "fi".
        table = make_config([
            ("X1", "ﬁx + 1"),
            ("ﬁx", None),
        ])
        self.assertEqual(self._edges(build_dependency_network(table)), {(0, 1)})

    def test_unparsable_expression_has_no_edges_and_is_reported(self):
        table = make_config([
            ("X1", "X2 +"),
            ("X2", None),
            ("X3", "X2 * 2"),
        ])
        unparsable = []
        net = build_dependency_network(table, unparsable)
        self.assertEqual(self._edges(net), {(2, 1)})
        self.assertEqual(unparsable, ["X1"])

    def test_dependency_to_unknown_variable_is_ignored(self):
        # "foo" no es ninguna variable de la tabla → no se crea arista.
        table = make_config([
            ("X1", "foo + 1"),
            ("X2", None),
        ])
        net = build_dependency_network(table)
        self.assertEqual(self._edges(net), set())


# --------------------------------------------------------------------- #
#  _temporal_weights — peso máximo por variable en llamadas temporales
# --------------------------------------------------------------------- #
class TestTemporalWeights(unittest.TestCase):
    def test_no_temporal_calls(self):
        self.assertEqual(_temporal_weights("X + 1"), {})
        self.assertEqual(_temporal_weights(""), {})

    def test_shift_single(self):
        self.assertEqual(_temporal_weights("shift(X,-20)"), {"X": 20})
        self.assertEqual(_temporal_weights("shift(X,+5)"), {"X": 5})

    def test_three_arg_call(self):
        # max(|-5|, |10|) = 10
        self.assertEqual(_temporal_weights("sum(X,-5,10)"), {"X": 10})
        self.assertEqual(_temporal_weights("mean(X,-3,7)"), {"X": 7})

    def test_multiple_calls_take_max(self):
        # Dos llamadas sobre la misma variable: gana la de mayor |arg|.
        expr = "shift(X,-3) + mean(X,-7,7)"
        self.assertEqual(_temporal_weights(expr), {"X": 7})

    def test_per_variable_independent(self):
        expr = "shift(X,-5) + mean(Y,-10,10)"
        self.assertEqual(_temporal_weights(expr), {"X": 5, "Y": 10})

    def test_ignores_non_temporal_references(self):
        # Y aparece sin envoltura temporal → no contribuye al peso.
        self.assertEqual(
            _temporal_weights("shift(X,-5) + Y"),
            {"X": 5},
        )

    def test_spaces_and_nesting(self):
        self.assertEqual(
            _temporal_weights("abs(mean( X , -3 , 3 )) + shift(Y, -4)"),
            {"X": 3, "Y": 4},
        )

    def test_ignores_calls_inside_string_literals(self):
        self.assertEqual(
            _temporal_weights('shift(X,-2) + len("shift(X,-50)")'),
            {"X": 2},
        )

    def test_ignores_calls_with_another_shape(self):
        # Sólo cuentan las formas que evalúa el constructor: shift con un
        # desplazamiento, el resto con dos, y siempre enteros literales.
        self.assertEqual(_temporal_weights("shift(X, -1, 2)"), {})
        self.assertEqual(_temporal_weights("sum(X, -3)"), {})
        self.assertEqual(_temporal_weights("mean(X, -3, k)"), {})

    def test_unparsable_expression_has_no_weights(self):
        self.assertEqual(_temporal_weights("shift(X, -3"), {})


# --------------------------------------------------------------------- #
#  build_dependency_network — pesos en aristas
# --------------------------------------------------------------------- #
class TestEdgeWeights(unittest.TestCase):
    def _weighted_edges(self, network):
        m = network.edges[0].edges.tocoo()
        return {
            (int(r), int(c)): float(w)
            for r, c, w in zip(m.row, m.col, m.data)
        }

    def test_shift_weight(self):
        table = make_config([
            ("X1", "shift(X2,-20)"),
            ("X2", None),
        ])
        net = build_dependency_network(table)
        self.assertEqual(self._weighted_edges(net), {(0, 1): 20.0})

    def test_three_arg_weight_takes_max_abs(self):
        table = make_config([
            ("X1", "sum(X2,-5,10)"),
            ("X2", None),
        ])
        net = build_dependency_network(table)
        self.assertEqual(self._weighted_edges(net), {(0, 1): 10.0})

    def test_non_temporal_reference_weight_is_one(self):
        # Y aparece sin función temporal → peso = 1.
        table = make_config([
            ("X1", "X2 + 1"),
            ("X2", None),
        ])
        net = build_dependency_network(table)
        self.assertEqual(self._weighted_edges(net), {(0, 1): 1.0})

    def test_mixed_temporal_and_plain(self):
        # X2 aparece temporal (shift,-5) y no-temporal (+ X2): gana el 5.
        table = make_config([
            ("X1", "shift(X2,-5) + X2"),
            ("X2", None),
        ])
        net = build_dependency_network(table)
        self.assertEqual(self._weighted_edges(net), {(0, 1): 5.0})

    def test_multiple_calls_take_max_across_window(self):
        table = make_config([
            ("X1", "shift(X2,-3) + mean(X2,-7,7)"),
            ("X2", None),
        ])
        net = build_dependency_network(table)
        # max(3, 7) = 7
        self.assertEqual(self._weighted_edges(net), {(0, 1): 7.0})

    def test_per_dependency_independent_weights(self):
        # X1 := shift(X2,-3) + mean(X3,-10,10)
        # Edge X1→X2 = 3, Edge X1→X3 = 10
        table = make_config([
            ("X1", "shift(X2,-3) + mean(X3,-10,10)"),
            ("X2", None),
            ("X3", None),
        ])
        net = build_dependency_network(table)
        self.assertEqual(
            self._weighted_edges(net),
            {(0, 1): 3.0, (0, 2): 10.0},
        )

    def test_chain_of_weights(self):
        # X1 depende temporalmente de X2 (5), X2 de X3 (-2..2 → 2).
        table = make_config([
            ("X1", "shift(X2,-5)"),
            ("X2", "mean(X3,-2,2)"),
            ("X3", None),
        ])
        net = build_dependency_network(table)
        self.assertEqual(
            self._weighted_edges(net),
            {(0, 1): 5.0, (1, 2): 2.0},
        )


# --------------------------------------------------------------------- #
#  Network structure — directed graph + node "expression" meta
# --------------------------------------------------------------------- #
class TestNetworkStructure(unittest.TestCase):
    def test_edges_are_directed(self):
        table = make_config([
            ("X1", "shift(X2,-5)"),
            ("X2", None),
        ])
        net = build_dependency_network(table)
        # The Network Explorer's arrowheads only show up when the edge
        # set is directed.
        self.assertTrue(net.edges[0].directed)

    def test_node_metadata_includes_expression(self):
        table = make_config([
            ("X1", "shift(X2,-5)"),
            ("X2", None),
        ])
        net = build_dependency_network(table)
        # `expression` is the literal text for derived variables and
        # empty for originals.
        exprs = net.nodes.get_column("expression")
        names = net.nodes.get_column("var_name")
        by_name = dict(zip([str(n) for n in names], [str(e) for e in exprs]))
        self.assertEqual(by_name["X1"], "shift(X2,-5)")
        self.assertEqual(by_name["X2"], "")


# --------------------------------------------------------------------- #
#  Widget — Warning.no_derived
# --------------------------------------------------------------------- #
class TestNoDerivedWarning(unittest.TestCase):
    """Smoke-test the widget-level warning that fires when every input
    row is an original variable."""

    def _make_widget(self):
        # Lazy import: WidgetTest pulls a lot of Qt machinery.
        from Orange.widgets.tests.base import WidgetTest
        from timefeatures.widgets.owvardependencygraph import (
            owvardependencygraph,
        )

        class _T(WidgetTest):
            def runTest(self):
                pass

        harness = _T()
        harness.setUpClass()
        widget = harness.create_widget(owvardependencygraph)
        return harness, widget

    def test_warns_when_every_row_is_original(self):
        harness, widget = self._make_widget()
        try:
            table = make_config([("X1", None), ("X2", None)])
            harness.send_signal(widget.Inputs.data, table)
            self.assertTrue(widget.Warning.no_derived.is_shown())
        finally:
            harness.tearDownClass()

    def test_no_warning_when_any_derived(self):
        harness, widget = self._make_widget()
        try:
            table = make_config([("X1", "X2 + 1"), ("X2", None)])
            harness.send_signal(widget.Inputs.data, table)
            self.assertFalse(widget.Warning.no_derived.is_shown())
            self.assertFalse(widget.Warning.unparsable.is_shown())
        finally:
            harness.tearDownClass()

    def test_warns_about_unparsable_expressions(self):
        harness, widget = self._make_widget()
        try:
            table = make_config([("X1", "X2 +"), ("X2", None)])
            harness.send_signal(widget.Inputs.data, table)
            self.assertTrue(widget.Warning.unparsable.is_shown())
            self.assertIn("X1", str(widget.Warning.unparsable))
        finally:
            harness.tearDownClass()


if __name__ == "__main__":
    unittest.main()
