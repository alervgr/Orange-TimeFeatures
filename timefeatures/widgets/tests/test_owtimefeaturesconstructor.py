"""Tests para timefeatures.widgets.owtimefeaturesconstructor.

Ejecutar:
    python -m unittest timefeatures.widgets.tests.test_owtimefeaturesconstructor
o:
    pytest timefeatures/widgets/tests/test_owtimefeaturesconstructor.py
"""
import math
import unittest

import numpy as np

import Orange

from timefeatures.widgets.owtimefeaturesconstructor import (
    ContinuousDescriptor,
    FeatureFunc,
    bind_variable,
    construct_variables,
    count_function,
    make_variable,
    max_function,
    mean_function,
    min_function,
    modificar_expression,
    sanitized_name,
    sd_function,
    shift_function,
    sum_function,
    topological_sort_descriptors,
)


# --------------------------------------------------------------------- #
#  modificar_expression
# --------------------------------------------------------------------- #
class TestModificarExpression(unittest.TestCase):
    """La numeración de cada llamada temporal debe ser independiente por
    función (shift, sum, mean, ...) y por orden de aparición."""

    def test_passthrough_when_no_time_function(self):
        self.assertEqual(modificar_expression("x + y * 2"), "x + y * 2")

    def test_single_shift(self):
        self.assertEqual(modificar_expression("shift(x,-5)"), "shift0(x,-5)")

    def test_two_shifts_get_independent_indices(self):
        self.assertEqual(
            modificar_expression("shift(x,-5)+shift(y,3)"),
            "shift0(x,-5)+shift1(y,3)",
        )

    def test_two_identical_shifts_still_numbered(self):
        # Las llamadas idénticas también se numeran (importante para enlazar
        # cada una con su columna en FeatureFunc.__call_table).
        self.assertEqual(
            modificar_expression("shift(x,-5)+shift(x,-5)"),
            "shift0(x,-5)+shift1(x,-5)",
        )

    def test_three_arg_functions(self):
        self.assertEqual(modificar_expression("sum(x,-5,5)"), "sum0(x,-5,5)")
        self.assertEqual(modificar_expression("mean(x,0,10)"), "mean0(x,0,10)")
        self.assertEqual(modificar_expression("count(x,1,2)"), "count0(x,1,2)")
        self.assertEqual(modificar_expression("min(x,1,2)"), "min0(x,1,2)")
        self.assertEqual(modificar_expression("max(x,1,2)"), "max0(x,1,2)")
        self.assertEqual(modificar_expression("sd(x,1,2)"), "sd0(x,1,2)")

    def test_each_function_has_its_own_counter(self):
        result = modificar_expression(
            "shift(x,1) + sum(x,1,2) + mean(x,1,2) + count(x,1,2)"
            " + min(x,1,2) + max(x,1,2) + sd(x,1,2)"
        )
        for token in (
            "shift0(x,1)", "sum0(x,1,2)", "mean0(x,1,2)", "count0(x,1,2)",
            "min0(x,1,2)", "max0(x,1,2)", "sd0(x,1,2)",
        ):
            self.assertIn(token, result)

    def test_signed_arguments_preserved(self):
        self.assertEqual(modificar_expression("shift(x,+5)"), "shift0(x,+5)")
        self.assertEqual(modificar_expression("sum(x,-5,+5)"), "sum0(x,-5,+5)")


# --------------------------------------------------------------------- #
#  Funciones de ventana (puras)
# --------------------------------------------------------------------- #
class TestShiftFunction(unittest.TestCase):
    def test_zero_shift_returns_var(self):
        self.assertEqual(shift_function(42, 0, tabla=[1, 2, 3], cont=1), 42)

    def test_positive_shift(self):
        self.assertEqual(
            shift_function(20, 1, tabla=[10, 20, 30, 40], cont=1), 30
        )

    def test_negative_shift(self):
        self.assertEqual(
            shift_function(20, -1, tabla=[10, 20, 30, 40], cont=1), 10
        )

    def test_out_of_bounds_below_returns_none(self):
        self.assertIsNone(shift_function(10, -1, tabla=[10, 20, 30], cont=0))

    def test_out_of_bounds_above_returns_none(self):
        self.assertIsNone(shift_function(30, 1, tabla=[10, 20, 30], cont=2))


class TestSumFunction(unittest.TestCase):
    def test_z_equals_x_returns_var(self):
        self.assertEqual(sum_function(5, 0, 0, tabla=[1, 2, 3], cont=1), 5)

    def test_sums_window(self):
        # range(0, 2) desde cont=1 → tabla[1]+tabla[2] = 2+3 = 5
        self.assertEqual(
            sum_function(2.0, 0, 1, tabla=[1.0, 2.0, 3.0, 4.0, 5.0], cont=1),
            5.0,
        )

    def test_skips_nan_values(self):
        # range(0, 3) desde cont=0 → tabla[0]+tabla[2] = 1+3 (NaN se ignora)
        self.assertEqual(
            sum_function(1.0, 0, 2, tabla=[1.0, float("nan"), 3.0], cont=0),
            4.0,
        )

    def test_all_nan_returns_none(self):
        self.assertIsNone(
            sum_function(
                float("nan"), 0, 1,
                tabla=[float("nan"), float("nan")], cont=0,
            )
        )


class TestMeanFunction(unittest.TestCase):
    def test_basic_mean(self):
        # range(0, 2) desde cont=0 → (tabla[0]+tabla[1])/2 = (2+4)/2 = 3
        self.assertEqual(
            mean_function(2.0, 0, 1, tabla=[2.0, 4.0, 6.0], cont=0),
            3.0,
        )

    def test_skips_nan(self):
        self.assertEqual(
            mean_function(
                1.0, 0, 2, tabla=[1.0, float("nan"), 3.0], cont=0
            ),
            2.0,
        )

    def test_all_nan_returns_none(self):
        self.assertIsNone(
            mean_function(
                float("nan"), 0, 1,
                tabla=[float("nan"), float("nan")], cont=0,
            )
        )


class TestCountFunction(unittest.TestCase):
    def test_z_equals_x_and_not_nan_returns_one(self):
        self.assertEqual(
            count_function(5.0, 0, 0, tabla=[5.0], cont=0), 1
        )

    def test_counts_non_nan(self):
        self.assertEqual(
            count_function(
                1.0, 0, 2, tabla=[1.0, float("nan"), 3.0], cont=0
            ),
            2,
        )


class TestMinMaxFunctions(unittest.TestCase):
    def test_min_basic(self):
        self.assertEqual(
            min_function(5.0, 0, 2, tabla=[5.0, 1.0, 3.0], cont=0),
            1.0,
        )

    def test_max_basic(self):
        self.assertEqual(
            max_function(5.0, 0, 2, tabla=[5.0, 1.0, 3.0], cont=0),
            5.0,
        )

    def test_min_skips_nan(self):
        self.assertEqual(
            min_function(
                5.0, 0, 2, tabla=[5.0, float("nan"), 3.0], cont=0
            ),
            3.0,
        )


class TestSDFunction(unittest.TestCase):
    def test_basic(self):
        # std de [2, 4, 6] = sqrt(8/3) ≈ 1.6329
        result = sd_function(2.0, 0, 2, tabla=[2.0, 4.0, 6.0], cont=0)
        self.assertAlmostEqual(result, math.sqrt(8 / 3), places=6)

    def test_raises_when_args_missing(self):
        with self.assertRaises(ValueError):
            sd_function(None, 0, 1, tabla=[1.0], cont=0)


# --------------------------------------------------------------------- #
#  sanitized_name
# --------------------------------------------------------------------- #
class TestSanitizedName(unittest.TestCase):
    def test_replaces_spaces(self):
        self.assertEqual(sanitized_name("hello world"), "hello_world")

    def test_replaces_special_chars(self):
        self.assertEqual(sanitized_name("a-b.c"), "a_b_c")

    def test_prepends_underscore_for_digit_start(self):
        self.assertEqual(sanitized_name("1abc"), "_1abc")

    def test_valid_name_unchanged(self):
        self.assertEqual(sanitized_name("variable_1"), "variable_1")


# --------------------------------------------------------------------- #
#  FeatureFunc — regresión del bug de chunking (5000 filas)
# --------------------------------------------------------------------- #
class TestFeatureFuncChunking(unittest.TestCase):
    """Orange.data.table._FromTableConversion trocea las tablas en bloques
    de max_rows_at_once=5000 al evaluar compute_value. Las funciones de
    ventana temporal (shift/sum/mean/...) deben mantener su contexto a
    través de las fronteras de los chunks."""

    @staticmethod
    def _transform(expression, n_rows, var_name="x"):
        domain = Orange.data.Domain([Orange.data.ContinuousVariable(var_name)])
        data = Orange.data.Table.from_numpy(
            domain, np.arange(n_rows, dtype=float).reshape(-1, 1)
        )
        desc = ContinuousDescriptor(
            name="y", expression=expression, meta=False,
            number_of_decimals=3,
        )
        desc, func = bind_variable(
            desc, list(data.domain.variables), data, use_values=False
        )
        var = make_variable(desc, func)
        new_domain = Orange.data.Domain(
            list(data.domain.attributes) + [var]
        )
        return data.transform(new_domain).get_column("y")

    def test_shift_crosses_chunk_boundary(self):
        n = 12_000  # > 2 chunks de 5000
        col = self._transform("shift(x,-20)", n)
        # Las primeras 20 filas no tienen "lag" → NaN
        self.assertEqual(int(np.sum(np.isnan(col[:20]))), 20)
        # Para i >= 20 → col[i] == x[i - 20] == i - 20
        np.testing.assert_array_equal(
            col[20:], np.arange(n - 20, dtype=float)
        )

    def test_no_chunking_for_small_dataset(self):
        n = 1_000  # cabe en un único chunk
        col = self._transform("shift(x,-5)", n)
        np.testing.assert_array_equal(
            col[5:], np.arange(n - 5, dtype=float)
        )

    def test_non_temporal_expression_unaffected(self):
        n = 12_000
        col = self._transform("x * 2", n)
        np.testing.assert_array_equal(col, np.arange(n, dtype=float) * 2)

    def test_set_source_invalidates_cache(self):
        """set_source debe limpiar el resultado cacheado y el mapa id→idx."""
        domain = Orange.data.Domain([Orange.data.ContinuousVariable("x")])
        data = Orange.data.Table.from_numpy(
            domain, np.array([[1.0], [2.0]])
        )
        desc = ContinuousDescriptor(
            name="y", expression="shift(x,-1)", meta=False,
            number_of_decimals=3,
        )
        _, func = bind_variable(
            desc, list(data.domain.variables), data, use_values=False
        )

        # Simulamos un caché ya poblado por una pasada por chunk previa
        func._full_result = [1.0, 2.0]
        func._id_to_idx = {0: 0, 1: 1}

        # Re-registrar origen debe invalidar
        func.set_source(data)
        self.assertIsNone(func._full_result)
        self.assertIsNone(func._id_to_idx)

    def test_cache_populated_via_transform(self):
        """La transformación de una tabla grande dispara chunking; tras
        completarse, el FeatureFunc debe tener su caché lleno."""
        n = 12_000
        domain = Orange.data.Domain([Orange.data.ContinuousVariable("x")])
        data = Orange.data.Table.from_numpy(
            domain, np.arange(n, dtype=float).reshape(-1, 1)
        )
        desc = ContinuousDescriptor(
            name="y", expression="shift(x,-5)", meta=False,
            number_of_decimals=3,
        )
        desc, func = bind_variable(
            desc, list(data.domain.variables), data, use_values=False
        )
        var = make_variable(desc, func)
        new_domain = Orange.data.Domain(
            list(data.domain.attributes) + [var]
        )
        data.transform(new_domain)
        # Tras la transformación con chunking activo, el caché está lleno
        self.assertIsNotNone(func._full_result)
        self.assertEqual(len(func._full_result), n)


# --------------------------------------------------------------------- #
#  FeatureFunc — endurecimiento de eval
# --------------------------------------------------------------------- #
class TestFeatureFuncEvalSafety(unittest.TestCase):
    """eval debe correr con __builtins__ vacío y solo la whitelist curada."""

    @staticmethod
    def _make_func(expression, rows=2):
        domain = Orange.data.Domain([Orange.data.ContinuousVariable("x")])
        data = Orange.data.Table.from_numpy(
            domain, np.arange(rows, dtype=float).reshape(-1, 1) + 1
        )
        desc = ContinuousDescriptor(
            name="y", expression=expression, meta=False,
            number_of_decimals=3,
        )
        _, func = bind_variable(
            desc, list(data.domain.variables), data, use_values=False
        )
        return func, data

    def test_blocks_import_builtin(self):
        # __import__ no está en la whitelist → NameError al evaluar
        func, data = self._make_func("__import__('os') if x else 0")
        with self.assertRaises(NameError):
            func(data)

    def test_blocks_open_builtin(self):
        func, data = self._make_func("open('/etc/passwd') if x else 0")
        with self.assertRaises(NameError):
            func(data)

    def test_allows_math_functions(self):
        func, data = self._make_func("sqrt(x)")
        result = func(data)
        # x = [1, 2] → sqrt(x) = [1.0, sqrt(2)]
        self.assertAlmostEqual(result[0], 1.0)
        self.assertAlmostEqual(result[1], math.sqrt(2))

    def test_allows_whitelisted_builtins(self):
        # abs está en __ALLOWED y debe seguir funcionando
        func, data = self._make_func("abs(x - 5)")
        result = func(data)
        self.assertEqual(result[0], 4)  # |1 - 5| = 4
        self.assertEqual(result[1], 3)  # |2 - 5| = 3


# --------------------------------------------------------------------- #
#  Chained descriptors — X2 referencing X1
# --------------------------------------------------------------------- #
class TestTopologicalSortDescriptors(unittest.TestCase):
    @staticmethod
    def _desc(name, expression):
        return ContinuousDescriptor(
            name=name, expression=expression, meta=False,
            number_of_decimals=3,
        )

    def _names(self, sorted_descs):
        return [d.name for d in sorted_descs]

    def test_no_dependencies_preserves_input_order(self):
        d1 = self._desc("X1", "a + 1")
        d2 = self._desc("X2", "b * 2")
        self.assertEqual(
            self._names(topological_sort_descriptors([d1, d2])),
            ["X1", "X2"],
        )

    def test_simple_chain_input_order(self):
        d1 = self._desc("X1", "a + 1")
        d2 = self._desc("X2", "X1 * 2")
        self.assertEqual(
            self._names(topological_sort_descriptors([d1, d2])),
            ["X1", "X2"],
        )

    def test_simple_chain_reverse_input_order(self):
        # User added X2 first by mistake, then X1. Topo sort puts the
        # dependency first.
        d1 = self._desc("X1", "a + 1")
        d2 = self._desc("X2", "X1 * 2")
        self.assertEqual(
            self._names(topological_sort_descriptors([d2, d1])),
            ["X1", "X2"],
        )

    def test_transitive_chain(self):
        d1 = self._desc("X1", "a + 1")
        d2 = self._desc("X2", "X1 * 2")
        d3 = self._desc("X3", "X2 + 1")
        self.assertEqual(
            self._names(topological_sort_descriptors([d3, d2, d1])),
            ["X1", "X2", "X3"],
        )

    def test_cycle_raises_value_error(self):
        d1 = self._desc("X1", "X2 + 1")
        d2 = self._desc("X2", "X1 + 1")
        with self.assertRaises(ValueError) as cm:
            topological_sort_descriptors([d1, d2])
        self.assertIn("Circular dependency", str(cm.exception))
        self.assertIn("X1", str(cm.exception))
        self.assertIn("X2", str(cm.exception))

    def test_invalid_expression_is_treated_as_no_dependency(self):
        # Syntactically invalid: ast.parse raises, so no deps recorded.
        d_bad = self._desc("X1", "this is not valid")
        d_good = self._desc("X2", "X1 + 1")
        self.assertEqual(
            self._names(topological_sort_descriptors([d_bad, d_good])),
            ["X1", "X2"],
        )

    def test_empty_expression_is_treated_as_no_dependency(self):
        d_empty = self._desc("X1", "")
        d_good = self._desc("X2", "X1 + 1")
        self.assertEqual(
            self._names(topological_sort_descriptors([d_empty, d_good])),
            ["X1", "X2"],
        )

    def test_self_reference_is_not_a_cycle(self):
        # ``X1 := shift(X1, -1)`` references the source-column X1 (if it
        # exists). We don't treat that as a self-dependency.
        d = self._desc("X1", "shift(X1, -1)")
        self.assertEqual(
            self._names(topological_sort_descriptors([d])),
            ["X1"],
        )

    def test_empty_input(self):
        self.assertEqual(topological_sort_descriptors([]), [])


class TestConstructVariablesChained(unittest.TestCase):
    """End-to-end: construct_variables now cascades, so X2 can read X1
    even though X1 is itself a derived variable."""

    @staticmethod
    def _build_data(rows):
        domain = Orange.data.Domain([Orange.data.ContinuousVariable("a")])
        return Orange.data.Table.from_numpy(
            domain, np.array(rows, dtype=float).reshape(-1, 1)
        )

    def test_chained_simple(self):
        # X1 = a + 1; X2 = X1 * 2.
        data = self._build_data([1.0, 2.0, 3.0])
        d1 = ContinuousDescriptor(
            name="X1", expression="a + 1", meta=False,
            number_of_decimals=3,
        )
        d2 = ContinuousDescriptor(
            name="X2", expression="X1 * 2", meta=False,
            number_of_decimals=3,
        )

        # Input order: X2 BEFORE X1 → cascade must topo-sort.
        variables, metas, transformed = construct_variables(
            [d2, d1], data, use_values=False
        )

        # Both variables produced (and X1 appears before X2 in the order).
        self.assertEqual([v.name for v in variables], ["X1", "X2"])
        self.assertEqual(metas, ())

        # Values cascade correctly.
        np.testing.assert_array_equal(
            np.asarray(transformed.get_column("X1")),
            np.array([2.0, 3.0, 4.0]),
        )
        np.testing.assert_array_equal(
            np.asarray(transformed.get_column("X2")),
            np.array([4.0, 6.0, 8.0]),
        )

    def test_chained_with_time_function_across_chunk_boundary(self):
        # X1 = shift(a, -1) and X2 = X1 + 1 over a 12 000-row table
        # forces the 5 000-row chunking path inside Orange's transform.
        # Both the chaining (X2 reads derived X1) and the chunking fix
        # (FeatureFunc._full_source caching) must hold.
        n = 12_000
        data = self._build_data(np.arange(n).tolist())

        # Note: the time-function regex inside FeatureFunc does NOT allow
        # whitespace between the args — that matches what the editor's
        # combobox inserts, so the user never sees the difference.
        d1 = ContinuousDescriptor(
            name="X1", expression="shift(a,-1)", meta=False,
            number_of_decimals=3,
        )
        d2 = ContinuousDescriptor(
            name="X2", expression="X1 + 1", meta=False,
            number_of_decimals=3,
        )

        variables, _, transformed = construct_variables(
            [d1, d2], data, use_values=False
        )

        self.assertEqual([v.name for v in variables], ["X1", "X2"])

        # X1[i] = a[i-1] = i-1 for i >= 1, NaN for i = 0.
        x1 = np.asarray(transformed.get_column("X1"), dtype=float)
        self.assertTrue(np.isnan(x1[0]))
        np.testing.assert_array_equal(
            x1[1:], np.arange(n - 1, dtype=float),
        )

        # X2[i] = X1[i] + 1 = i for i >= 1, NaN for i = 0.
        x2 = np.asarray(transformed.get_column("X2"), dtype=float)
        self.assertTrue(np.isnan(x2[0]))
        np.testing.assert_array_equal(
            x2[1:], np.arange(1, n, dtype=float),
        )

    def test_empty_descriptors_returns_input_unchanged(self):
        data = self._build_data([1.0, 2.0])
        variables, metas, transformed = construct_variables(
            [], data, use_values=False
        )
        self.assertEqual(variables, ())
        self.assertEqual(metas, ())
        self.assertIs(transformed, data)


if __name__ == "__main__":
    unittest.main()
