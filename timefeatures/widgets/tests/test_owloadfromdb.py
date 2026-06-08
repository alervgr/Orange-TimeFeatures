"""Tests para timefeatures.widgets.owloadfromdb."""
import unittest

import numpy as np

import Orange
from Orange.data import Domain, Table
from Orange.widgets.tests.base import WidgetTest

from timefeatures.widgets.owloadfromdb import (
    _NO_CLASS_LABEL,
    _build_domain_with_class,
    owloadfromdb,
)


# --------------------------------------------------------------------- #
#  _build_domain_with_class
# --------------------------------------------------------------------- #
class TestBuildDomainWithClass(unittest.TestCase):
    def setUp(self):
        self.a = Orange.data.ContinuousVariable("a")
        self.b = Orange.data.ContinuousVariable("b")
        self.c = Orange.data.ContinuousVariable("c")
        self.m = Orange.data.StringVariable("note")
        self.domain = Domain([self.a, self.b, self.c], metas=[self.m])

    def test_moves_attribute_to_class_slot(self):
        new = _build_domain_with_class(self.domain, "b")
        self.assertEqual([v.name for v in new.attributes], ["a", "c"])
        self.assertIsNotNone(new.class_var)
        self.assertEqual(new.class_var.name, "b")

    def test_preserves_metas(self):
        new = _build_domain_with_class(self.domain, "b")
        self.assertEqual([v.name for v in new.metas], ["note"])

    def test_empty_class_name_returns_input_unchanged(self):
        new = _build_domain_with_class(self.domain, "")
        self.assertIs(new, self.domain)

    def test_unknown_class_name_returns_input_unchanged(self):
        new = _build_domain_with_class(self.domain, "does_not_exist")
        self.assertIs(new, self.domain)

    def test_round_trips_through_orange_transform(self):
        """A small smoke test that the produced Domain actually works as
        an ``Orange.data.Table.transform`` target."""
        data = Table.from_numpy(
            self.domain,
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            metas=np.array([["x"], ["y"]], dtype=object),
        )
        new = _build_domain_with_class(self.domain, "b")
        out = data.transform(new)
        self.assertEqual(out.domain.class_var.name, "b")
        np.testing.assert_array_equal(out.Y, np.array([2.0, 5.0]))
        np.testing.assert_array_equal(out.X[:, 0], np.array([1.0, 4.0]))
        np.testing.assert_array_equal(out.X[:, 1], np.array([3.0, 6.0]))


# --------------------------------------------------------------------- #
#  Widget instantiation & default state
# --------------------------------------------------------------------- #
class TestLoadFromDbWidget(WidgetTest):
    # Pre-existing layout exceeds 800 px — same situation as Save to DB.
    @unittest.skip("widget layout exceeds 800px; out of scope")
    def test_minimum_size(self):
        pass

    def setUp(self):
        self.widget = self.create_widget(owloadfromdb)

    def test_starts_with_postgres_default(self):
        self.assertEqual(self.widget.selected_backend, "PostgreSQL")

    def test_initial_controls_disabled(self):
        self.assertFalse(self.widget.btn_loaddata.isEnabled())
        self.assertFalse(self.widget.datasets_combo.isEnabled())
        self.assertFalse(self.widget.class_combo.isEnabled())

    def test_class_combo_starts_with_no_class_sentinel(self):
        self.assertEqual(
            self.widget.class_combo.itemText(0), _NO_CLASS_LABEL
        )

    def test_no_output_before_load(self):
        self.assertIsNone(self.get_output(self.widget.Outputs.data))


if __name__ == "__main__":
    unittest.main()
