"""Variable Dependency Graph widget.

Toma una tabla de configuración (Variable / Expression) y construye un
grafo dirigido donde cada arista `A → B` significa "la expresión que
define A hace referencia a B".
"""
import math
import re

import numpy as np
from scipy import sparse as sp

import Orange
from Orange.data import DiscreteVariable, Domain, StringVariable, Table
from Orange.widgets import gui, settings
from Orange.widgets.widget import Input, Output, Msg, OWWidget
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin

from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout

from orangecontrib.network import Network
from orangecontrib.network.network.base import DirectedEdges


# Sanitización idéntica a la del Time Features Constructor para que los
# nombres encajen con los que aparecen escritos en las expresiones.
_NAME_FIX_RE = re.compile(r'[ \-]')

# Captura una llamada a función temporal (shift/sum/mean/count/min/max/sd),
# extrae la variable referenciada como primer argumento y el resto de
# argumentos numéricos. Aceptamos 1 o 2 enteros tras la coma para cubrir
# shift(var, n) y las demás (var, n, m).
_TIME_CALL_RE = re.compile(
    r'\b(?:shift|sum|mean|count|min|max|sd)\(\s*'
    r'(?P<var>[^,()\s]+)\s*'
    r'(?P<args>(?:,\s*[-+]?\d+\s*){1,2})\)'
)
_INT_RE = re.compile(r'[-+]?\d+')


def _sanitize_name(value):
    return _NAME_FIX_RE.sub('_', str(value))


def _expression_or_none(value):
    """Devuelve la expresión como str, o ``None`` si la celda está vacía.

    Acepta tanto ``Orange.data.Value`` (de DiscreteVariable) como floats
    sueltos. ``NaN``, cadena vacía y los marcadores ``"?"`` / ``"None"``
    se interpretan como ausencia de expresión (variable original).
    """
    try:
        if math.isnan(float(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value)
    if text in ("", "?", "NaN", "nan", "None"):
        return None
    return text


def _temporal_weights(expression):
    """Per variable referenced inside a temporal call of ``expression``,
    return the maximum ``|arg|`` aggregated across every call that
    mentions it. The bigger the value, the larger the time window the
    variable participates in.

    Returns
    -------
    dict[str, int]
        variable → max(|arg|). Empty if no temporal calls appear in
        the expression.
    """
    weights = {}
    for match in _TIME_CALL_RE.finditer(expression):
        var = match.group("var")
        args = [int(a) for a in _INT_RE.findall(match.group("args"))]
        if not args:
            continue
        max_abs = max(abs(a) for a in args)
        weights[var] = max(weights.get(var, 0), max_abs)
    return weights


def build_dependency_network(config_table):
    """Build a :class:`Network` from a configuration table.

    Edges carry a sparse-matrix weight equal to ``max(1, window)`` where
    ``window`` is the largest absolute argument among the temporal calls
    in the source expression that reference the destination variable; 1
    for purely non-temporal references. Network Explorer's
    *Scale edge widths to weights* picks this value up directly.

    Node metadata
    -------------
    ``var_name`` (str), ``var_type`` (``Derived`` / ``Original``) and
    ``expression`` (the literal expression text — empty for original
    variables) are exposed under ``network.nodes``.

    The graph is directed: an edge from ``A`` to ``B`` means *A depends
    on B*.

    Parameters
    ----------
    config_table : Orange.data.Table
        Table with at least the columns ``Variable`` (name) and
        ``Expression``.

    Returns
    -------
    Network
    """
    # 1. Locate the two columns by name (they may live in attributes or
    #    metas depending on whence the table came).
    all_vars = {v.name: v for v in
                list(config_table.domain.variables)
                + list(config_table.domain.metas)}
    var_col = all_vars["Variable"]
    expr_col = all_vars["Expression"]

    rows = [
        (_sanitize_name(row[var_col]), _expression_or_none(row[expr_col]))
        for row in config_table
    ]

    names = [name for name, _ in rows]
    # name → position lookup; O(1) instead of list.index().
    name_to_idx = {name: idx for idx, name in enumerate(names)}

    # Compile the reference detector once. ``\b`` avoids substring
    # matches (so ``X1`` doesn't trigger inside ``X10``). If the
    # configuration table is empty there's nothing to search.
    pattern = (
        re.compile(r'\b(' + '|'.join(re.escape(n) for n in names) + r')\b')
        if names else None
    )

    # 2. Walk every row and collect one record per discovered edge.
    row_edges, col_edges, edge_weights = [], [], []
    var_type = []
    expressions = []
    for src_idx, (_, expression) in enumerate(rows):
        if expression is None:
            # No expression → original variable, no outgoing edges.
            var_type.append(1)
            expressions.append("")
            continue
        var_type.append(0)
        expressions.append(expression)
        if pattern is None:
            continue

        temporal = _temporal_weights(expression)

        seen = set()
        for match in pattern.finditer(expression):
            dep = match.group(1)
            dep_idx = name_to_idx.get(dep)
            if dep_idx is None or dep in seen:
                continue
            seen.add(dep)

            window = temporal.get(dep, 0)
            # Keep the sparse weight ≥ 1 so consumers that filter out
            # exact zeros still see plain-reference edges.
            row_edges.append(src_idx)
            col_edges.append(dep_idx)
            edge_weights.append(max(1, window))

    n = len(names)
    sparse = sp.csr_matrix(
        (np.asarray(edge_weights, dtype=float), (row_edges, col_edges)),
        shape=(n, n),
    )

    # 3. Build per-node metadata.
    nodes_domain = Domain(
        [], [],
        metas=[
            StringVariable("var_name"),
            DiscreteVariable("var_type", values=["Derived", "Original"]),
            StringVariable("expression"),
        ],
    )
    node_metas = np.empty((n, 3), dtype=object)
    if n:
        node_metas[:, 0] = names
        node_metas[:, 1] = var_type
        node_metas[:, 2] = expressions
    nodes = Table.from_numpy(
        nodes_domain,
        np.zeros((n, 0)),
        np.zeros((n, 0)),
        node_metas,
    )

    # 4. Assemble the directed network. Previously we passed the sparse
    # matrix directly to ``Network``, which auto-wrapped it as
    # ``UndirectedEdges`` — a latent bug given that A→B is not the same
    # as B→A in a dependency graph.
    directed = DirectedEdges(sparse, name="depends_on")
    network = Network(range(n), directed, name="dependency")
    network.nodes = nodes
    return network


class owvardependencygraph(OWWidget, ConcurrentWidgetMixin):
    name = "Variable Dependency Graph"
    description = "Construct a graph with all the connections between the variables"
    icon = "icons/graphgenerator.svg"
    keywords = "variable dependency graph, function, graph, dependency, variable"
    priority = 2240

    # Reservado para futuros tipos de grafo. Por ahora sólo hay uno y la
    # generación es directa, así que el setting se conserva por compat.
    graph_type = settings.Setting(0)

    want_main_area = False
    resizing_enabled = False
    settings_version = 3

    class Error(OWWidget.Error):
        generation_error = Msg("{}")

    class Warning(OWWidget.Warning):
        no_derived = Msg(
            "Input has no derived variables; the dependency graph is empty."
        )

    class Inputs:
        data = Input("Variable Definitions", Orange.data.Table)

    class Outputs:
        network = Output("Network", Network)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ConcurrentWidgetMixin.__init__(self)
        self.controlArea.setMinimumWidth(360)

        self.data = None

        box = gui.vBox(self.controlArea, "Graph generator")

        toplayout = QHBoxLayout()
        toplayout.setContentsMargins(0, 0, 0, 0)
        box.layout().addLayout(toplayout)

        buttonlayout = QVBoxLayout(spacing=10)
        buttonlayout.setContentsMargins(0, 0, 0, 0)

        self.btn_generate = QPushButton(
            "Generate", toolTip="Generate dependency graph.",
            minimumWidth=10,
        )
        self.btn_generate.clicked.connect(self.generate)
        self.btn_generate.setEnabled(False)
        buttonlayout.addWidget(self.btn_generate)
        toplayout.addLayout(buttonlayout, 0)

    @Inputs.data
    def setData(self, data=None):
        self.data = data
        self.Error.generation_error.clear()

        if data is None:
            self.Outputs.network.send(None)
            self.btn_generate.setEnabled(False)
            return

        domain = data.domain
        all_names = {v.name for v in list(domain.variables) + list(domain.metas)}
        if "Variable" not in all_names or "Expression" not in all_names:
            self.Error.generation_error(
                "Input must be a configuration table with 'Variable' and "
                "'Expression' columns."
            )
            self.Outputs.network.send(None)
            self.btn_generate.setEnabled(False)
            return

        self.btn_generate.setEnabled(True)
        self.generate()

    def generate(self):
        self.Error.generation_error.clear()
        self.Warning.no_derived.clear()
        if self.data is None:
            self.Outputs.network.send(None)
            return

        try:
            network = build_dependency_network(self.data)
        except Exception as exc:  # pylint: disable=broad-except
            self.Error.generation_error(str(exc))
            self.Outputs.network.send(None)
            return

        # Heads-up: if every row of the configuration table is an
        # original variable, the resulting graph has no edges. The user
        # might have forgotten to fill in expressions.
        if network.number_of_nodes() > 0:
            var_type_col = network.nodes.get_column("var_type")
            if all(int(value) == 1 for value in var_type_col):
                self.Warning.no_derived()

        self.Outputs.network.send(network)
