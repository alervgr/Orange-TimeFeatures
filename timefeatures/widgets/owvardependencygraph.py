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
from Orange.data import Table, Domain, StringVariable, DiscreteVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import Input, Output, Msg, OWWidget
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin

from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout

from orangecontrib.network import Network


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
    """Por cada variable referenciada en una función temporal de
    ``expression``, devuelve el máximo ``|arg|`` numérico entre TODAS las
    llamadas que la mencionan. La idea: cuanto mayor el peso, mayor la
    ventana temporal con la que esa variable interviene.

    Returns
    -------
    dict[str, int]
        variable → max(|arg|). Vacío si no hay llamadas temporales.
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
    """Construye un :class:`Network` a partir de una tabla de configuración.

    Las aristas llevan peso = máximo ``|arg|`` de las llamadas temporales
    en la expresión origen que referencian a la variable destino, o 1
    cuando la referencia es puramente no-temporal.

    Parameters
    ----------
    config_table : Orange.data.Table
        Tabla con dos columnas: ``Variable`` (nombre) y ``Expression``.

    Returns
    -------
    Network
        Con ``network.nodes`` poblado: metas ``var_name`` (str) y
        ``var_type`` (Derived / Original). El sparse matrix de aristas
        tiene los pesos numéricos como valores.
    """
    # 1. Una sola pasada para extraer (nombre saneado, expresión opcional).
    rows = [
        (_sanitize_name(row[0]), _expression_or_none(row[1]))
        for row in config_table
    ]

    names = [name for name, _ in rows]
    # Mapa nombre → posición → O(1) lookup en vez de list.index (O(n)).
    name_to_idx = {name: idx for idx, name in enumerate(names)}

    # Regex compilada una vez. \b evita matchear sub-strings (p.ej. "X1"
    # dentro de "X10"). Si no hay variables, no hace falta buscar nada.
    pattern = (
        re.compile(r'\b(' + '|'.join(re.escape(n) for n in names) + r')\b')
        if names else None
    )

    # 2. Detectar dependencias y calcular pesos.
    row_edges, col_edges, edge_weights = [], [], []
    var_type = []
    for src_idx, (_, expression) in enumerate(rows):
        if expression is None:
            # Sin expresión → variable original.
            var_type.append(1)
            continue
        var_type.append(0)
        if pattern is None:
            continue

        # Pesos por variable según las funciones temporales.
        temporal = _temporal_weights(expression)

        seen = set()
        for match in pattern.finditer(expression):
            dep = match.group(1)
            dep_idx = name_to_idx.get(dep)
            if dep_idx is None or dep in seen:
                continue
            seen.add(dep)
            row_edges.append(src_idx)
            col_edges.append(dep_idx)
            # Peso temporal si lo hay; si la referencia es no-temporal, 1.
            edge_weights.append(temporal.get(dep, 1))

    n = len(names)
    edges = sp.csr_matrix(
        (np.asarray(edge_weights, dtype=float),
         (row_edges, col_edges)),
        shape=(n, n),
    )

    # 3. Construir los nodos como meta-attributes.
    nodes_domain = Domain(
        [], [],
        metas=[
            StringVariable("var_name"),
            DiscreteVariable("var_type", values=["Derived", "Original"]),
        ],
    )
    metas = np.empty((n, 2), dtype=object)
    if n:
        metas[:, 0] = names
        metas[:, 1] = var_type
    nodes = Table.from_numpy(
        nodes_domain,
        np.zeros((n, 0)),
        np.zeros((n, 0)),
        metas,
    )

    network = Network(range(n), edges, name="dependency")
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
        if (len(domain) < 2
                or domain[0].name != "Variable"
                or domain[1].name != "Expression"):
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
        if self.data is None:
            self.Outputs.network.send(None)
            return

        try:
            network = build_dependency_network(self.data)
        except Exception as exc:  # pylint: disable=broad-except
            self.Error.generation_error(str(exc))
            network = None

        self.Outputs.network.send(network)
