import re
from functools import wraps

import Orange
from Orange.widgets.widget import OWWidget
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin

import numpy as np
from scipy import sparse as sp

from Orange.data import Table, Domain, StringVariable
from Orange.widgets import gui, widget, settings
from Orange.widgets.widget import Output, Msg
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout

from orangecontrib.network import Network
from orangewidget.utils.signals import Input


def from_row_col(f):
    @wraps(f)
    def wrapped(*args, data):
        row, col, *n, data = f(*args, data)

        variables = []

        for variable in data:
            var_arreglada = str(variable[0])
            var_arreglada = var_arreglada.replace(" ", "_").replace("-", "_")
            variables.append(var_arreglada)

        expresion_regular = r'\b(' + '|'.join(map(re.escape, variables)) + r')\b'

        print(expresion_regular)
        print("------------------------------------")
        relaciones = {}

        for datos in data:
            variable = str(datos[0])
            variable = variable.replace(" ", "_").replace("-", "_")
            if datos[1] is not None:
                relaciones[variable] = []
                for match in re.finditer(expresion_regular, str(datos[1])):
                    for group in match.groups():
                        if group:
                            relaciones[variable].append(group)

        print(relaciones)

        row_edges, col_edges = [], []
        for i, variable in enumerate(relaciones):
            for related_var in relaciones[variable]:
                if related_var in relaciones:
                    # Agrega la variable actual como nodo origen
                    row_edges.append(i)
                    # Agrega la variable relacionada (related_var) como nodo de destino
                    j = list(relaciones.keys()).index(related_var)
                    col_edges.append(j)

        nombres_variables = list(relaciones.keys())
        nombres_variables = np.array(nombres_variables).reshape(-1, 1)

        n = len(relaciones)
        edges = sp.csr_matrix((np.ones(len(row_edges)), (row_edges, col_edges)), shape=(n, n))
        print(edges)
        return Network(range(n), edges, name=f"{f.__name__}{args}"), nombres_variables

    return wrapped


@from_row_col
def grafo(n_nodos, data=None):
    return np.arange(len(data)), np.arange(len(data)), data


class OWTFGraphGenerator(OWWidget, ConcurrentWidgetMixin):
    name = "T.F Graph Generator"
    description = "Construct a graph with all the conexions between the variables"
    icon = "icons/graphgenerator.svg"
    keywords = "time feature graph generator, function, graph"
    priority = 2240

    GRAPH_TYPES = (
        grafo,)

    graph_type = settings.Setting(0)

    want_main_area = False

    settings_version = 3

    class Error(widget.OWWidget.Error):
        generation_error = Msg("{}")

    class Inputs:
        data = Input("Data", Orange.data.Table)

    class Outputs:
        network = Output("Network", Network)

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        ConcurrentWidgetMixin.__init__(self)
        self.data = None

        box = gui.vBox(self.controlArea, "Generador de grafos")

        toplayout = QHBoxLayout()
        toplayout.setContentsMargins(0, 0, 0, 0)
        box.layout().addLayout(toplayout)

        buttonlayout = QVBoxLayout(spacing=10)
        buttonlayout.setContentsMargins(0, 0, 0, 0)

        self.btn_generate = QPushButton(
            "Generar", toolTip="Generar grafo",
            minimumWidth=10
        )
        self.btn_generate.clicked.connect(self.generate)
        buttonlayout.addWidget(self.btn_generate)
        toplayout.addLayout(buttonlayout, 0)

    @Inputs.data
    def setData(self, data=None):

        self.data = data
        self.btn_generate.setEnabled(bool(self.data))

    def generate(self):
        if self.data is None:
            self.error("No data provided.")
            return

        n_nodos = [len(self.data.get_column(self.data.domain[0]))]
        func = self.GRAPH_TYPES[self.graph_type]

        self.Error.generation_error.clear()
        try:
            network, nombres_variables = func(n_nodos, data=self.data)
        except ValueError as exc:
            self.Error.generation_error(exc)
            network = None
        else:
            n = len(network.nodes)
            network.nodes = Table(Domain([], [], [StringVariable("nombre_var")]),
                                  np.zeros((n, 0)), np.zeros((n, 0)),
                                  np.arange(n).reshape((n, 1)))

            network.nodes[:, "nombre_var"] = nombres_variables

        self.Outputs.network.send(network)
