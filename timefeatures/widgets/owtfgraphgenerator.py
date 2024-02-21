import math
import re
from functools import wraps

import Orange
from Orange.widgets.widget import OWWidget
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin

import numpy as np
from scipy import sparse as sp

from Orange.data import Table, Domain, StringVariable, DiscreteVariable
from Orange.widgets import gui, widget, settings
from Orange.widgets.widget import Output, Msg
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout

from orangecontrib.network import Network
from orangewidget.utils.signals import Input


def from_row_col(f):
    @wraps(f)
    def wrapped(*args, data):
        data = f(*args, data)

        variables = []
        tipo_var = []

        for variable in data:
            var_arreglada = str(variable[0])
            var_arreglada = var_arreglada.replace(" ", "_").replace("-", "_")
            variables.append(var_arreglada)

        expresion_regular = r'\b(' + '|'.join(map(re.escape, variables)) + r')\b'

        relaciones = {}

        for datos in data:
            variable = str(datos[0])
            variable = variable.replace(" ", "_").replace("-", "_")
            if not math.isnan(datos[1]):
                tipo_var.append(0)
            else:
                tipo_var.append(1)
            relaciones[variable] = []
            for match in re.finditer(expresion_regular, str(datos[1])):
                for group in match.groups():
                    if group and group not in relaciones[variable]:
                        relaciones[variable].append(group)

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

        tipo_var_reshaped = np.array(tipo_var).reshape(-1, 1)

        n = len(relaciones)
        edges = sp.csr_matrix((np.ones(len(row_edges)), (row_edges, col_edges)), shape=(n, n))
        return Network(range(n), edges, name=f"{f.__name__}{args}"), nombres_variables, tipo_var_reshaped

    return wrapped


@from_row_col
def grafo(data=None):
    return data


class owtfgraphgenerator(OWWidget, ConcurrentWidgetMixin):
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

        box = gui.vBox(self.controlArea, "Graph generator")

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
        self.generate()

    def generate(self):
        if self.data is None:
            self.error("No data provided.")
            return

        func = self.GRAPH_TYPES[self.graph_type]

        self.Error.generation_error.clear()
        try:
            network, nombres_variables, tipo_var_reshaped = func(data=self.data)
        except ValueError as exc:
            self.Error.generation_error(exc)
            network = None
        else:
            n = len(network.nodes)
            network.nodes = Table(Domain([], [], [StringVariable("nombre_var"), DiscreteVariable("tipo_var", values=["Derived", "Original"])]),
                                  np.zeros((n, 0)), np.zeros((n, 0)),
                                  np.arange(2*n).reshape((n, 2)))

            network.nodes[:, "nombre_var"] = nombres_variables
            network.nodes[:, "tipo_var"] = tipo_var_reshaped

        self.Outputs.network.send(network)
