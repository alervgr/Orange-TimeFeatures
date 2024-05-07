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

'''

CALCULO DE PONDERACIONES PARA LOS PESOS DE LAS ARISTAS.

def calculate_weight(expression):
    # Encontrar todas las coincidencias con las funciones temporales y almacenarlas
    matches_shift = list(re.finditer(r'shift\(([^,]+),([-+]?\d+)\)', expression))
    matches_sum = list(re.finditer(r'sum\(([^,]+),([-+]?\d+),([-+]?\d+)\)', expression))
    matches_mean = list(re.finditer(r'mean\(([^,]+),([-+]?\d+),([-+]?\d+)\)', expression))
    matches_count = list(re.finditer(r'count\(([^,]+),([-+]?\d+),([-+]?\d+)\)', expression))
    matches_min = list(re.finditer(r'min\(([^,]+),([-+]?\d+),([-+]?\d+)\)', expression))
    matches_max = list(re.finditer(r'max\(([^,]+),([-+]?\d+),([-+]?\d+)\)', expression))
    matches_sd = list(re.finditer(r'sd\(([^,]+),([-+]?\d+),([-+]?\d+)\)', expression))

    valores = {}

    # Iterar sobre todas las coincidencias de shift y cambiar la expresión
    for match in matches_shift:
        variable_name = match.group(1)
        shift_value = match.group(2)

        valores[variable_name] = -abs(int(shift_value))

    for match in matches_sum:
        variable_name = match.group(1)
        sum_value1 = match.group(2)
        sum_value2 = match.group(3)

        valores[variable_name] = -max(abs(int(sum_value1)), abs(int(sum_value2)))

    for match in matches_mean:
        variable_name = match.group(1)
        mean_value1 = match.group(2)
        mean_value2 = match.group(3)

        valores[variable_name] = -max(abs(int(mean_value1)), abs(int(mean_value2)))

    for match in matches_count:
        variable_name = match.group(1)
        count_value1 = match.group(2)
        count_value2 = match.group(3)

        valores[variable_name] = -max(abs(int(count_value1)), abs(int(count_value2)))

    for match in matches_min:
        variable_name = match.group(1)
        min_value1 = match.group(2)
        min_value2 = match.group(3)

        valores[variable_name] = -max(abs(int(min_value1)), abs(int(min_value2)))

    for match in matches_max:
        variable_name = match.group(1)
        max_value1 = match.group(2)
        max_value2 = match.group(3)

        valores[variable_name] = -max(abs(int(max_value1)), abs(int(max_value2)))

    for match in matches_sd:
        variable_name = match.group(1)
        sd_value1 = match.group(2)
        sd_value2 = match.group(3)

        valores[variable_name] = -max(abs(int(sd_value1)), abs(int(sd_value2)))

    return valores.values()'''

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
        # edge_weights = []

        for datos in data:
            variable = str(datos[0])
            variable = variable.replace(" ", "_").replace("-", "_")
            if not math.isnan(datos[1]) and str(datos[1]) != "NaN":
                tipo_var.append(0)
                # edge_weights_exp = calculate_weight(str(datos[1]))
                # for weight in edge_weights_exp:
                    # edge_weights.append(int(weight))
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

        # np_edge_weights = np.array(edge_weights)

        n = len(relaciones)
        # edges = sp.csr_matrix((np_edge_weights, (row_edges, col_edges)), shape=(n, n))
        edges = sp.csr_matrix((np.ones(len(row_edges)), (row_edges, col_edges)), shape=(n, n))
        return Network(range(n), edges, name=f"{f.__name__}{args}"), nombres_variables, tipo_var_reshaped

    return wrapped


@from_row_col
def grafo(data=None):
    return data


class owvardependencygraph(OWWidget, ConcurrentWidgetMixin):
    name = "Variable Dependency Graph"
    description = "Construct a graph with all the conexions between the variables"
    icon = "icons/graphgenerator.svg"
    keywords = "variable dependency graph, function, graph, dependency, variable"
    priority = 2240

    GRAPH_TYPES = (
        grafo,)

    graph_type = settings.Setting(0)

    want_main_area = False

    resizing_enabled = False

    settings_version = 3

    class Error(widget.OWWidget.Error):
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
            minimumWidth=10
        )
        self.btn_generate.clicked.connect(self.generate)
        self.btn_generate.setEnabled(False)
        buttonlayout.addWidget(self.btn_generate)
        toplayout.addLayout(buttonlayout, 0)

    @Inputs.data
    def setData(self, data=None):

        self.data = data

        if self.data is not None:
            if len(self.data.domain) >= 1 and (self.data.domain[0].name != "Variable" or self.data.domain[1].name != "Expression"):
                self.Error.generation_error("You need a configuration table (Variable-Expression).")
                self.Outputs.network.send(None)
            else:
                self.generate()
                self.btn_generate.setEnabled(True)
        else:
            self.Error.clear()
            self.Outputs.network.send(None)
            self.btn_generate.setEnabled(False)

    def generate(self):

        func = self.GRAPH_TYPES[self.graph_type]

        self.Error.generation_error.clear()
        try:
            network, nombres_variables, tipo_var_reshaped = func(data=self.data)
        except ValueError as exc:
            self.Error.generation_error(exc)
            network = None
        else:
            n = len(network.nodes)
            network.nodes = Table(Domain([], [], [StringVariable("var_name"), DiscreteVariable("var_type", values=["Derived", "Original"])]),
                                  np.zeros((n, 0)), np.zeros((n, 0)),
                                  np.arange(2*n).reshape((n, 2)))

            network.nodes[:, "var_name"] = nombres_variables
            network.nodes[:, "var_type"] = tipo_var_reshaped

        self.Outputs.network.send(network)
