import re
from functools import wraps

import Orange
import networkx.classes
import scipy as sp
from Orange.widgets.widget import OWWidget
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin
import string
from collections import defaultdict

import numpy as np

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QSpinBox

from Orange.data import Table, Domain, StringVariable
from Orange.widgets import gui, widget, settings
from Orange.widgets.widget import Output, Msg
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout
from networkx import Graph, DiGraph

from orangecontrib.network import Network
from orangecontrib.network.network.base import Edges
# __all__ is defined, pylint: disable=wildcard-import, unused-wildcard-import
from orangecontrib.network.network.generate import *
from orangewidget.utils.signals import Input


def from_row_col(f):
    @wraps(f)
    def wrapped(*args, data):
        row, col, *n, data = f(*args, data)

        expresion_regular = r'shift\(([^,]+),[-\d]+\)|sum\(([^,]+),[-\d]+,[-\d]+\)|mean\(([^,]+),[-\d]+,' \
                            r'[-\d]+\)|count\(([^,]+),[-\d]+,[-\d]+\)|min\(([^,]+),[-\d]+,[-\d]+\)|max\(([^,]+),' \
                            r'[-\d]+,[-\d]+\)|sd\(([^,]+),[-\d]+,[-\d]+\)'

        coincidencias = {}

        for datos in data:
            if datos[1] is not None:
                print(datos[1])
                for column_name_match_tempfunc in re.finditer(expresion_regular, str(datos[1])):
                    if column_name_match_tempfunc.group(1):
                        tabla = column_name_match_tempfunc.group(1)
                        coincidencias[str(datos[0])] = tabla
                    elif column_name_match_tempfunc.group(2):
                        tabla = column_name_match_tempfunc.group(2)
                        coincidencias[str(datos[0])] = tabla
                    elif column_name_match_tempfunc.group(3):
                        tabla = column_name_match_tempfunc.group(3)
                        coincidencias[str(datos[0])] = tabla
                    elif column_name_match_tempfunc.group(4):
                        tabla = column_name_match_tempfunc.group(4)
                        coincidencias[str(datos[0])] = tabla
                    elif column_name_match_tempfunc.group(5):
                        tabla = column_name_match_tempfunc.group(5)
                        coincidencias[str(datos[0])] = tabla
                    elif column_name_match_tempfunc.group(6):
                        tabla = column_name_match_tempfunc.group(6)
                        coincidencias[str(datos[0])] = tabla
                    elif column_name_match_tempfunc.group(7):
                        tabla = column_name_match_tempfunc.group(7)
                        coincidencias[str(datos[0])] = tabla

        print(coincidencias)

        n = n[0] if n else max(np.max(row), np.max(col)) + 1
        edges = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
        return Network(
            range(n), edges,
            name=f"{f.__name__}{args}".replace(",)", ")"))

    return wrapped


@from_row_col
def grafo(n, data):
    return np.arange(n - 1), np.arange(n - 1) + 1, n, data


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
            network = func(n_nodos, self.data)
        except ValueError as exc:
            self.Error.generation_error(exc)
            network = None
        else:
            n = len(network.nodes)
            network.nodes = Table(Domain([], [], [StringVariable("id")]),
                                  np.zeros((n, 0)), np.zeros((n, 0)),
                                  np.arange(n).reshape((n, 1)))
        self.Outputs.network.send(network)
