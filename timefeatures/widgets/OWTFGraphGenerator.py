import Orange
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

from orangecontrib.network import Network
from orangecontrib.network.network.base import Edges
# __all__ is defined, pylint: disable=wildcard-import, unused-wildcard-import
from orangecontrib.network.network.generate import *
from orangewidget.utils.signals import Input


class OWTFGraphGenerator(OWWidget, ConcurrentWidgetMixin):
    name = "T.F Graph Generator"
    description = "Construct a graph with all the conexions between the variables"
    icon = "icons/graphgenerator.svg"
    keywords = "time feature graph generator, function, graph"
    priority = 2240

    GRAPH_TYPES = (
        path,)

    graph_type = settings.Setting(0)

    want_main_area = False

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

        func = self.GRAPH_TYPES[self.graph_type]
        args = [len(self.data.get_column(self.data.domain[0]))]
        self.Error.generation_error.clear()
        try:
            network = func(*args)
        except ValueError as exc:
            self.Error.generation_error(exc)
            network = None
        else:
            n = len(network.nodes)
            network.nodes = Table(Domain([], [], [StringVariable("id")]),
                                  np.zeros((n, 0)), np.zeros((n, 0)),
                                  np.arange(n).reshape((n, 1)))
        self.Outputs.network.send(network)
