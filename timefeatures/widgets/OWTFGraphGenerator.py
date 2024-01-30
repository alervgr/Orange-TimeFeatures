import Orange
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

from orangecontrib.network import Network
# __all__ is defined, pylint: disable=wildcard-import, unused-wildcard-import
from orangecontrib.network.network.generate import *
from orangewidget.utils.signals import Input


class OWTFGraphGenerator(OWWidget, ConcurrentWidgetMixin):
    name = "T.F Graph Generator"
    description = "Construct a graph with all the conexions between the variables"
    icon = "icons/graphgenerator.svg"
    keywords = "time feature graph generator, function, graph"
    priority = 2240

    class Inputs:
        data = Input("Data", Orange.data.Table)

    class Outputs:
        network = Output("Network", Network)

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)
        ConcurrentWidgetMixin.__init__(self)
        self.data = None

    @Inputs.data
    def setData(self, data=None):

        print("a")
        self.data = data
        print(self.data)



