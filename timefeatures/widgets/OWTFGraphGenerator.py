from Orange.widgets.widget import OWWidget
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin

class OWTFGraphGenerator(OWWidget, ConcurrentWidgetMixin):
    name = "T.F Graph Generator"
    description = "Construct a graph with all the conexions between the variables"
    icon = "icons/graphgenerator.svg"
    keywords = "time feature graph generator, function, graph"
    priority = 2240