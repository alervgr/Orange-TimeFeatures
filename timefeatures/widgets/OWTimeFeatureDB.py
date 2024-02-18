import Orange
from AnyQt.QtWidgets import QComboBox, QTextEdit, QMessageBox, QApplication
from AnyQt.QtGui import QCursor
from AnyQt.QtCore import Qt
from datetime import datetime

from Orange.data import Table
from Orange.data.sql.backend import Backend
from Orange.data.sql.backend.base import BackendError
from Orange.data.sql.table import SqlTable, LARGE_TABLE, AUTO_DL_LIMIT
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.itemmodels import PyListModel
from Orange.widgets.utils.owbasesql import OWBaseSql
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Msg
from PyQt5.QtWidgets import QGridLayout, QLineEdit, QPushButton, QSizePolicy, QLabel
from orangewidget.utils.signals import Input

MAX_DL_LIMIT = 1000000


def is_postgres(backend):
    return getattr(backend, 'display_name', '') == "PostgreSQL"


class TableModel(PyListModel):
    def data(self, index, role=Qt.DisplayRole):
        row = index.row()
        if role == Qt.DisplayRole:
            return str(self[row])
        return super().data(index, role)


class BackendModel(PyListModel):
    def data(self, index, role=Qt.DisplayRole):
        row = index.row()
        if role == Qt.DisplayRole:
            return self[row].display_name
        return super().data(index, role)


class OWTimeFeatureDB(OWBaseSql):
    name = "Time Feature DB"
    description = "Save data in to a DB."
    icon = "icons/savedatadb.svg"
    priority = 2240
    keywords = "sql table, save, data"

    class Inputs:
        data = Input("Data", Orange.data.Table)

    class Outputs:
        pass

    settings_version = 2

    buttons_area_orientation = None

    selected_backend = Setting(None)
    table = Setting(None)
    sql = Setting("")
    guess_values = Setting(True)
    download = Setting(False)

    materialize = Setting(False)
    materialize_table_name = Setting("")

    class Information(OWBaseSql.Information):
        data_sampled = Msg("Data description was generated from a sample.")

    class Warning(OWBaseSql.Warning):
        missing_extension = Msg("Database is missing extensions: {}")

    class Error(OWBaseSql.Error):
        no_backends = Msg("Please install a backend to use this widget.")

    def __init__(self):
        # Lint
        self.backends = None
        self.backendcombo = None
        self.data = None
        self.rows = 0
        self.cols = 0
        self.target = None
        super().__init__()

    def update_labels(self):
        self.target_label.setText("Class: " + str(self.target))
        self.rows_label.setText("Rows: " + str(self.rows))
        self.cols_label.setText("Columns: " + str(self.cols))

    @Inputs.data
    @check_sql_input
    def setData(self, data=None):

        self.data = data
        self.btn_savedata.setEnabled(bool(self.data))
        self.rows = len(self.data)
        self.cols = len(self.data.domain)
        target_variable = self.data.domain.class_var
        if target_variable is not None:
            if isinstance(target_variable, Orange.data.DiscreteVariable):
                self.target = "categorical"
            if isinstance(target_variable, Orange.data.ContinuousVariable):
                self.target = "numeric"
        else:
            self.target = None
        self.update_labels()

    def _setup_gui(self):
        super()._setup_gui()
        layoutA = QGridLayout()
        layoutA.setSpacing(3)
        gui.widgetBox(self.controlArea, orientation=layoutA, box='Save dataset')
        self.target_label = QLabel()
        layoutA.addWidget(self.target_label, 0, 0)
        self.rows_label = QLabel()
        layoutA.addWidget(self.rows_label, 1, 0)
        self.cols_label = QLabel()
        gui.attributeIconDict
        layoutA.addWidget(self.cols_label, 2, 0)
        self.tableName = QLineEdit(
            placeholderText="Table name...", toolTip="Table name")
        layoutA.addWidget(self.tableName, 3, 0)
        self.btn_savedata = QPushButton(
            "Save", toolTip="Save a dataset into a DB",
            minimumWidth=120
        )
        self.btn_savedata.clicked.connect(self.saveData)
        layoutA.addWidget(self.btn_savedata, 3, 2)
        self._add_backend_controls()

    def _add_backend_controls(self):
        box = self.serverbox
        self.backends = BackendModel(Backend.available_backends())
        self.backendcombo = QComboBox(box)
        if self.backends:
            self.backendcombo.setModel(self.backends)
            names = [backend.display_name for backend in self.backends]
            if self.selected_backend and self.selected_backend in names:
                self.backendcombo.setCurrentText(self.selected_backend)
        else:
            self.Error.no_backends()
            box.setEnabled(False)
        self.backendcombo.currentTextChanged.connect(self.__backend_changed)
        box.layout().insertWidget(0, self.backendcombo)

    def __backend_changed(self):
        backend = self.get_backend()
        self.selected_backend = backend.display_name if backend else None

    def create_master_table(self):
        query = f"""
        CREATE TABLE IF NOT EXISTS datasets (
            name VARCHAR(30) PRIMARY KEY NOT NULL,
            datetime TIMESTAMP NOT NULL,
            rows INT NOT NULL,
            cols INT NOT NULL,
            class VARCHAR(30)
        )
        """
        try:
            with self.backend.execute_sql_query(query):
                pass
        except BackendError as ex:
            self.Error.connection(str(ex))

    def create_table(self, table_name):

        contMetasOriginales = 0
        cont = 0
        variables = []
        tipo_var = []
        tiene_class = 0

        if self.data.domain.class_var:
            variables.append(self.data.domain.class_var)
            tiene_class += 1

        for i in range(0, len(self.data.domain) - tiene_class):

            if contMetasOriginales == 0:
                contMetasOriginales += len(self.data.domain.metas)
                cont = contMetasOriginales
            if cont > 0:
                i -= cont

            variables.append(self.data.domain[i])

        create_table_query = f"CREATE TABLE {table_name} ("
        for variable in variables:
            if isinstance(variable, Orange.data.DiscreteVariable):
                create_table_query += f'"{str(variable.name)}" VARCHAR(30),'
            elif isinstance(variable, Orange.data.ContinuousVariable):
                create_table_query += f'"{variable.name}" FLOAT(10),'
            elif isinstance(variable, Orange.data.TimeVariable):
                create_table_query += f'"{variable.name}" TIMESTAMP,'
            elif isinstance(variable, Orange.data.StringVariable):
                create_table_query += f'"{str(variable.name)}" NUMERIC(30),'

        create_table_query = create_table_query[:-1]
        create_table_query += ")"

        print(create_table_query)

        try:
            with self.backend.execute_sql_query(create_table_query):
                pass
        except BackendError as ex:
            self.Error.connection(str(ex))

        insert_query = f"INSERT INTO {table_name} VALUES ("
        for i in range(len(variables)):
            insert_query += "%s,"
        insert_query = insert_query[:-1]  # Eliminar la coma final
        insert_query += ")"

        print(insert_query)

        for instance in self.data:
            data_row = [instance[i].value for i in range(len(variables))]  # Generar lista de valores para cada fila
            if self.data.domain.class_var:
                class_value = data_row[-1]  # Obtiene el valor de la clase
                del data_row[-1]  # Elimina la clase de su posición anterior
                data_row.insert(0, class_value)  # Inserta la clase al principio de la lista
            try:
                with self.backend.execute_sql_query(insert_query, params=data_row):
                    pass
            except BackendError as ex:
                self.Error.connection(str(ex))

    def saveData(self):

        self.clear()

        if self.tableName.text() == "":
            self.Error.connection("Table name must be filled.")
        elif self.servertext.text() == "" or self.databasetext.text() == "":
            self.Error.connection("Host and database fields must be filled.")
        elif self.data.domain.metas:
            self.Error.connection("Dataset with meta variables are not allowed.")
        else:
            self.create_master_table()
            query = "INSERT INTO public.datasets (name, datetime, rows, cols, class) VALUES ('" + self.tableName.text() + "','" + datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S') + "','" + str(self.rows) + "','" + str(self.cols) + "','" + str(
                self.target) + "');"

            try:
                with self.backend.execute_sql_query(query):
                    pass
                    self.create_table(self.tableName.text())
            except BackendError as ex:
                self.Error.connection(str(ex))

    def highlight_error(self, text=""):
        err = ['', 'QLineEdit {border: 2px solid red;}']
        self.servertext.setStyleSheet(err['server' in text or 'host' in text])
        self.usernametext.setStyleSheet(err['role' in text])
        self.databasetext.setStyleSheet(err['database' in text])

    def get_backend(self):
        if self.backendcombo.currentIndex() < 0:
            return None
        return self.backends[self.backendcombo.currentIndex()]

    def on_connection_success(self):
        super().on_connection_success()

    def on_connection_error(self, err):
        super().on_connection_error(err)
        self.highlight_error(str(err).split("\n")[0])

    def clear(self):
        self.Error.connection.clear()
        self.highlight_error()

    @classmethod
    def migrate_settings(cls, settings, version):
        if version < 2:
            # Until Orange version 3.4.4 username and password had been stored
            # in Settings.
            cm = cls._credential_manager(settings["host"], settings["port"])
            cm.username = settings["username"]
            cm.password = settings["password"]


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWTimeFeatureDB).run()
