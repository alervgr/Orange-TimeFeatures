import Orange
from AnyQt.QtWidgets import QComboBox
from AnyQt.QtCore import Qt
from datetime import datetime

from Orange.data import Table
from Orange.data.sql.backend import Backend
from Orange.data.sql.backend.base import BackendError
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.itemmodels import PyListModel
from Orange.widgets.utils.owbasesql import OWBaseSql
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Msg, OWWidget
from PyQt5.QtWidgets import QGridLayout, QLineEdit, QPushButton, QSizePolicy, QLabel
from orangewidget.utils.signals import Input

import re

import smtplib, ssl, time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

MAX_DL_LIMIT = 1000000


def is_postgres(backend):
    return getattr(backend, 'display_name', '') == "PostgreSQL"


class BackendModel(PyListModel):
    def data(self, index, role=Qt.DisplayRole):
        row = index.row()
        if role == Qt.DisplayRole:
            return self[row].display_name
        return super().data(index, role)


class owsavetodb(OWBaseSql, OWWidget):
    name = "Save to DB"
    description = "Save a dataset into a DB."
    icon = "icons/savedatadb.svg"
    priority = 2241
    keywords = "sql table, save, data, db, dataset"

    class Inputs:
        data = Input("Data", Orange.data.Table)

    class Outputs:
        pass

    settings_version = 2
    buttons_area_orientation = None
    selected_backend = Setting(None)
    sql = Setting("")

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
        target_variable = ""
        if self.data is not None:
            self.rows = len(self.data)
            self.cols = len(self.data.domain)
            target_variable = self.data.domain.class_var
        else:
            self.rows = 0
            self.cols = 0
            self.target = "None"
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
        self.target_label.setText("Class: None")
        layoutA.addWidget(self.target_label, 0, 0)
        self.rows_label = QLabel()
        self.rows_label.setText("Rows: 0")
        layoutA.addWidget(self.rows_label, 1, 0)
        self.cols_label = QLabel()
        self.cols_label.setText("Columns: 0")
        layoutA.addWidget(self.cols_label, 2, 0)
        self.emailDirection = QLineEdit(
            placeholderText="Email... (Optional)", toolTip="Email direction")
        layoutA.addWidget(self.emailDirection, 4, 0)
        self.tableName = QLineEdit(
            placeholderText="Table name...", toolTip="Table name")
        layoutA.addWidget(self.tableName, 3, 0)
        self.btn_savedata = QPushButton(
            "Save", toolTip="Save a dataset into a DB",
            minimumWidth=120
        )
        self.btn_savedata.clicked.connect(self.saveData)
        self.btn_savedata.setEnabled(False)
        layoutA.addWidget(self.btn_savedata, 4, 2)
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
            class VARCHAR(30),
            class_name VARCHAR(30)
        )
        """
        try:
            with self.backend.execute_sql_query(query):
                pass
        except BackendError as ex:
            self.Error.connection(str(ex))

    def create_table(self, table_name):
        start_time = time.time()
        self.progressBarInit()
        contBar = 0
        contMetasOriginales = 0
        cont = 0
        variables = []
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
                create_table_query += f'"{str(variable.name)}" VARCHAR,'
            elif isinstance(variable, Orange.data.ContinuousVariable):
                create_table_query += f'"{variable.name}" FLOAT(10),'
            elif isinstance(variable, Orange.data.TimeVariable):
                create_table_query += f'"{variable.name}" TIMESTAMP,'
            elif isinstance(variable, Orange.data.StringVariable):
                create_table_query += f'"{str(variable.name)}" VARCHAR,'

        create_table_query = create_table_query[:-1]
        create_table_query += ")"

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

        for instance in self.data:
            data_row = []
            contBar += 1
            self.progressBarSet((contBar + 1) * 100 / len(self.data))

            for i in range(len(variables)):
                if cont > 0:
                    i -= cont
                data_row.append(instance[i].value)
            if self.data.domain.class_var:
                class_value = data_row[-1]  # Obtiene el valor de la clase
                del data_row[-1]  # Elimina la clase de su posición anterior
                data_row.insert(0, class_value)  # Inserta la clase al principio de la lista
            try:
                with self.backend.execute_sql_query(insert_query, params=data_row):
                    pass
            except BackendError as ex:
                self.Error.connection(str(ex))

        self.progressBarFinished()
        if str(self.emailDirection.text()) != "":
            end_time = time.time()
            time_elapsed = end_time - start_time
            time_elapsed = round(time_elapsed, 3)
            self.send_mail(str(self.emailDirection.text()), time_elapsed)

    def send_mail(self, mail, time_elapsed):

        # Configuración de la conexión
        sender = 'savetodbodm@gmail.com'
        password = 'arnj lakd lyol rakg'
        server = 'smtp.gmail.com'
        port = 587

        # Configuración del destinatario
        to = mail

        # Configuración de las cabeceras y del mensaje
        message = MIMEMultipart("alternative")

        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime('%d-%m-%Y %H:%M:%S')

        message["Subject"] = "Upload completed - Save To DB - " + str(formatted_datetime)
        message["From"] = sender
        message["To"] = to

        if self.data.domain.class_var:
            class_name = self.data.domain.class_var.name
        else:
            class_name = None

        body = f"""\
        <html>
            <head>
            </head>
            <body>
                <h1>Save to DB - Widget</h1>
                <p>Your data upload has been completed!</p>
                <p>-Table information:</p>
                <ul>
                    <li>Table name: {str(self.tableName.text())}.</li>
                    <li>{str(self.rows_label.text())}.</li>
                    <li>{str(self.cols_label.text())}.</li>
                    <li>Class name: {str(class_name)}.</li>
                    <li>{str(self.target_label.text())}.</li>
                </ul>
                <p>-Connection information:</p>
                <ul>
                    <li>Server: {str(self.servertext.text())}.</li>
                    <li>Database name: {str(self.databasetext.text())}.</li>
                    <li>Time Elapsed: {str(time_elapsed)}s.</li>
                </ul>
            </body>
        </html>
        """

        part = MIMEText(body, "html")
        message.attach(part)

        # Envío del mensaje
        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(server, port=port) as smtp:
                smtp.starttls(context=context)
                smtp.login(sender, password)
                smtp.send_message(message)
        except Exception as ex:
            self.Error.connection(str(ex))

    def saveData(self):

        self.clear()

        email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

        if self.tableName.text() == "":
            self.Error.connection("Table name must be filled.")
        elif self.servertext.text() == "" or self.databasetext.text() == "":
            self.Error.connection("Host and database fields must be filled.")
        elif self.emailDirection.text() != "":
            if not re.match(email_regex, self.emailDirection.text()):
                self.Error.connection("The field email must be an email.")
            else:
                self.insert_data()
        else:
            self.insert_data()

    def insert_data(self):
        self.create_master_table()

        if self.data.domain.class_var:
            class_name = self.data.domain.class_var.name
        else:
            class_name = None

        query = "INSERT INTO public.datasets (name, datetime, rows, cols, class, class_name) VALUES ('" + self.tableName.text().lower() + "','" + datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S') + "','" + str(self.rows) + "','" + str(self.cols) + "','" + str(
            self.target) + "','" + str(class_name) + "');"

        try:
            with self.backend.execute_sql_query(query):
                pass
                self.create_table(self.tableName.text().lower())
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
    WidgetPreview(owsavetodb).run()
