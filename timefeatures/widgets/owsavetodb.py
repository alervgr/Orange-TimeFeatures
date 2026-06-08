import re
import smtplib
import ssl
import time
from contextlib import contextmanager
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import Orange
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QComboBox
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
from PyQt5.QtWidgets import (
    QGridLayout, QLabel, QLineEdit, QPushButton, QSizePolicy,
)
from orangewidget.utils.signals import Input

MAX_DL_LIMIT = 1000000

# PostgreSQL identifier rules: letter/underscore start, alphanumeric/_, max 63.
# MySQL identifier rules are looser (allow digit start, 64 chars) but accepting
# this subset works on both dialects.
TABLE_NAME_REGEX = re.compile(r'^[A-Za-z_][A-Za-z0-9_]{0,62}$')


def quote_ident(name):
    """PostgreSQL-style identifier quoting. Kept at module level for the
    test suite; widget code goes through ``self.dialect.quote_ident``."""
    return '"' + str(name).replace('"', '""') + '"'


def is_postgres(backend):
    return getattr(backend, 'display_name', '') == "PostgreSQL"


# --------------------------------------------------------------------- #
#  Dialect abstraction
# --------------------------------------------------------------------- #
class _Dialect:
    """Per-RDBMS SQL differences (identifier quoting, column types) plus a
    factory that returns an object with ``execute_sql_query(query, params)``
    matching Orange's Backend interface."""

    name = ""

    @staticmethod
    def quote_ident(name):  # pragma: no cover - abstract
        raise NotImplementedError

    @staticmethod
    def column_type(var):  # pragma: no cover - abstract
        raise NotImplementedError

    def backend_factory(self):  # pragma: no cover - abstract
        raise NotImplementedError


class _PostgresDialect(_Dialect):
    name = "PostgreSQL"

    @staticmethod
    def quote_ident(name):
        return '"' + str(name).replace('"', '""') + '"'

    @staticmethod
    def column_type(var):
        # TimeVariable must be checked before ContinuousVariable because
        # in Orange the former inherits from the latter.
        if isinstance(var, Orange.data.TimeVariable):
            return "TIMESTAMP"
        if isinstance(var, Orange.data.DiscreteVariable):
            return "VARCHAR(255)"
        if isinstance(var, Orange.data.ContinuousVariable):
            return "DOUBLE PRECISION"
        if isinstance(var, Orange.data.StringVariable):
            return "TEXT"
        return "TEXT"

    def backend_factory(self):
        for backend in Backend.available_backends():
            if backend.display_name == "PostgreSQL":
                return backend
        return None


class _MySQLDialect(_Dialect):
    name = "MySQL"

    @staticmethod
    def quote_ident(name):
        # MySQL quotes identifiers with backticks; escape internal backticks
        # by doubling.
        return '`' + str(name).replace('`', '``') + '`'

    @staticmethod
    def column_type(var):
        # TimeVariable must be checked before ContinuousVariable because
        # in Orange the former inherits from the latter.
        if isinstance(var, Orange.data.TimeVariable):
            # TIMESTAMP in MySQL is limited to 1970-2038; DATETIME is wider.
            return "DATETIME"
        if isinstance(var, Orange.data.DiscreteVariable):
            return "VARCHAR(255)"
        if isinstance(var, Orange.data.ContinuousVariable):
            # MySQL's FLOAT(10) means (precision, scale); plain DOUBLE is
            # what users actually want.
            return "DOUBLE"
        if isinstance(var, Orange.data.StringVariable):
            return "TEXT"
        return "TEXT"

    def backend_factory(self):
        return _MySQLBackend


class _MySQLBackend:
    """Minimal Backend-compatible wrapper around ``pymysql``.

    The widget only writes, so we don't implement the full Backend
    introspection API — just what ``execute_sql_query(query, params)``
    needs to be a drop-in for ``self.backend`` in ``owsavetodb``.
    """

    display_name = "MySQL"

    def __init__(self, params):
        try:
            import pymysql  # noqa: WPS433 - optional dep, imported lazily
        except ImportError as ex:
            raise BackendError(
                "MySQL support requires the 'pymysql' package. "
                "Install it with: pip install pymysql"
            ) from ex

        try:
            self.conn = pymysql.connect(
                host=params.get("host") or "localhost",
                port=int(params.get("port") or 3306),
                user=params.get("user"),
                password=params.get("password") or "",
                database=params.get("database"),
                autocommit=False,
            )
        except Exception as ex:
            raise BackendError(str(ex)) from ex

    @contextmanager
    def execute_sql_query(self, query, params=None):
        cursor = self.conn.cursor()
        try:
            if params:
                cursor.execute(query, tuple(params))
            else:
                cursor.execute(query)
            self.conn.commit()
            yield cursor
        except Exception as ex:
            self.conn.rollback()
            raise BackendError(str(ex)) from ex
        finally:
            cursor.close()


_DIALECTS = {
    "PostgreSQL": _PostgresDialect(),
    "MySQL": _MySQLDialect(),
}


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
    keywords = "sql table, save, data, db, dataset, postgres, postgresql, mysql"

    class Inputs:
        data = Input("Data", Orange.data.Table)

    class Outputs:
        pass

    settings_version = 2
    buttons_area_orientation = None
    # The selected dialect ("PostgreSQL" or "MySQL"). Defaults to PostgreSQL
    # so old workflows behave the same.
    selected_backend = Setting("PostgreSQL")
    sql = Setting("")

    class Warning(OWBaseSql.Warning):
        missing_extension = Msg("Database is missing extensions: {}")

    class Error(OWBaseSql.Error):
        no_backends = Msg("Please install a backend to use this widget.")

    def __init__(self):
        # Lint
        self.backendcombo = None
        self.data = None
        self.rows = 0
        self.cols = 0
        self.target = None
        super().__init__()

    # ------------------------------------------------------------------ #
    #  Dialect handling
    # ------------------------------------------------------------------ #
    @property
    def dialect(self):
        return _DIALECTS.get(self.selected_backend, _DIALECTS["PostgreSQL"])

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
        self.backendcombo = QComboBox(box)
        for name in _DIALECTS:
            self.backendcombo.addItem(name)
        if self.selected_backend in _DIALECTS:
            self.backendcombo.setCurrentText(self.selected_backend)
        else:
            self.selected_backend = "PostgreSQL"
            self.backendcombo.setCurrentText("PostgreSQL")
        self.backendcombo.currentTextChanged.connect(self.__backend_changed)
        box.layout().insertWidget(0, self.backendcombo)

    def __backend_changed(self):
        self.selected_backend = self.backendcombo.currentText()

    # ------------------------------------------------------------------ #
    #  Data persistence
    # ------------------------------------------------------------------ #
    def create_master_table(self):
        qi = self.dialect.quote_ident
        query = f"""
        CREATE TABLE IF NOT EXISTS {qi('datasets')} (
            {qi('name')} VARCHAR(63) PRIMARY KEY NOT NULL,
            {qi('datetime')} TIMESTAMP NOT NULL,
            {qi('rows')} INT NOT NULL,
            {qi('cols')} INT NOT NULL,
            {qi('class')} VARCHAR(30),
            {qi('class_name')} VARCHAR(63)
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

        qi = self.dialect.quote_ident
        ct = self.dialect.column_type
        col_defs = ", ".join(
            f"{qi(variable.name)} {ct(variable)}"
            for variable in variables
        )
        create_table_query = f"CREATE TABLE {qi(table_name)} ({col_defs})"

        try:
            with self.backend.execute_sql_query(create_table_query):
                pass
        except BackendError as ex:
            self.Error.connection(str(ex))

        placeholders = ", ".join(["%s"] * len(variables))
        insert_query = (
            f"INSERT INTO {qi(table_name)} VALUES ({placeholders})"
        )

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
                del data_row[-1]            # Elimina la clase de su posición anterior
                data_row.insert(0, class_value)  # Inserta la clase al principio
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
                    <li>Backend: {self.selected_backend}.</li>
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
        elif not TABLE_NAME_REGEX.match(self.tableName.text().lower()):
            self.Error.connection(
                "Table name must start with a letter or underscore and "
                "contain only letters, digits and underscores (max 63 chars)."
            )
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

        table_name = self.tableName.text().lower()
        qi = self.dialect.quote_ident
        # No schema qualifier: works in PostgreSQL (uses `public` by default)
        # and in MySQL (no schema concept).
        query = (
            f"INSERT INTO {qi('datasets')} "
            f"({qi('name')}, {qi('datetime')}, {qi('rows')}, "
            f"{qi('cols')}, {qi('class')}, {qi('class_name')}) "
            "VALUES (%s, %s, %s, %s, %s, %s)"
        )
        params = [
            table_name,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            self.rows,
            self.cols,
            str(self.target),
            class_name,
        ]

        try:
            with self.backend.execute_sql_query(query, params=params):
                pass
            self.create_table(table_name)
        except BackendError as ex:
            self.Error.connection(str(ex))

    def highlight_error(self, text=""):
        err = ['', 'QLineEdit {border: 2px solid red;}']
        self.servertext.setStyleSheet(err['server' in text or 'host' in text])
        self.usernametext.setStyleSheet(err['role' in text])
        self.databasetext.setStyleSheet(err['database' in text])

    def get_backend(self):
        """OWBaseSql calls this from ``connect()`` to obtain a callable
        ``backend(params)`` that returns an object exposing
        ``execute_sql_query``. We delegate to the active dialect."""
        factory = self.dialect.backend_factory()
        if factory is None:
            self.Error.no_backends()
            return None
        return factory

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
