import re
import smtplib
import ssl
import time
from contextlib import contextmanager
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import Orange
import Orange.data.pandas_compat as pc
from AnyQt.QtCore import QObject, QThread, Qt, pyqtSignal
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
PANDAS_SQL_CHUNKSIZE = 1000
CONNECTION_STATUS_STYLES = {
    "neutral": "QLabel { color: #888; padding-top: 4px; }",
    "success": "QLabel { color: #2e7d32; font-weight: 600; padding-top: 4px; }",
    "error": "QLabel { color: #c62828; font-weight: 600; padding-top: 4px; }",
}

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


def _sql_export_variables(table):
    """Return variables in the same order used by the legacy INSERT loop."""
    domain = table.domain
    variables = []
    if domain.class_var:
        variables.append(domain.class_var)
        variables.extend(reversed(domain.metas))
        variables.extend(domain.attributes)
    else:
        variables.extend(reversed(domain.metas))
        variables.extend(domain.attributes)
        variables.extend(domain.class_vars)
    return variables


def _dataframe_for_sql_export(table):
    frame = pc.table_to_frame(table, include_metas=True)
    variables = _sql_export_variables(table)
    frame = frame.loc[:, [variable.name for variable in variables]].copy()
    for column in frame.columns:
        series = frame[column]
        if str(getattr(series, "dtype", "")) == "category":
            frame[column] = series.astype(object)
    return frame, variables


def _iter_dataframe_chunks(frame, chunksize=PANDAS_SQL_CHUNKSIZE):
    if len(frame) == 0:
        yield frame
        return
    for start in range(0, len(frame), chunksize):
        yield frame.iloc[start:start + chunksize]


def _sqlalchemy_dtype_for_variable(var, sqltypes, dialect_name):
    # TimeVariable must be checked before ContinuousVariable because
    # in Orange the former inherits from the latter.
    if isinstance(var, Orange.data.TimeVariable):
        return sqltypes.DateTime()
    if isinstance(var, Orange.data.DiscreteVariable):
        return sqltypes.String(length=255)
    if isinstance(var, Orange.data.ContinuousVariable):
        if dialect_name == "PostgreSQL":
            from sqlalchemy.dialects import postgresql
            double_precision = getattr(postgresql, "DOUBLE_PRECISION", None)
            if double_precision is not None:
                return double_precision()
        if dialect_name == "MySQL":
            from sqlalchemy.dialects import mysql
            double = getattr(mysql, "DOUBLE", None)
            if double is not None:
                return double()
        return sqltypes.Float(precision=53)
    if isinstance(var, Orange.data.StringVariable):
        return sqltypes.Text()
    return sqltypes.Text()


def _sqlalchemy_modules():
    try:
        from sqlalchemy import create_engine, text, types as sqltypes
        from sqlalchemy.engine import URL
    except ImportError as ex:
        raise BackendError(
            "SQLAlchemy support is required to save data. "
            "Install it with: pip install SQLAlchemy"
        ) from ex
    return create_engine, URL, text, sqltypes


def _create_sqlalchemy_engine(dialect, host, port, database, username, password):
    create_engine, URL, _, _ = _sqlalchemy_modules()
    try:
        db_port = int(port) if port else None
        url = URL.create(
            drivername=dialect.sqlalchemy_drivername,
            username=username,
            password=password,
            host=host,
            port=db_port,
            database=database,
        )
        return create_engine(url)
    except Exception as ex:
        raise BackendError(str(ex)) from ex


def _create_master_table_sql(dialect):
    qi = dialect.quote_ident
    return f"""
    CREATE TABLE IF NOT EXISTS {qi('datasets')} (
        {qi('name')} VARCHAR(63) PRIMARY KEY NOT NULL,
        {qi('datetime')} TIMESTAMP NOT NULL,
        {qi('rows')} INT NOT NULL,
        {qi('cols')} INT NOT NULL,
        {qi('class')} VARCHAR(30),
        {qi('class_name')} VARCHAR(63)
    )
    """


def _insert_metadata_sql(dialect):
    qi = dialect.quote_ident
    return (
        f"INSERT INTO {qi('datasets')} "
        f"({qi('name')}, {qi('datetime')}, {qi('rows')}, "
        f"{qi('cols')}, {qi('class')}, {qi('class_name')}) "
        "VALUES (:dataset_name, :created_at, :row_count, :col_count, "
        ":target_type, :class_name)"
    )


def _send_completion_mail(
        mail, selected_backend, server_text, database_text, table_name,
        rows_text, cols_text, class_name, target_text, time_elapsed):
    # Configuración de la conexión
    sender = 'savetodbodm@gmail.com'
    password = 'arnj lakd lyol rakg'
    server = 'smtp.gmail.com'
    port = 587

    message = MIMEMultipart("alternative")

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime('%d-%m-%Y %H:%M:%S')

    message["Subject"] = "Upload completed - Save To DB - " + str(formatted_datetime)
    message["From"] = sender
    message["To"] = mail

    body = f"""\
    <html>
        <head>
        </head>
        <body>
            <h1>Save to DB - Widget</h1>
            <p>Your data upload has been completed!</p>
            <p>-Table information:</p>
            <ul>
                <li>Table name: {table_name}.</li>
                <li>{rows_text}.</li>
                <li>{cols_text}.</li>
                <li>Class name: {class_name}.</li>
                <li>{target_text}.</li>
            </ul>
            <p>-Connection information:</p>
            <ul>
                <li>Backend: {selected_backend}.</li>
                <li>Server: {server_text}.</li>
                <li>Database name: {database_text}.</li>
                <li>Time Elapsed: {time_elapsed}s.</li>
            </ul>
        </body>
    </html>
    """

    part = MIMEText(body, "html")
    message.attach(part)

    context = ssl.create_default_context()
    with smtplib.SMTP(server, port=port) as smtp:
        smtp.starttls(context=context)
        smtp.login(sender, password)
        smtp.send_message(message)


class _UploadWorker(QObject):
    progress_changed = pyqtSignal(float)
    status_changed = pyqtSignal(str)
    finished = pyqtSignal(float)
    failed = pyqtSignal(str)

    def __init__(self, *, table, dialect, connection_params, metadata,
                 email_params):
        super().__init__()
        self.table = table
        self.dialect = dialect
        self.connection_params = connection_params
        self.metadata = metadata
        self.email_params = email_params

    def run(self):
        start_time = time.time()
        engine = None
        try:
            self.status_changed.emit("Preparing data...")
            self.progress_changed.emit(2)
            frame, variables = _dataframe_for_sql_export(self.table)
            _, _, text, sqltypes = _sqlalchemy_modules()
            dtype = {
                variable.name: _sqlalchemy_dtype_for_variable(
                    variable, sqltypes, self.dialect.name
                )
                for variable in variables
            }

            self.status_changed.emit("Connecting to database...")
            self.progress_changed.emit(8)
            engine = _create_sqlalchemy_engine(self.dialect, **self.connection_params)

            total_chunks = (
                (len(frame) + PANDAS_SQL_CHUNKSIZE - 1) //
                PANDAS_SQL_CHUNKSIZE
            ) or 1

            with engine.begin() as connection:
                self.status_changed.emit("Creating metadata table...")
                connection.execute(text(_create_master_table_sql(self.dialect)))
                self.progress_changed.emit(12)

                self.status_changed.emit("Writing dataset metadata...")
                connection.execute(
                    text(_insert_metadata_sql(self.dialect)),
                    self.metadata["params"],
                )
                self.progress_changed.emit(18)

                for index, chunk in enumerate(_iter_dataframe_chunks(frame)):
                    self.status_changed.emit(
                        f"Uploading rows {index + 1}/{total_chunks}..."
                    )
                    chunk.to_sql(
                        self.metadata["table_name"],
                        con=connection,
                        if_exists="fail" if index == 0 else "append",
                        index=False,
                        dtype=dtype,
                        method="multi",
                    )
                    self.progress_changed.emit(
                        18 + ((index + 1) * 82 / total_chunks)
                    )

            time_elapsed = round(time.time() - start_time, 3)
            if self.email_params["mail"]:
                self.status_changed.emit("Sending completion email...")
                _send_completion_mail(time_elapsed=time_elapsed, **self.email_params)

            self.progress_changed.emit(100)
            self.finished.emit(time_elapsed)
        except Exception as ex:
            self.failed.emit(str(ex))
        finally:
            if engine is not None:
                engine.dispose()


# --------------------------------------------------------------------- #
#  Dialect abstraction
# --------------------------------------------------------------------- #
class _Dialect:
    """Per-RDBMS SQL differences (identifier quoting, column types) plus a
    factory that returns an object with ``execute_sql_query(query, params)``
    matching Orange's Backend interface."""

    name = ""
    sqlalchemy_drivername = ""

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
    sqlalchemy_drivername = "postgresql+psycopg2"

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
    sqlalchemy_drivername = "mysql+pymysql"

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
        self.connection_status_label = None
        self.data = None
        self.rows = 0
        self.cols = 0
        self.target = None
        self._uploading = False
        self._upload_thread = None
        self._upload_worker = None
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
        self.btn_savedata.setEnabled(bool(self.data) and not self._uploading)
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
        self._add_connection_status()

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

    def _add_connection_status(self):
        self.connection_status_label = QLabel()
        self.connection_status_label.setWordWrap(True)
        self.serverbox.layout().addWidget(self.connection_status_label)
        self._set_connection_status("Not connected", "neutral")

    def _set_connection_status(self, text, state="neutral"):
        if self.connection_status_label is None:
            return
        self.connection_status_label.setText(text)
        self.connection_status_label.setStyleSheet(
            CONNECTION_STATUS_STYLES.get(
                state, CONNECTION_STATUS_STYLES["neutral"]
            )
        )

    def __backend_changed(self):
        if self._uploading:
            return
        self.selected_backend = self.backendcombo.currentText()
        self.backend = None
        self._set_connection_status("Not connected", "neutral")

    def connect(self):
        if self._uploading:
            return
        self.backend = None
        self._set_connection_status("Connecting...", "neutral")
        super().connect()
        if self.backend is None and not self.Error.connection.is_shown():
            self._set_connection_status("Not connected", "neutral")

    # ------------------------------------------------------------------ #
    #  Data persistence
    # ------------------------------------------------------------------ #
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
        if self._uploading:
            return

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
        if self.backend is None:
            self.Error.connection("Connect to a database before saving.")
            return

        try:
            _sqlalchemy_modules()
        except BackendError as ex:
            self.Error.connection(str(ex))
            return

        if self.data.domain.class_var:
            class_name = self.data.domain.class_var.name
        else:
            class_name = None

        table_name = self.tableName.text().lower()
        self._start_upload(table_name, class_name)

    def _start_upload(self, table_name, class_name):
        self._check_db_settings()
        metadata = {
            "table_name": table_name,
            "params": {
                "dataset_name": table_name,
                "created_at": datetime.now(),
                "row_count": self.rows,
                "col_count": self.cols,
                "target_type": str(self.target),
                "class_name": class_name,
            },
        }
        connection_params = {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "username": self.username,
            "password": self.password,
        }
        email_params = {
            "mail": str(self.emailDirection.text()),
            "selected_backend": self.selected_backend,
            "server_text": str(self.servertext.text()),
            "database_text": str(self.databasetext.text()),
            "table_name": str(self.tableName.text()),
            "rows_text": str(self.rows_label.text()),
            "cols_text": str(self.cols_label.text()),
            "class_name": class_name,
            "target_text": str(self.target_label.text()),
        }

        self._uploading = True
        self._set_upload_controls_enabled(False)
        self.progressBarInit()
        self.progressBarSet(0)
        self._set_connection_status("Starting upload...", "neutral")

        self._upload_thread = QThread(self)
        self._upload_worker = _UploadWorker(
            table=self.data,
            dialect=self.dialect,
            connection_params=connection_params,
            metadata=metadata,
            email_params=email_params,
        )
        self._upload_worker.moveToThread(self._upload_thread)
        self._upload_thread.started.connect(self._upload_worker.run)
        self._upload_worker.progress_changed.connect(self.progressBarSet)
        self._upload_worker.status_changed.connect(self._on_upload_status)
        self._upload_worker.finished.connect(self._on_upload_finished)
        self._upload_worker.failed.connect(self._on_upload_failed)
        self._upload_worker.finished.connect(lambda _: self._upload_thread.quit())
        self._upload_worker.failed.connect(lambda _: self._upload_thread.quit())
        self._upload_thread.finished.connect(self._upload_worker.deleteLater)
        self._upload_thread.finished.connect(self._upload_thread.deleteLater)
        self._upload_thread.finished.connect(self._on_upload_thread_finished)
        self._upload_thread.start()

    def _on_upload_status(self, message):
        self._set_connection_status(message, "neutral")

    def _on_upload_finished(self, time_elapsed):
        self.progressBarSet(100)
        self.progressBarFinished()
        self._uploading = False
        self._set_upload_controls_enabled(True)
        self._set_connection_status(
            f"Upload completed in {time_elapsed}s", "success"
        )

    def _on_upload_failed(self, message):
        self.progressBarFinished()
        self._uploading = False
        self._set_upload_controls_enabled(True)
        self.Error.connection(message)
        self._set_connection_status(f"Upload failed: {message}", "error")

    def _on_upload_thread_finished(self):
        self._upload_thread = None
        self._upload_worker = None

    def _set_upload_controls_enabled(self, enabled):
        for widget in (
                self.connectbutton, self.btn_savedata, self.backendcombo,
                self.servertext, self.databasetext, self.usernametext,
                self.passwordtext, self.tableName, self.emailDirection):
            widget.setEnabled(enabled)
        self.btn_savedata.setEnabled(enabled and bool(self.data))

    def onDeleteWidget(self):
        if self._upload_thread is not None and self._upload_thread.isRunning():
            self._upload_thread.quit()
            self._upload_thread.wait()
        super().onDeleteWidget()

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
        port = f":{self.port}" if self.port else ""
        self._set_connection_status(
            f"Connected to {self.selected_backend}: "
            f"{self.host}{port}/{self.database}",
            "success",
        )

    def on_connection_error(self, err):
        super().on_connection_error(err)
        message = str(err).split("\n")[0]
        self.highlight_error(message)
        self._set_connection_status(f"Connection failed: {message}", "error")

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
