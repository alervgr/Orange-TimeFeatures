"""Load datasets persisted by the **Save to DB** widget directly into
an Orange ``Table``, optionally marking the class column on the fly so
no Select Columns widget is needed downstream."""

import Orange
import Orange.data.pandas_compat as pc
from AnyQt.QtCore import QObject, QThread, pyqtSignal
from AnyQt.QtWidgets import (
    QComboBox, QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QTableWidget, QTableWidgetItem
)

from Orange.data import Domain
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owbasesql import OWBaseSql
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Msg, Output, OWWidget
from orangewidget.utils.combobox import ComboBoxSearch

from timefeatures.widgets.owsavetodb import (
    CONNECTION_STATUS_STYLES,
    _DIALECTS,
    _create_sqlalchemy_engine,
    _sqlalchemy_modules,
)


# Sentinel label used inside the class-column combo when the user wants
# the dataset loaded without a class.
_NO_CLASS_LABEL = "(no class)"


def _build_domain_with_class(domain, class_name):
    """Return a new :class:`~Orange.data.Domain` where ``class_name`` (a
    string) becomes the class variable.

    The column is moved out of ``domain.attributes`` into the class slot;
    metas are preserved. If ``class_name`` is empty or not in the
    attributes, the original domain is returned unchanged.
    """
    if not class_name:
        return domain
    class_var = None
    new_attrs = []
    for var in domain.attributes:
        if var.name == class_name:
            class_var = var
        else:
            new_attrs.append(var)
    if class_var is None:
        return domain
    return Domain(new_attrs, class_var, metas=domain.metas)


class _ListDatasetsWorker(QObject):
    """Read the metadata table on a background thread so the canvas stays
    responsive while we wait for the round-trip."""

    finished = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, dialect, connection_params):
        super().__init__()
        self.dialect = dialect
        self.connection_params = connection_params

    _TABLE_MISSING_MARKERS = (
        "doesn't exist", "does not exist", "no such table",
        "unknown table", "relation", "1146",
    )

    def run(self):
        engine = None
        try:
            _, _, text, _ = _sqlalchemy_modules()
            engine = _create_sqlalchemy_engine(
                self.dialect, **self.connection_params
            )
            qi = self.dialect.quote_ident
            query = (
                f"SELECT {qi('name')}, {qi('datetime')}, {qi('rows')}, "
                f"{qi('cols')}, {qi('class')}, {qi('class_name')} "
                f"FROM {qi('datasets')} "
                f"ORDER BY {qi('datetime')} DESC"
            )
            with engine.begin() as connection:
                result = connection.execute(text(query))
                datasets = [dict(row) for row in result.mappings()]
            self.finished.emit(datasets)
        except Exception as ex:  # pylint: disable=broad-except
            msg = str(ex).lower()
            if any(m in msg for m in self._TABLE_MISSING_MARKERS):
                self.finished.emit([])
            else:
                self.failed.emit(str(ex))
        finally:
            if engine is not None:
                engine.dispose()


class _LoadTableWorker(QObject):
    """Pull a dataset into a ``pandas.DataFrame`` on a background thread."""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress_changed = pyqtSignal(float)

    def __init__(self, dialect, connection_params, table_name, total_rows):
        super().__init__()
        self.dialect = dialect
        self.connection_params = connection_params
        self.table_name = table_name
        self.total_rows = total_rows
        self.is_cancelled = False

    def run(self):
        import pandas as pd
        engine = None
        try:
            _, _, text, _ = _sqlalchemy_modules()
            engine = _create_sqlalchemy_engine(
                self.dialect, **self.connection_params
            )
            qi = self.dialect.quote_ident
            with engine.begin() as connection:
                chunks = []
                loaded_rows = 0
                for chunk in pd.read_sql(
                    text(f"SELECT * FROM {qi(self.table_name)}"),
                    connection,
                    chunksize=1000
                ):
                    if self.is_cancelled:
                        raise Exception("Load cancelled by user.")
                    chunks.append(chunk)
                    loaded_rows += len(chunk)
                    if self.total_rows > 0:
                        self.progress_changed.emit(10 + (loaded_rows / self.total_rows) * 60)
                
                if chunks:
                    frame = pd.concat(chunks, ignore_index=True)
                else:
                    frame = pd.read_sql(text(f"SELECT * FROM {qi(self.table_name)} LIMIT 0"), connection)
            self.finished.emit(frame)
        except Exception as ex:  # pylint: disable=broad-except
            self.failed.emit(str(ex))
        finally:
            if engine is not None:
                engine.dispose()


class _DeleteDatasetWorker(QObject):
    """Drop a dataset table and remove its ``datasets`` metadata row on a
    background thread, so the GUI stays responsive even on slow DROPs."""

    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, dialect, connection_params, table_name):
        super().__init__()
        self.dialect = dialect
        self.connection_params = connection_params
        self.table_name = table_name

    def run(self):
        engine = None
        try:
            _, _, text, _ = _sqlalchemy_modules()
            engine = _create_sqlalchemy_engine(
                self.dialect, **self.connection_params
            )
            qi = self.dialect.quote_ident
            with engine.begin() as connection:
                connection.execute(
                    text(f"DROP TABLE IF EXISTS {qi(self.table_name)}")
                )
                connection.execute(
                    text(
                        f"DELETE FROM {qi('datasets')} "
                        f"WHERE {qi('name')} = :name"
                    ),
                    {"name": self.table_name},
                )
            self.finished.emit(self.table_name)
        except Exception as ex:  # pylint: disable=broad-except
            self.failed.emit(str(ex))
        finally:
            if engine is not None:
                engine.dispose()


class owloadfromdb(OWBaseSql, OWWidget):
    name = "Load from DB"
    description = (
        "Load a dataset previously persisted by Save to DB, optionally "
        "marking the class column directly."
    )
    icon = "icons/loaddatadb.svg"
    priority = 2242
    keywords = (
        "sql table, load, read, fetch, data, db, dataset, postgres, "
        "postgresql, mysql"
    )

    class Inputs:
        pass

    class Outputs:
        data = Output("Data", Orange.data.Table)

    settings_version = 1
    buttons_area_orientation = None
    selected_backend = Setting("PostgreSQL")
    sql = Setting("")
    # Persisted across workflow save/load so a reopen restores the user's
    # last choice as soon as the connection comes back up.
    selected_dataset = Setting("", schema_only=True)
    selected_class = Setting("", schema_only=True)

    class Warning(OWBaseSql.Warning):
        no_datasets = Msg(
            "No datasets found. Use the 'Save to DB' node to save "
            "data first — it will create the required tables automatically."
        )

    class Error(OWBaseSql.Error):
        no_backends = Msg("Please install a backend to use this widget.")

    def __init__(self):
        self.backendcombo = None
        self.connection_status_label = None
        self.datasets_combo = None
        self.class_combo = None
        self.dataset_info_label = None
        self.btn_loaddata = None
        self.btn_refresh = None
        self.btn_delete = None
        self.btn_cancel = None
        self.preview_table = None
        self.data = None
        self._busy = False
        self._available = {}
        self._thread = None
        self._worker = None
        # Marked True after ``super().__init__()`` if the workflow
        # restored a persisted ``selected_dataset`` — that triggers a
        # one-shot ``load_data()`` once the first dataset listing comes
        # back, so reopening a saved workflow lands directly on the
        # data without the user having to click Load.
        self._auto_load_pending = False
        super().__init__()
        if self.selected_dataset:
            self._auto_load_pending = True

    # ------------------------------------------------------------------ #
    #  Dialect & connection
    # ------------------------------------------------------------------ #
    @property
    def dialect(self):
        return _DIALECTS.get(self.selected_backend, _DIALECTS["PostgreSQL"])

    def _setup_gui(self):
        super()._setup_gui()
        self._add_backend_controls()
        self._add_connection_status()
        self._add_dataset_controls()

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
        self.backendcombo.currentTextChanged.connect(self._on_backend_changed)
        box.layout().insertWidget(0, self.backendcombo)

    def _add_connection_status(self):
        self.connection_status_label = QLabel()
        self.connection_status_label.setWordWrap(True)
        self.serverbox.layout().addWidget(self.connection_status_label)
        self._set_connection_status("Not connected", "neutral")

    def _add_dataset_controls(self):
        layout = QGridLayout()
        layout.setSpacing(3)
        # Col 1 absorbs any extra horizontal space so the combo grows and
        # the buttons stay flush to the right edge instead of clumping
        # together in the middle.
        layout.setColumnStretch(1, 1)
        gui.widgetBox(self.controlArea, orientation=layout, box="Dataset")

        # Row 0: Dataset combo (searchable) + Refresh + Delete, packed in
        # a horizontal sub-layout so the combo stretches and the two
        # buttons keep their natural width with visible spacing between
        # them.
        layout.addWidget(QLabel("Dataset:"), 0, 0)

        # ``setSpacing(0)`` + explicit ``addSpacing`` calls so the gaps
        # between combo / ↻ / Delete are big enough to read on macOS,
        # where the native button chrome bleeds into the neighbour and
        # makes a 6 px gap look like the widgets are touching.
        dataset_row = QHBoxLayout()
        dataset_row.setSpacing(0)
        dataset_row.setContentsMargins(0, 0, 0, 0)

        # ``ComboBoxSearch`` is a drop-in QComboBox subclass that adds a
        # filter box in the popup — handy when there are dozens of saved
        # datasets to scroll through.
        self.datasets_combo = ComboBoxSearch()
        self.datasets_combo.setEnabled(False)
        self.datasets_combo.currentTextChanged.connect(self._on_dataset_changed)
        dataset_row.addWidget(self.datasets_combo, 1)  # stretch=1: take the slack

        dataset_row.addSpacing(20)

        self.btn_refresh = QPushButton("↻")
        self.btn_refresh.setToolTip(
            "Reload the list of datasets registered on the server."
        )
        self.btn_refresh.setFixedWidth(44)
        self.btn_refresh.setEnabled(False)
        self.btn_refresh.clicked.connect(self._refresh_datasets)
        dataset_row.addWidget(self.btn_refresh)

        dataset_row.addSpacing(20)

        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setToolTip(
            "Drop the selected dataset's table and remove it from the "
            "datasets registry."
        )
        self.btn_delete.setEnabled(False)
        self.btn_delete.clicked.connect(self._delete_dataset)
        dataset_row.addWidget(self.btn_delete)

        layout.addLayout(dataset_row, 0, 1, 1, 3)

        # Row 1: Info label spans all four columns.
        self.dataset_info_label = QLabel(
            "Connect to a database to list available datasets."
        )
        self.dataset_info_label.setWordWrap(True)
        layout.addWidget(self.dataset_info_label, 1, 0, 1, 4)

        # Row 2: Class column selector spans cols 1-3.
        layout.addWidget(QLabel("Class column:"), 2, 0)
        self.class_combo = QComboBox()
        self.class_combo.setEnabled(False)
        self.class_combo.addItem(_NO_CLASS_LABEL)
        self.class_combo.currentTextChanged.connect(self._on_class_changed)
        layout.addWidget(self.class_combo, 2, 1, 1, 3)

        self.btn_loaddata = QPushButton(
            "Load",
            toolTip="Load the selected dataset into Orange.",
            minimumWidth=120,
        )
        self.btn_loaddata.clicked.connect(self.load_data)
        self.btn_loaddata.setEnabled(False)
        
        self.btn_cancel = QPushButton("Cancel", minimumWidth=80)
        self.btn_cancel.clicked.connect(self.cancelLoad)
        self.btn_cancel.setEnabled(False)
        
        button_row = QHBoxLayout()
        button_row.setSpacing(0)
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.addStretch(1)
        button_row.addWidget(self.btn_cancel)
        button_row.addSpacing(20)
        button_row.addWidget(self.btn_loaddata)
        layout.addLayout(button_row, 3, 0, 1, 4)

        # Row 4: Preview Table
        layout.addWidget(QLabel("Data Preview (Top 50 rows):"), 4, 0, 1, 4)
        self.preview_table = QTableWidget()
        self.preview_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.preview_table.setSelectionMode(QTableWidget.NoSelection)
        self.preview_table.setFixedHeight(150)
        layout.addWidget(self.preview_table, 5, 0, 1, 4)

    def _set_connection_status(self, message, state="neutral"):
        if self.connection_status_label is None:
            return
        self.connection_status_label.setText(message)
        self.connection_status_label.setStyleSheet(
            CONNECTION_STATUS_STYLES.get(
                state, CONNECTION_STATUS_STYLES["neutral"]
            )
        )

    def _on_backend_changed(self):
        if self._busy:
            return
        self.selected_backend = self.backendcombo.currentText()
        self.backend = None
        self._reset_dataset_controls()
        self._set_connection_status("Not connected", "neutral")

    def _reset_dataset_controls(self):
        self.datasets_combo.blockSignals(True)
        self.datasets_combo.clear()
        self.datasets_combo.blockSignals(False)
        self.datasets_combo.setEnabled(False)

        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        self.class_combo.addItem(_NO_CLASS_LABEL)
        self.class_combo.blockSignals(False)
        self.class_combo.setEnabled(False)

        self.btn_loaddata.setEnabled(False)
        if self.btn_cancel is not None:
            self.btn_cancel.setEnabled(False)
        if self.btn_refresh is not None:
            self.btn_refresh.setEnabled(False)
        if self.btn_delete is not None:
            self.btn_delete.setEnabled(False)
        if self.preview_table is not None:
            self.preview_table.clear()
            self.preview_table.setRowCount(0)
            self.preview_table.setColumnCount(0)
        self.dataset_info_label.setText(
            "Connect to a database to list available datasets."
        )
        self._available = {}

    def _connection_params(self):
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "username": self.username,
            "password": self.password,
        }

    def connect(self):
        if self._busy:
            return
        self.backend = None
        self._reset_dataset_controls()
        self._set_connection_status("Connecting...", "neutral")
        super().connect()
        if self.backend is None and not self.Error.connection.is_shown():
            self._set_connection_status("Not connected", "neutral")

    def get_backend(self):
        """``OWBaseSql.connect`` calls this to obtain a class-like factory
        accepting ``(host, port, database, user, password)``."""
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
        self._populate_datasets()

    def on_connection_error(self, err):
        super().on_connection_error(err)
        message = str(err).split("\n")[0]
        self._set_connection_status(f"Connection failed: {message}", "error")

    # ------------------------------------------------------------------ #
    #  Dataset listing
    # ------------------------------------------------------------------ #
    def _populate_datasets(self):
        self.Warning.no_datasets.clear()
        self._set_connection_status("Listing datasets...", "neutral")
        self._busy = True

        self._thread = QThread(self)
        self._worker = _ListDatasetsWorker(
            dialect=self.dialect,
            connection_params=self._connection_params(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_datasets_loaded)
        self._worker.failed.connect(self._on_datasets_failed)
        self._worker.finished.connect(lambda _: self._thread.quit())
        self._worker.failed.connect(lambda _: self._thread.quit())
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    def _on_datasets_loaded(self, datasets):
        self._busy = False
        self._available = {ds["name"]: ds for ds in datasets}

        self.datasets_combo.blockSignals(True)
        self.datasets_combo.clear()
        for ds in datasets:
            self.datasets_combo.addItem(ds["name"])
        self.datasets_combo.blockSignals(False)

        # Refresh is always usable once we've successfully listed the
        # datasets table — even with zero rows, the user might want to
        # re-poll after a parallel Save to DB.
        self.btn_refresh.setEnabled(True)

        if not datasets:
            self.Warning.no_datasets()
            self.datasets_combo.setEnabled(False)
            self.btn_delete.setEnabled(False)
            # Cancel any pending auto-load: the saved dataset doesn't
            # exist on the server (probably got dropped from outside).
            self._auto_load_pending = False
            self._set_connection_status(
                f"Connected to {self.selected_backend} — no datasets found",
                "neutral",
            )
            return

        self.datasets_combo.setEnabled(True)
        self.btn_delete.setEnabled(True)

        names = [ds["name"] for ds in datasets]
        persisted_was_found = self.selected_dataset in names
        if persisted_was_found:
            self.datasets_combo.setCurrentText(self.selected_dataset)
        else:
            self.datasets_combo.setCurrentIndex(0)
            self.selected_dataset = names[0]

        # Force a refresh in case the current text didn't change.
        self._on_dataset_changed(self.datasets_combo.currentText())
        port = f":{self.port}" if self.port else ""
        plural = "s" if len(datasets) != 1 else ""
        self._set_connection_status(
            f"Connected to {self.selected_backend}: "
            f"{self.host}{port}/{self.database} "
            f"({len(datasets)} dataset{plural})",
            "success",
        )

        # Auto-load on workflow open: if a persisted dataset just came
        # back successfully, fire Load now so the user lands on the
        # data without an extra click. Only honoured once per widget
        # lifetime — manual Refreshes clear the flag.
        if self._auto_load_pending and persisted_was_found:
            self._auto_load_pending = False
            self.load_data()

    def _on_datasets_failed(self, message):
        self._busy = False
        self._set_connection_status(
            f"Failed to list datasets: {message}", "error"
        )
        self.Error.connection(message)

    def _on_thread_finished(self):
        self._thread = None
        self._worker = None

    # ------------------------------------------------------------------ #
    #  Refresh / Delete
    # ------------------------------------------------------------------ #
    def _refresh_datasets(self):
        """Re-list the datasets without dropping the connection.

        Distinct from the connection flow because the auto-load flag is
        cleared first — Refresh is an explicit "show me the latest" and
        should not surprise the user by loading a dataset behind their
        back.
        """
        if self._busy:
            return
        self._auto_load_pending = False
        self._populate_datasets()

    def _delete_dataset(self):
        """Drop the selected dataset's table and unregister it. Always
        gated by a confirmation dialog — there's no Orange-side undo."""
        if self._busy or not self.selected_dataset:
            return

        target = self.selected_dataset
        answer = QMessageBox.question(
            self,
            "Delete dataset?",
            (
                f"This will drop the table <b>{target}</b> from the "
                f"database and remove its entry from the "
                f"<code>datasets</code> registry.<br><br>"
                f"This action cannot be undone."
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        self._busy = True
        self._set_load_controls_enabled(False)
        self._set_connection_status(f"Deleting {target}...", "neutral")

        self._thread = QThread(self)
        self._worker = _DeleteDatasetWorker(
            dialect=self.dialect,
            connection_params=self._connection_params(),
            table_name=target,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_delete_finished)
        self._worker.failed.connect(self._on_delete_failed)
        self._worker.finished.connect(lambda *_: self._thread.quit())
        self._worker.failed.connect(lambda *_: self._thread.quit())
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    def _on_delete_finished(self, table_name):
        self._busy = False
        self._set_load_controls_enabled(True)
        self._set_connection_status(f"Deleted {table_name}.", "success")
        # The persisted selection no longer exists; reset the persisted
        # picks and pull the fresh list so the combo reflects reality.
        self.selected_dataset = ""
        self.selected_class = ""
        self._auto_load_pending = False
        self._populate_datasets()

    def _on_delete_failed(self, message):
        self._busy = False
        self._set_load_controls_enabled(True)
        self.Error.connection(message)
        self._set_connection_status(f"Delete failed: {message}", "error")

    # ------------------------------------------------------------------ #
    #  Per-dataset reaction
    # ------------------------------------------------------------------ #
    def _on_dataset_changed(self, name):
        self.selected_dataset = name
        if not name or name not in self._available:
            self.class_combo.setEnabled(False)
            self.btn_loaddata.setEnabled(False)
            self.dataset_info_label.setText("(no dataset selected)")
            return

        ds = self._available[name]
        info = (
            f"<b>{name}</b><br>"
            f"Saved: {ds['datetime']}<br>"
            f"{ds['rows']} rows · {ds['cols']} columns"
        )
        if ds.get("class_name"):
            info += f"<br>Original class: <code>{ds['class_name']}</code>"
            if ds.get("class"):
                info += f" ({ds['class']})"
        self.dataset_info_label.setText(info)

        # header back if 0, but 50 is also fast) so it's fine to run synchronously here.
        try:
            _, _, text, _ = _sqlalchemy_modules()
            engine = _create_sqlalchemy_engine(
                self.dialect, **self._connection_params()
            )
            qi = self.dialect.quote_ident
            with engine.begin() as connection:
                result = connection.execute(
                    text(f"SELECT * FROM {qi(name)} LIMIT 50")
                )
                columns = list(result.keys())
                rows = [dict(row) for row in result.mappings()]
            engine.dispose()
        except Exception as ex:  # pylint: disable=broad-except
            self.Error.connection(str(ex))
            return
            
        self.preview_table.clear()
        self.preview_table.setColumnCount(len(columns))
        self.preview_table.setHorizontalHeaderLabels(columns)
        self.preview_table.setRowCount(len(rows))
        for r_idx, row in enumerate(rows):
            for c_idx, col in enumerate(columns):
                item = QTableWidgetItem(str(row[col]) if row[col] is not None else "")
                self.preview_table.setItem(r_idx, c_idx, item)

        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        self.class_combo.addItem(_NO_CLASS_LABEL)
        for col in columns:
            self.class_combo.addItem(col)
        self.class_combo.blockSignals(False)
        self.class_combo.setEnabled(True)

        # Pick the default: persisted choice → metadata's class_name → none.
        chosen = ""
        if self.selected_class and self.selected_class in columns:
            chosen = self.selected_class
        elif ds.get("class_name") and ds["class_name"] in columns:
            chosen = ds["class_name"]

        self.class_combo.setCurrentText(chosen or _NO_CLASS_LABEL)
        self.selected_class = chosen

        self.btn_loaddata.setEnabled(True)

    def _on_class_changed(self, text):
        if text == _NO_CLASS_LABEL:
            self.selected_class = ""
        else:
            self.selected_class = text

    # ------------------------------------------------------------------ #
    #  Loading
    # ------------------------------------------------------------------ #
    def load_data(self):
        if self._busy or not self.selected_dataset:
            return

        self.Error.connection.clear()
        self._busy = True
        self._set_load_controls_enabled(False)
        self.progressBarInit()
        self.progressBarSet(10)
        self._set_connection_status(
            f"Loading {self.selected_dataset}...", "neutral"
        )

        self._thread = QThread(self)
        
        total_rows = self._available[self.selected_dataset].get("rows", 0)
        self._worker = _LoadTableWorker(
            dialect=self.dialect,
            connection_params=self._connection_params(),
            table_name=self.selected_dataset,
            total_rows=total_rows,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress_changed.connect(self.progressBarSet)
        self._worker.finished.connect(self._on_table_loaded)
        self._worker.failed.connect(self._on_table_failed)
        self._worker.finished.connect(lambda *_: self._thread.quit())
        self._worker.failed.connect(lambda *_: self._thread.quit())
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    def _on_table_loaded(self, frame):
        self.progressBarSet(70)
        try:
            table = pc.table_from_frame(frame)
            class_name = self.selected_class
            if class_name:
                new_domain = _build_domain_with_class(table.domain, class_name)
                if new_domain is not table.domain:
                    table = table.transform(new_domain)
        except Exception as ex:  # pylint: disable=broad-except
            self._on_table_failed(str(ex))
            return

        self.Outputs.data.send(table)
        self.progressBarSet(100)
        self.progressBarFinished()
        self._busy = False
        self._set_load_controls_enabled(True)
        self._set_connection_status(
            f"Loaded {self.selected_dataset} ({len(table)} rows)",
            "success",
        )

    def _on_table_failed(self, message):
        self.progressBarFinished()
        self._busy = False
        self._set_load_controls_enabled(True)
        self.Error.connection(message)
        self._set_connection_status(f"Load failed: {message}", "error")
        self.Outputs.data.send(None)

    def cancelLoad(self):
        if self._worker is not None:
            self._worker.is_cancelled = True
            if self.btn_cancel is not None:
                self.btn_cancel.setEnabled(False)
            self._set_connection_status("Cancelling load...", "neutral")

    def _set_load_controls_enabled(self, enabled):
        for widget in (
            self.connectbutton, self.btn_loaddata, self.backendcombo,
            self.servertext, self.databasetext, self.usernametext,
            self.passwordtext, self.datasets_combo, self.class_combo,
            self.btn_refresh, self.btn_delete,
        ):
            if widget is not None:
                widget.setEnabled(enabled)
        if self.btn_cancel is not None:
            self.btn_cancel.setEnabled(not enabled)

    def onDeleteWidget(self):
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
        super().onDeleteWidget()

    def clear(self):
        self.Error.connection.clear()


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(owloadfromdb).run()
