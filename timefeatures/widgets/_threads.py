"""Background threads for the database widgets.

Each operation runs ``worker.run()`` on its own :class:`WorkerThread`, and
the widget is told which thread ended through ``released(thread)``. That
lets a widget start a new operation from a callback of the previous one
(e.g. auto-load right after listing) without the first thread's cleanup
clearing the references of the second.

Threads are parentless and kept alive in ``_LIVE`` until they end, so a
widget can be deleted mid-operation without blocking the canvas on
``wait()`` and without destroying a ``QThread`` that is still running
(Qt aborts the process when that happens).
"""
import atexit
import functools

from AnyQt.QtCore import QThread, QTimer, pyqtSignal

_LIVE = set()

_WORKER_SIGNALS = ("finished", "failed", "progress_changed", "status_changed")


class WorkerThread(QThread):
    """Runs ``worker.run()`` directly, without an event loop: the thread
    ends as soon as the work does, so nobody has to call ``quit()``."""

    released = pyqtSignal(object)

    def __init__(self, worker):
        super().__init__()
        self.worker = worker
        self.finished.connect(self._on_finished)

    def run(self):
        self.worker.run()

    def _on_finished(self):
        # ``finished`` is emitted from the thread right before it ends;
        # wait() returns almost at once and makes the object safe to drop.
        self.wait()
        self.released.emit(self)
        # Drop the last reference on a later turn of the event loop:
        # destroying a QThread inside its own signal is not safe.
        QTimer.singleShot(0, functools.partial(_LIVE.discard, self))


def start_worker(worker, on_released):
    """Run ``worker`` on a new thread. ``on_released(thread)`` is called in
    the GUI thread once the thread has ended."""
    thread = WorkerThread(worker)
    thread.released.connect(on_released)
    _LIVE.add(thread)
    thread.start()
    return thread


def abandon(thread):
    """Detach a running operation from a widget that is going away: ask
    the worker to stop (if it supports cancelling) and cut its signals so
    no callback reaches the deleted widget. The thread then ends on its
    own."""
    worker = thread.worker
    if hasattr(worker, "is_cancelled"):
        worker.is_cancelled = True
    for name in _WORKER_SIGNALS:
        signal = getattr(worker, name, None)
        if signal is None:
            continue
        try:
            signal.disconnect()
        except TypeError:  # nothing connected
            pass
    try:
        thread.released.disconnect()
    except TypeError:
        pass


@atexit.register
def _wait_for_live_threads():
    # A QThread destroyed while running aborts the process, so on exit
    # cancel whatever is still running and wait for it to end.
    for thread in list(_LIVE):
        if hasattr(thread.worker, "is_cancelled"):
            thread.worker.is_cancelled = True
    for thread in list(_LIVE):
        thread.wait()
