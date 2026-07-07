import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


def test_main_window_responds_to_window_width(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(MainWindow, "_ensure_training_run_weights_available", lambda self: None)

    window = MainWindow()
    try:
        cases = [
            (960, Qt.Vertical, 2, 2),
            (1180, Qt.Horizontal, 3, 2),
            (1320, Qt.Horizontal, 3, 4),
            (1520, Qt.Horizontal, 6, 4),
        ]
        for width, image_orientation, action_columns, metric_columns in cases:
            window.resize(width, 720)
            app.processEvents()
            window._apply_responsive_layout(width)

            assert window.minimumWidth() <= width
            assert window.image_splitter.orientation() == image_orientation
            assert window._top_action_columns == action_columns
            assert window._metric_columns == metric_columns
            assert window.main_splitter.count() == 3
            assert all(size > 0 for size in window.main_splitter.sizes())
    finally:
        window.deleteLater()
        app.processEvents()
