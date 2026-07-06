APP_QSS = """
QMainWindow {
    background: #f4f7fb;
}

QWidget {
    color: #182235;
    font-family: "Microsoft YaHei UI", "Segoe UI", Arial;
    font-size: 13px;
}

QFrame#TopBar,
QFrame#Panel,
QFrame#PreviewPanel,
QFrame#MetricCard,
QFrame#StatusBar {
    background: #ffffff;
    border: 1px solid #dfe7f2;
    border-radius: 8px;
}

QLabel#AppTitle {
    color: #122033;
    font-size: 22px;
    font-weight: 700;
}

QLabel#SubTitle,
QLabel#Muted,
QLabel#StatusText {
    color: #66758a;
}

QLabel#SectionTitle {
    color: #182235;
    font-size: 15px;
    font-weight: 700;
}

QLabel#MetricValue {
    color: #155eef;
    font-size: 26px;
    font-weight: 800;
}

QLabel#ImageView {
    background: #edf2f8;
    border: 1px solid #d8e1ee;
    border-radius: 8px;
    color: #8794a8;
}

QPushButton {
    background: #eef4ff;
    color: #1b4fc4;
    border: 1px solid #cfe0ff;
    border-radius: 7px;
    min-height: 34px;
    padding: 6px 12px;
    font-weight: 600;
}

QPushButton:hover {
    background: #e2ecff;
    border-color: #a9c4ff;
}

QPushButton:pressed {
    background: #d2e0ff;
}

QPushButton:disabled {
    background: #eef1f5;
    color: #9ca8b8;
    border-color: #e0e5ed;
}

QPushButton#PrimaryButton {
    background: #155eef;
    color: #ffffff;
    border-color: #155eef;
}

QPushButton#PrimaryButton:hover {
    background: #0f4fd0;
}

QPushButton#DangerButton {
    background: #fff0f0;
    color: #c92a2a;
    border-color: #ffd1d1;
}

QPushButton#DangerButton:hover {
    background: #ffe3e3;
}

QComboBox,
QSpinBox,
QDoubleSpinBox,
QLineEdit {
    background: #ffffff;
    border: 1px solid #ccd8e8;
    border-radius: 7px;
    min-height: 30px;
    padding: 3px 8px;
}

QComboBox:hover,
QSpinBox:hover,
QDoubleSpinBox:hover,
QLineEdit:hover {
    border-color: #9ab6df;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #dce6f5;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #155eef;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #ffffff;
    border: 2px solid #155eef;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}

QCheckBox {
    color: #344256;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 5px;
    border: 1px solid #b9c7da;
    background: #ffffff;
}

QCheckBox::indicator:checked {
    background: #155eef;
    border-color: #155eef;
}

QProgressBar {
    background: #e8eef7;
    border: 0;
    border-radius: 5px;
    height: 10px;
    text-align: center;
    color: transparent;
}

QProgressBar::chunk {
    background: #155eef;
    border-radius: 5px;
}

QTableWidget {
    background: #ffffff;
    border: 1px solid #dfe7f2;
    border-radius: 8px;
    gridline-color: #edf2f8;
    selection-background-color: #e7f0ff;
    selection-color: #182235;
}

QHeaderView::section {
    background: #f6f9fd;
    color: #44546a;
    border: 0;
    border-bottom: 1px solid #dfe7f2;
    padding: 7px 8px;
    font-weight: 700;
}
"""

