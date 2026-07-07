APP_QSS = """
QMainWindow {
    background: #eef4f8;
}

QWidget {
    color: #172033;
    font-family: "Microsoft YaHei UI", "Segoe UI", Arial;
    font-size: 13px;
}

QFrame#TopBar {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #ffffff, stop:0.42 #effaf6, stop:0.72 #fff7e8, stop:1 #f1f5ff);
    border: 1px solid #d8e5ec;
    border-radius: 8px;
}

QFrame#Panel,
QFrame#PreviewPanel,
QFrame#StatusBar {
    background: #ffffff;
    border: 1px solid #d8e3ec;
    border-radius: 8px;
}

QFrame#PreviewPanel {
    background: #fbfdff;
}

QFrame#MetricCard {
    background: #ffffff;
    border: 1px solid #dbe6ee;
    border-radius: 8px;
}

QFrame#MetricCard[accent="blue"] {
    background: #f3f7ff;
    border-color: #b9cef8;
}

QFrame#MetricCard[accent="teal"] {
    background: #effbf8;
    border-color: #9fd8cf;
}

QFrame#MetricCard[accent="amber"] {
    background: #fff8e8;
    border-color: #f4cf86;
}

QFrame#MetricCard[accent="coral"] {
    background: #fff2ef;
    border-color: #f2b6a6;
}

QLabel#AppTitle {
    color: #11203a;
    font-size: 22px;
    font-weight: 800;
}

QLabel#SubTitle,
QLabel#Muted,
QLabel#StatusText {
    color: #5e6f82;
}

QLabel#SectionTitle {
    color: #172033;
    font-size: 15px;
    font-weight: 800;
}

QLabel#MetricValue {
    color: #1e5ad7;
    font-size: 26px;
    font-weight: 800;
}

QLabel#MetricValue[accent="teal"] {
    color: #087f72;
}

QLabel#MetricValue[accent="amber"] {
    color: #b76a00;
}

QLabel#MetricValue[accent="coral"] {
    color: #c24124;
}

QLabel#ImageView {
    background: #eef4f7;
    border: 1px solid #d2dfe8;
    border-radius: 8px;
    color: #7f8ea1;
}

QPushButton {
    background: #f4f8fb;
    color: #25415f;
    border: 1px solid #cbd9e6;
    border-radius: 7px;
    min-height: 34px;
    padding: 6px 12px;
    font-weight: 650;
}

QPushButton:hover {
    background: #eaf2f8;
    border-color: #9eb6cc;
}

QPushButton:pressed {
    background: #dce9f2;
}

QPushButton:disabled {
    background: #eef1f5;
    color: #9ca8b8;
    border-color: #e0e5ed;
}

QPushButton#PrimaryButton {
    background: #1268d3;
    color: #ffffff;
    border-color: #1268d3;
}

QPushButton#PrimaryButton:hover {
    background: #0b59b8;
}

QPushButton#DangerButton {
    background: #fff1ee;
    color: #bd2f18;
    border-color: #f5b4a6;
}

QPushButton#DangerButton:hover {
    background: #ffe5df;
}

QPushButton[tone="teal"] {
    background: #eaf9f6;
    color: #08756c;
    border-color: #9fd8cf;
}

QPushButton[tone="teal"]:hover {
    background: #d8f2ed;
}

QPushButton[tone="amber"] {
    background: #fff6df;
    color: #936000;
    border-color: #efc66c;
}

QPushButton[tone="amber"]:hover {
    background: #ffedc2;
}

QPushButton[tone="coral"] {
    background: #fff1ed;
    color: #ad3a20;
    border-color: #efb09f;
}

QPushButton[tone="coral"]:hover {
    background: #ffe3dc;
}

QPushButton[tone="blue"] {
    background: #eef4ff;
    color: #1b4fc4;
    border-color: #b9cef8;
}

QPushButton[tone="blue"]:hover {
    background: #e1ebff;
}

QComboBox,
QSpinBox,
QDoubleSpinBox,
QLineEdit,
QPlainTextEdit {
    background: #ffffff;
    border: 1px solid #c8d6e3;
    border-radius: 7px;
    min-height: 30px;
    padding: 3px 8px;
    selection-background-color: #cfe7ff;
}

QComboBox:hover,
QSpinBox:hover,
QDoubleSpinBox:hover,
QLineEdit:hover,
QPlainTextEdit:hover {
    border-color: #8eb1d0;
}

QComboBox:focus,
QSpinBox:focus,
QDoubleSpinBox:focus,
QLineEdit:focus,
QPlainTextEdit:focus {
    border-color: #43a99b;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #dce7ef;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #138bd0, stop:0.52 #22a699, stop:1 #f0a928);
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #ffffff;
    border: 2px solid #22a699;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}

QCheckBox {
    color: #334459;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 5px;
    border: 1px solid #afc1d2;
    background: #ffffff;
}

QCheckBox::indicator:checked {
    background: #22a699;
    border-color: #16877d;
}

QProgressBar {
    background: #e5edf4;
    border: 0;
    border-radius: 5px;
    height: 10px;
    text-align: center;
    color: transparent;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1268d3, stop:0.55 #22a699, stop:1 #f0a928);
    border-radius: 5px;
}

QTableWidget {
    background: #ffffff;
    alternate-background-color: #f6fafc;
    border: 1px solid #d8e3ec;
    border-radius: 8px;
    gridline-color: #edf2f6;
    selection-background-color: #d9effa;
    selection-color: #172033;
}

QTableWidget::item {
    padding: 4px 6px;
}

QHeaderView::section {
    background: #f0f6f8;
    color: #3d5065;
    border: 0;
    border-bottom: 1px solid #d8e3ec;
    padding: 7px 8px;
    font-weight: 800;
}
"""
