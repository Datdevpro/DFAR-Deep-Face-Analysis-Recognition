"""Theme tokens and stylesheet for a modern technical dark UI."""

BG = "#0F1115"
PANEL = "#151922"
PANEL_ALT = "#1B2130"
TEXT = "#E6EAF2"
MUTED = "#9AA4B5"
ACCENT = "#4C8DFF"
SUCCESS = "#49C27D"
ERROR = "#E35D6A"
BORDER = "#2A3242"

STYLESHEET = f"""
QMainWindow {{
    background: {BG};
    color: {TEXT};
}}
QWidget {{
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}}
QMenuBar {{
    background: {PANEL};
    color: {TEXT};
    border-bottom: 1px solid {BORDER};
    padding: 4px 8px;
}}
QMenuBar::item:selected {{
    background: {PANEL_ALT};
}}
QMenu {{
    background: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
}}
QMenu::item:selected {{
    background: {PANEL_ALT};
}}
QFrame#headerpanel, QFrame#sidebar, QFrame#mainpanel, QFrame#settingspanel, QFrame#hudcard,
QFrame#navRail, QFrame#panelCard {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
QFrame#navRail {{
    border-radius: 12px;
}}
QGroupBox {{
    font-weight: 600;
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-top: 12px;
    padding: 12px 10px 10px 10px;
    background: {PANEL_ALT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {MUTED};
    font-size: 12px;
}}
QLabel {{
    color: {TEXT};
}}
QLabel#title {{
    font-size: 21px;
    font-weight: 700;
}}
QLabel#sectionTitle {{
    font-size: 14px;
    font-weight: 600;
}}
QLabel#subtle {{
    color: {MUTED};
    font-size: 12px;
}}
QLabel#statusDot {{
    min-width: 10px;
    max-width: 10px;
    min-height: 10px;
    max-height: 10px;
    border-radius: 5px;
    background: {MUTED};
}}
QLineEdit, QComboBox, QTextEdit {{
    color: {TEXT};
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 6px;
    selection-background-color: {ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox::down-arrow {{
    width: 10px;
    height: 10px;
}}
QComboBox QAbstractItemView {{
    color: {TEXT};
    background: #1E2636;
    border: 1px solid {ACCENT};
    border-radius: 6px;
    selection-background-color: {ACCENT};
    selection-color: #ffffff;
    outline: none;
    padding: 4px;
}}
QComboBox QAbstractItemView::item {{
    color: {TEXT};
    padding: 6px 10px;
    min-height: 26px;
    border-radius: 4px;
}}
QComboBox QAbstractItemView::item:hover {{
    background: #2a3857;
    color: #ffffff;
}}
QComboBox QAbstractItemView::item:selected {{
    background: {ACCENT};
    color: #ffffff;
}}
QTextEdit#logPanel {{
    font-family: "Cascadia Code", "JetBrains Mono", monospace;
    font-size: 12px;
}}
QPushButton {{
    color: {TEXT};
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 8px 12px;
    text-align: left;
}}
QPushButton:hover {{
    border: 1px solid {ACCENT};
    background: #232d42;
}}
QPushButton:pressed {{
    background: #2a3857;
}}
QPushButton#primary {{
    background: {ACCENT};
    color: #0B1220;
    border: 1px solid {ACCENT};
    font-weight: 600;
}}
QPushButton#primary:hover {{
    background: #75A8FF;
}}
QPushButton#ghost {{
    background: transparent;
    color: {MUTED};
}}
QPushButton#danger {{
    background: #4A1F28;
    border: 1px solid {ERROR};
    color: #FFDDE2;
}}
QPushButton#danger:hover {{
    background: #642631;
}}
QPushButton#run {{
    background: {SUCCESS};
    color: #0B1220;
    border: 1px solid {SUCCESS};
    font-weight: 700;
}}
QPushButton#run:hover {{
    background: #5ED492;
}}
QPushButton#save {{
    background: {ACCENT};
    color: #0B1220;
    border: 1px solid {ACCENT};
    font-weight: 600;
}}
QPushButton#save:hover {{
    background: #75A8FF;
}}
QPushButton#nav {{
    background: transparent;
    border: 1px solid transparent;
    text-align: left;
    padding: 10px 12px;
}}
QPushButton#nav:checked {{
    background: #232d42;
    border: 1px solid {ACCENT};
    color: {TEXT};
}}
QPushButton#nav:hover {{
    border: 1px solid {BORDER};
    background: #1c2434;
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
QListWidget {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QListWidget::item:selected {{
    background: #2a3857;
}}
QSlider::groove:horizontal {{
    height: 6px;
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
    background: {ACCENT};
}}
QCheckBox {{
    color: {MUTED};
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid {BORDER};
    background: {PANEL_ALT};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}
QProgressBar {{
    color: {TEXT};
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    text-align: center;
}}
QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 5px;
}}
"""

