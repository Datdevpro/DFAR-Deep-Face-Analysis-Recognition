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
QFrame#headerpanel, QFrame#sidebar, QFrame#mainpanel, QFrame#settingspanel, QFrame#hudcard {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 10px;
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

