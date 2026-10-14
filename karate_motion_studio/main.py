#!/usr/bin/env python3
"""
Karate Motion Studio — Desktop Motion Capture Application for Karate Shotokan.
Entry Point.
"""
import os
import sys

# Silence internal C++ glog / TFLite / absl messages via environment
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_LOGGING_MIN_SEVERITY"] = "3"


from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qt_compat import QtCore, QtGui, QtWidgets
from gui.main_window import MainWindow


def main():
    # High-DPI support
    if hasattr(QtCore.Qt.ApplicationAttribute, "AA_EnableHighDpiScaling"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt.ApplicationAttribute, "AA_UseHighDpiPixmaps"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Karate Motion Studio")
    app.setOrganizationName("Bunkai Explorer")
    app_font = QtGui.QFont("Segoe UI")
    app_font.setPointSize(9)
    app.setFont(app_font)

    # Dark Palette fallback
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(26, 27, 30))
    dark_palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(223, 223, 224))
    dark_palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(18, 19, 21))
    dark_palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(33, 35, 40))
    dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor(223, 223, 224))
    dark_palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor(223, 223, 224))
    dark_palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(223, 223, 224))
    dark_palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(43, 45, 51))
    dark_palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(223, 223, 224))
    dark_palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor(224, 82, 96))
    dark_palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(224, 82, 96))
    dark_palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(255, 255, 255))
    app.setPalette(dark_palette)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
