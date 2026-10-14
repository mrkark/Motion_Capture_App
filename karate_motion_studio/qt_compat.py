"""
Qt Compatibility Layer for Karate Motion Studio.
Transparently supports PySide6 and PyQt6.
"""
import sys

QT_API = None

try:
    from PyQt6 import QtCore, QtGui, QtWidgets
    from PyQt6 import QtWebEngineWidgets, QtWebChannel
    from PyQt6.QtWebEngineCore import QWebEngineSettings
    QT_API = "PyQt6"
    Signal = QtCore.pyqtSignal
    Slot = QtCore.pyqtSlot
    Property = QtCore.pyqtProperty
except ImportError:
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
        from PySide6 import QtWebEngineWidgets, QtWebChannel
        from PySide6.QtWebEngineCore import QWebEngineSettings
        QT_API = "PySide6"
        Signal = QtCore.Signal
        Slot = QtCore.Slot
        Property = QtCore.Property
    except ImportError as e:
        raise ImportError(
            "Neither PyQt6 nor PySide6 with WebEngine is available. "
            "Please install PyQt6 and PyQt6-WebEngine: pip install PyQt6 PyQt6-WebEngine"
        ) from e

__all__ = [
    "QT_API",
    "QtCore",
    "QtGui",
    "QtWidgets",
    "QtWebEngineWidgets",
    "QWebEngineSettings",
    "QtWebChannel",
    "Signal",
    "Slot",
    "Property",
]
