"""
Custom widgets for GUI application.
"""

from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class FileSelector(QWidget):
    """Widget for file selection."""

    files_selected = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # File list
        self.file_list = QListWidget()
        layout.addWidget(self.file_list)

        # Buttons
        btn_layout = QHBoxLayout()

        add_btn = QPushButton("Add Files")
        add_btn.clicked.connect(self.add_files)
        btn_layout.addWidget(add_btn)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_selected)
        btn_layout.addWidget(remove_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_all)
        btn_layout.addWidget(clear_btn)

        layout.addLayout(btn_layout)

    def add_files(self):
        """Add files to list."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Files",
            "",
            "PDF Files (*.pdf);;Image Files (*.png *.jpg *.jpeg *.tiff);;All Files (*.*)",
        )

        if files:
            for file_path in files:
                self.file_list.addItem(file_path)

            self.files_selected.emit(self.get_files())

    def remove_selected(self):
        """Remove selected files from list."""
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))

        self.files_selected.emit(self.get_files())

    def clear_all(self):
        """Clear all files."""
        self.file_list.clear()
        self.files_selected.emit([])

    def get_files(self):
        """Get list of file paths."""
        return [
            Path(self.file_list.item(i).text()) for i in range(self.file_list.count())
        ]


class LogViewer(QWidget):
    """Widget for viewing logs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: monospace; font-size: 10pt;")
        layout.addWidget(self.log_text)

        # Clear button
        clear_btn = QPushButton("Clear Logs")
        clear_btn.clicked.connect(self.clear)
        layout.addWidget(clear_btn)

    def append(self, message: str):
        """Append message to log."""
        self.log_text.append(message)

    def clear(self):
        """Clear logs."""
        self.log_text.clear()


class ProgressWidget(QWidget):
    """Widget for showing progress."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

    def set_status(self, status: str):
        """Set status text."""
        self.status_label.setText(status)
