"""
PyQt5 Desktop Application for Hybrid PDF OCR.
"""

import sys
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from util.logging import get_logger, setup_logging

logger = get_logger(__name__)


class HybridOCRApp(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.selected_files = []

    def init_ui(self):
        """Initialize UI components."""
        self.setWindowTitle("Hybrid PDF OCR")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Title
        title = QLabel("Hybrid PDF OCR System")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; padding: 20px;")
        layout.addWidget(title)

        # File selection section
        file_section = QHBoxLayout()

        self.file_label = QLabel("No files selected")
        file_section.addWidget(self.file_label)

        select_btn = QPushButton("Select Files")
        select_btn.clicked.connect(self.select_files)
        file_section.addWidget(select_btn)

        layout.addLayout(file_section)

        # Process button
        self.process_btn = QPushButton("Process OCR")
        self.process_btn.clicked.connect(self.process_ocr)
        self.process_btn.setEnabled(False)
        self.process_btn.setStyleSheet("font-size: 16px; padding: 10px;")
        layout.addWidget(self.process_btn)

        # Log output
        log_label = QLabel("Log Output:")
        layout.addWidget(log_label)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.log_output)

        # Status bar
        self.statusBar().showMessage("Ready")

    def select_files(self):
        """Open file dialog to select PDF/image files."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select PDF or Image Files",
            "",
            "PDF Files (*.pdf);;Image Files (*.png *.jpg *.jpeg *.tiff);;All Files (*.*)",
        )

        if files:
            self.selected_files = [Path(f) for f in files]
            self.file_label.setText(f"{len(self.selected_files)} file(s) selected")
            self.process_btn.setEnabled(True)
            self.log(f"Selected {len(self.selected_files)} file(s)")

    def process_ocr(self):
        """Process OCR on selected files."""
        if not self.selected_files:
            QMessageBox.warning(self, "No Files", "Please select files first")
            return

        self.log("Starting OCR processing...")
        self.process_btn.setEnabled(False)
        self.statusBar().showMessage("Processing...")

        try:
            # TODO: Implement actual OCR processing
            # This would integrate with the connectors/engines/router modules

            for file_path in self.selected_files:
                self.log(f"Processing: {file_path.name}")

            self.log("OCR processing completed!")
            QMessageBox.information(
                self, "Success", "OCR processing completed successfully!"
            )

        except Exception as e:
            self.log(f"Error: {str(e)}")
            QMessageBox.critical(self, "Error", f"OCR processing failed: {str(e)}")

        finally:
            self.process_btn.setEnabled(True)
            self.statusBar().showMessage("Ready")

    def log(self, message: str):
        """
        Add message to log output.

        Args:
            message: Log message
        """
        self.log_output.append(message)
        logger.info(message)


def main():
    """Main entry point."""
    # Setup logging
    setup_logging(log_level='INFO')

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Hybrid PDF OCR")

    # Create and show main window
    window = HybridOCRApp()
    window.show()

    # Run event loop
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
