"""
Background workers for GUI (placeholder).
"""

from pathlib import Path
from typing import List

from PyQt5.QtCore import QObject, QThread, pyqtSignal


class OCRWorker(QObject):
    """Worker for running OCR in background thread."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, file_paths: List[Path]):
        super().__init__()
        self.file_paths = file_paths

    def run(self):
        """Run OCR processing."""
        try:
            self.progress.emit("Starting OCR processing...")

            results = {}

            for file_path in self.file_paths:
                self.progress.emit(f"Processing {file_path.name}...")

                # TODO: Implement actual OCR processing
                # This would call the appropriate engines/connectors

                results[str(file_path)] = {"status": "success", "text": ""}

            self.progress.emit("Processing complete!")
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))
