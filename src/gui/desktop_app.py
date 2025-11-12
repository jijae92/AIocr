"""
PyQt5 Desktop Application for Hybrid PDF OCR.

Comprehensive GUI with PDF OCR tab, file/folder selection, page range,
engine selection, threshold settings, progress tracking, result preview,
export options, and error view with thumbnails.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt, QThreadPool, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QTextCharFormat, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
)

from cache.store import CacheManager
from util.logging import get_logger, setup_logging
from util.config import load_config

logger = get_logger(__name__)


class HybridOCRApp(QMainWindow):
    """
    Main application window with comprehensive PDF OCR interface.

    Features:
    - File/folder selection with page range
    - Engine selector (auto/docai/donut/trocr/tatr/pix2tex)
    - Threshold configuration
    - Progress tracking with time estimation
    - Result preview with highlighting
    - Export options (TXT/JSON/MD/DOCX/Searchable PDF)
    - Cache integration
    - Error view with thumbnails
    """

    def __init__(self, config_path: Optional[Path] = None):
        super().__init__()

        # Load configuration
        self.config = load_config(config_path) if config_path else load_config()

        # Initialize cache
        cache_dir = Path(self.config.get('app', {}).get('cache_dir', '~/.cache/hybrid-pdf-ocr')).expanduser()
        cache_config = self.config.get('cache', {})
        self.cache_manager = CacheManager(
            cache_dir=cache_dir,
            enabled=cache_config.get('enabled', True),
            max_size_mb=cache_config.get('max_size_mb', 1024),
            ttl_seconds=cache_config.get('ttl', 3600),
        )

        # Thread pool for parallel processing
        self.thread_pool = QThreadPool()
        max_workers = self.config.get('device', {}).get('num_workers', 4)
        self.thread_pool.setMaxThreadCount(max_workers)

        # Processing state
        self.selected_files: List[Path] = []
        self.current_results: Dict[str, any] = {}
        self.failed_pages: List[Tuple[Path, int, str]] = []  # (file, page_num, reason)
        self.processing_start_time: Optional[float] = None
        self.pages_processed = 0
        self.total_pages = 0

        self.init_ui()

    def init_ui(self):
        """Initialize comprehensive UI."""
        self.setWindowTitle("Hybrid PDF OCR System")
        self.setGeometry(100, 100, 1400, 900)

        # Central widget with tab interface
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Title
        title = QLabel("Hybrid PDF OCR System")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("padding: 15px;")
        main_layout.addWidget(title)

        # Tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # PDF OCR Tab
        self.ocr_tab = self.create_ocr_tab()
        self.tabs.addTab(self.ocr_tab, "PDF OCR")

        # Results Tab
        self.results_tab = self.create_results_tab()
        self.tabs.addTab(self.results_tab, "Results")

        # Errors Tab
        self.errors_tab = self.create_errors_tab()
        self.tabs.addTab(self.errors_tab, "Errors")

        # Logs Tab
        self.logs_tab = self.create_logs_tab()
        self.tabs.addTab(self.logs_tab, "Logs")

        # Status bar
        self.statusBar().showMessage("Ready")

    def create_ocr_tab(self) -> QWidget:
        """Create comprehensive PDF OCR tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # File selection section
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()
        file_group.setLayout(file_layout)

        # File list
        self.file_list = QListWidget()
        file_layout.addWidget(self.file_list)

        # File buttons
        file_btn_layout = QHBoxLayout()

        add_files_btn = QPushButton("Add Files")
        add_files_btn.clicked.connect(self.add_files)
        file_btn_layout.addWidget(add_files_btn)

        add_folder_btn = QPushButton("Add Folder")
        add_folder_btn.clicked.connect(self.add_folder)
        file_btn_layout.addWidget(add_folder_btn)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_selected)
        file_btn_layout.addWidget(remove_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_files)
        file_btn_layout.addWidget(clear_btn)

        file_layout.addLayout(file_btn_layout)
        layout.addWidget(file_group)

        # Processing options
        options_group = QGroupBox("Processing Options")
        options_layout = QFormLayout()
        options_group.setLayout(options_layout)

        # Page range
        page_range_layout = QHBoxLayout()
        self.page_start = QSpinBox()
        self.page_start.setMinimum(1)
        self.page_start.setMaximum(9999)
        self.page_start.setValue(1)
        self.page_start.setPrefix("From: ")
        page_range_layout.addWidget(self.page_start)

        self.page_end = QSpinBox()
        self.page_end.setMinimum(1)
        self.page_end.setMaximum(9999)
        self.page_end.setValue(9999)
        self.page_end.setPrefix("To: ")
        page_range_layout.addWidget(self.page_end)

        self.all_pages_check = QCheckBox("All Pages")
        self.all_pages_check.setChecked(True)
        self.all_pages_check.stateChanged.connect(self.toggle_page_range)
        page_range_layout.addWidget(self.all_pages_check)

        options_layout.addRow("Page Range:", page_range_layout)

        # Engine selection
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["auto", "docai", "donut", "trocr", "tatr", "pix2tex"])
        options_layout.addRow("Engine:", self.engine_combo)

        # Thresholds
        self.docai_threshold = QDoubleSpinBox()
        self.docai_threshold.setRange(0.0, 1.0)
        self.docai_threshold.setSingleStep(0.05)
        self.docai_threshold.setValue(self.config.get('thresholds', {}).get('docai_confidence', 0.85))
        options_layout.addRow("DocAI Threshold:", self.docai_threshold)

        self.low_conf_threshold = QDoubleSpinBox()
        self.low_conf_threshold.setRange(0.0, 1.0)
        self.low_conf_threshold.setSingleStep(0.05)
        self.low_conf_threshold.setValue(self.config.get('thresholds', {}).get('low_confidence', 0.5))
        options_layout.addRow("Low Confidence:", self.low_conf_threshold)

        # Ensemble mode
        self.ensemble_check = QCheckBox("Enable Ensemble")
        self.ensemble_check.setChecked(self.config.get('ensemble', {}).get('enabled', True))
        options_layout.addRow("Ensemble Mode:", self.ensemble_check)

        # Cache
        self.cache_check = QCheckBox("Use Cache")
        self.cache_check.setChecked(self.config.get('cache', {}).get('enabled', True))
        options_layout.addRow("Cache:", self.cache_check)

        layout.addWidget(options_group)

        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        progress_group.setLayout(progress_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("Ready to process")
        self.progress_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.progress_label)

        self.time_label = QLabel("Estimated time: --:--")
        self.time_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.time_label)

        layout.addWidget(progress_group)

        # Action buttons
        btn_layout = QHBoxLayout()

        self.process_btn = QPushButton("Process OCR")
        self.process_btn.clicked.connect(self.process_ocr)
        self.process_btn.setEnabled(False)
        self.process_btn.setStyleSheet("font-size: 14px; padding: 10px; font-weight: bold;")
        btn_layout.addWidget(self.process_btn)

        self.reprocess_low_conf_btn = QPushButton("Reprocess Low Confidence Pages")
        self.reprocess_low_conf_btn.clicked.connect(self.reprocess_low_confidence)
        self.reprocess_low_conf_btn.setEnabled(False)
        btn_layout.addWidget(self.reprocess_low_conf_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)

        return tab

    def create_results_tab(self) -> QWidget:
        """Create results preview tab with highlighting."""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # Splitter for file list and preview
        splitter = QSplitter(Qt.Horizontal)

        # Results file list
        self.results_list = QListWidget()
        self.results_list.itemClicked.connect(self.show_result_preview)
        splitter.addWidget(self.results_list)

        # Preview panel
        preview_widget = QWidget()
        preview_layout = QVBoxLayout()
        preview_widget.setLayout(preview_layout)

        # Confidence indicator
        self.confidence_label = QLabel("Average Confidence: --")
        self.confidence_label.setStyleSheet("font-size: 12pt; padding: 5px;")
        preview_layout.addWidget(self.confidence_label)

        # Text preview with highlighting
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        preview_layout.addWidget(self.preview_text)

        # Export options
        export_layout = QHBoxLayout()

        export_label = QLabel("Export as:")
        export_layout.addWidget(export_label)

        export_txt_btn = QPushButton("TXT")
        export_txt_btn.clicked.connect(lambda: self.export_result("txt"))
        export_layout.addWidget(export_txt_btn)

        export_json_btn = QPushButton("JSON")
        export_json_btn.clicked.connect(lambda: self.export_result("json"))
        export_layout.addWidget(export_json_btn)

        export_md_btn = QPushButton("Markdown")
        export_md_btn.clicked.connect(lambda: self.export_result("markdown"))
        export_layout.addWidget(export_md_btn)

        export_docx_btn = QPushButton("DOCX")
        export_docx_btn.clicked.connect(lambda: self.export_result("docx"))
        export_layout.addWidget(export_docx_btn)

        export_pdf_btn = QPushButton("Searchable PDF")
        export_pdf_btn.clicked.connect(lambda: self.export_result("searchable_pdf"))
        export_layout.addWidget(export_pdf_btn)

        export_layout.addStretch()
        preview_layout.addLayout(export_layout)

        splitter.addWidget(preview_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)

        return tab

    def create_errors_tab(self) -> QWidget:
        """Create error view with thumbnails and failure reasons."""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)

        # Error count label
        self.error_count_label = QLabel("No errors")
        self.error_count_label.setStyleSheet("font-size: 12pt; font-weight: bold; padding: 10px;")
        layout.addWidget(self.error_count_label)

        # Scroll area for error items
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.error_container = QWidget()
        self.error_layout = QVBoxLayout()
        self.error_container.setLayout(self.error_layout)
        scroll.setWidget(self.error_container)

        layout.addWidget(scroll)

        # Clear errors button
        clear_errors_btn = QPushButton("Clear Errors")
        clear_errors_btn.clicked.connect(self.clear_errors)
        layout.addWidget(clear_errors_btn)

        return tab

    def create_logs_tab(self) -> QWidget:
        """Create logs tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("font-family: monospace; font-size: 10pt;")
        layout.addWidget(self.log_output)

        # Clear logs button
        clear_logs_btn = QPushButton("Clear Logs")
        clear_logs_btn.clicked.connect(self.clear_logs)
        layout.addWidget(clear_logs_btn)

        return tab

    def toggle_page_range(self, state):
        """Toggle page range inputs."""
        enabled = state != Qt.Checked
        self.page_start.setEnabled(enabled)
        self.page_end.setEnabled(enabled)

    def add_files(self):
        """Add PDF/image files."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select PDF or Image Files",
            "",
            "PDF Files (*.pdf);;Image Files (*.png *.jpg *.jpeg *.tiff);;All Files (*.*)",
        )

        if files:
            for file_path in files:
                path = Path(file_path)
                if path not in self.selected_files:
                    self.selected_files.append(path)
                    self.file_list.addItem(file_path)

            self.process_btn.setEnabled(len(self.selected_files) > 0)
            self.log(f"Added {len(files)} file(s)")

    def add_folder(self):
        """Add all PDFs from a folder."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder",
            "",
        )

        if folder:
            folder_path = Path(folder)
            pdf_files = list(folder_path.glob("*.pdf"))

            for file_path in pdf_files:
                if file_path not in self.selected_files:
                    self.selected_files.append(file_path)
                    self.file_list.addItem(str(file_path))

            self.process_btn.setEnabled(len(self.selected_files) > 0)
            self.log(f"Added {len(pdf_files)} PDF(s) from folder")

    def remove_selected(self):
        """Remove selected files from list."""
        for item in self.file_list.selectedItems():
            row = self.file_list.row(item)
            self.file_list.takeItem(row)
            if row < len(self.selected_files):
                del self.selected_files[row]

        self.process_btn.setEnabled(len(self.selected_files) > 0)

    def clear_files(self):
        """Clear all files."""
        self.file_list.clear()
        self.selected_files.clear()
        self.process_btn.setEnabled(False)
        self.log("Cleared all files")

    def process_ocr(self):
        """Process OCR on selected files."""
        if not self.selected_files:
            QMessageBox.warning(self, "No Files", "Please select files first")
            return

        # Get processing options
        engine = self.engine_combo.currentText()
        page_range = None if self.all_pages_check.isChecked() else (
            self.page_start.value(),
            self.page_end.value()
        )

        # Prepare for processing
        self.current_results.clear()
        self.failed_pages.clear()
        self.pages_processed = 0
        self.processing_start_time = time.time()

        # Update UI
        self.process_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log(f"Starting OCR processing with engine: {engine}")

        # TODO: Implement actual OCR processing with workers
        # This will be implemented in workers.py

        self.log("OCR processing initiated")
        self.statusBar().showMessage("Processing...")

    def reprocess_low_confidence(self):
        """Reprocess pages with low confidence."""
        # TODO: Implement reprocessing of low confidence pages
        self.log("Reprocessing low confidence pages...")

    def cancel_processing(self):
        """Cancel ongoing processing."""
        # TODO: Implement cancellation
        self.log("Processing cancelled")
        self.cancel_btn.setEnabled(False)
        self.process_btn.setEnabled(True)
        self.statusBar().showMessage("Cancelled")

    def show_result_preview(self, item: QListWidgetItem):
        """Show preview of selected result."""
        file_path = item.text()

        if file_path in self.current_results:
            result = self.current_results[file_path]

            # Update confidence label
            avg_conf = result.get('average_confidence', 0.0)
            conf_color = self.get_confidence_color(avg_conf)
            self.confidence_label.setText(f"Average Confidence: {avg_conf:.2%}")
            self.confidence_label.setStyleSheet(f"font-size: 12pt; padding: 5px; color: {conf_color};")

            # Show text with highlighting
            self.preview_text.clear()
            text = result.get('text', '')

            # TODO: Add highlighting for low confidence blocks
            self.preview_text.setPlainText(text)

    def get_confidence_color(self, confidence: float) -> str:
        """Get color for confidence level."""
        if confidence >= 0.9:
            return "green"
        elif confidence >= 0.7:
            return "orange"
        else:
            return "red"

    def export_result(self, format: str):
        """Export current result in specified format."""
        # TODO: Implement export functionality
        self.log(f"Exporting result as {format.upper()}...")

    def add_error_item(self, file_path: Path, page_num: int, reason: str, thumbnail: Optional[QPixmap] = None):
        """Add error item to error view."""
        error_widget = QWidget()
        error_layout = QHBoxLayout()
        error_widget.setLayout(error_layout)

        # Thumbnail
        if thumbnail:
            thumb_label = QLabel()
            thumb_label.setPixmap(thumbnail.scaled(100, 100, Qt.KeepAspectRatio))
            error_layout.addWidget(thumb_label)

        # Error info
        info_layout = QVBoxLayout()
        file_label = QLabel(f"<b>{file_path.name}</b> - Page {page_num}")
        info_layout.addWidget(file_label)

        reason_label = QLabel(reason)
        reason_label.setWordWrap(True)
        info_layout.addWidget(reason_label)

        error_layout.addLayout(info_layout)
        error_layout.addStretch()

        # Add to error layout
        self.error_layout.addWidget(error_widget)

        # Update error count
        self.failed_pages.append((file_path, page_num, reason))
        self.error_count_label.setText(f"{len(self.failed_pages)} error(s)")

    def clear_errors(self):
        """Clear all errors."""
        while self.error_layout.count():
            child = self.error_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.failed_pages.clear()
        self.error_count_label.setText("No errors")
        self.log("Errors cleared")

    def clear_logs(self):
        """Clear log output."""
        self.log_output.clear()

    def log(self, message: str):
        """
        Add message to log output.

        Args:
            message: Log message
        """
        timestamp = time.strftime("%H:%M:%S")
        self.log_output.append(f"[{timestamp}] {message}")
        logger.info(message)

    def update_progress(self, pages_done: int, total_pages: int):
        """
        Update progress bar and time estimate.

        Args:
            pages_done: Number of pages completed
            total_pages: Total number of pages
        """
        self.pages_processed = pages_done
        self.total_pages = total_pages

        if total_pages > 0:
            percentage = int((pages_done / total_pages) * 100)
            self.progress_bar.setValue(percentage)

            # Calculate time remaining
            if self.processing_start_time and pages_done > 0:
                elapsed = time.time() - self.processing_start_time
                avg_time_per_page = elapsed / pages_done
                remaining_pages = total_pages - pages_done
                estimated_remaining = avg_time_per_page * remaining_pages

                minutes = int(estimated_remaining / 60)
                seconds = int(estimated_remaining % 60)

                self.time_label.setText(f"Estimated time: {minutes:02d}:{seconds:02d}")
                self.progress_label.setText(f"Processing page {pages_done}/{total_pages}")

    def closeEvent(self, event):
        """Handle window close event."""
        # Clean up thread pool
        self.thread_pool.waitForDone()
        event.accept()


def main():
    """Main entry point."""
    # Setup logging
    setup_logging(log_level='INFO')

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Hybrid PDF OCR")
    app.setStyle("Fusion")  # Modern cross-platform style

    # Create and show main window
    config_path = Path("configs/app.yaml") if Path("configs/app.yaml").exists() else None
    window = HybridOCRApp(config_path)
    window.show()

    # Run event loop
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
