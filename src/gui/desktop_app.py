"""
PyQt5 Desktop Application for Hybrid PDF OCR.

Comprehensive GUI with PDF OCR tab, file/folder selection, page range,
engine selection, threshold settings, progress tracking, result preview,
export options, and error view with thumbnails.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure the top-level `src` directory is on sys.path when this file
# is executed directly (e.g., `python src/gui/desktop_app.py`).
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1].parent  # .../src
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)

from cache.store import CacheManager
from gui.workers import PDFOCRWorker
from util.logging import get_logger, setup_logging
from util.config import load_config

logger = get_logger(__name__)

_instance_lock_handle = None

# Marker file to detect recent app exits and avoid
# rapid macOS relaunch loops from a bundled .app.
_recent_exit_marker = Path.home() / ".hybrid_ocr_recent_exit"
_recent_exit_timeout_seconds = 5


def acquire_single_instance_lock() -> bool:
    """
    Acquire a cross-platform single-instance lock.

    This prevents multiple GUI processes from running simultaneously,
    which can manifest as windows opening repeatedly on some systems
    or with certain launchers / bundlers.
    """
    global _instance_lock_handle

    lock_file = Path.home() / ".hybrid_ocr_gui.lock"
    try:
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_file.open("w")
    except OSError:
        # If we cannot create the lock file, do not block startup
        return True

    try:
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                handle.close()
                return False
        else:
            import fcntl

            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                handle.close()
                return False
    except Exception:
        # On any unexpected error, fall back to allowing startup
        handle.close()
        return True

    _instance_lock_handle = handle
    return True


class CollapsibleBox(QWidget):
    """A collapsible box widget with toggle functionality."""

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.toggle_button = QPushButton(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        self.toggle_button.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 8px;
                border: 1px solid #555;
                background-color: #3c3c3c;
                color: white;
                font-weight: bold;
                border-radius: 3px;
            }
            QPushButton:checked {
                background-color: #4a4a4a;
            }
            QPushButton:hover {
                background-color: #505050;
            }
        """)
        self.toggle_button.clicked.connect(self.on_toggle)

        self.content_area = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_area.setLayout(self.content_layout)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.toggle_button)
        main_layout.addWidget(self.content_area)
        self.setLayout(main_layout)

        self.update_button_text()

    def on_toggle(self):
        """Toggle content visibility."""
        checked = self.toggle_button.isChecked()
        self.content_area.setVisible(checked)
        self.update_button_text()

    def update_button_text(self):
        """Update button text with arrow indicator."""
        title = self.toggle_button.text().replace("▼ ", "").replace("▶ ", "")
        if self.toggle_button.isChecked():
            self.toggle_button.setText(f"▼ {title}")
        else:
            self.toggle_button.setText(f"▶ {title}")

    def setTitle(self, title):
        """Set the title of the collapsible box."""
        self.toggle_button.setText(title)
        self.update_button_text()

    def addWidget(self, widget):
        """Add a widget to the content area."""
        self.content_layout.addWidget(widget)

    def addLayout(self, layout):
        """Add a layout to the content area."""
        self.content_layout.addLayout(layout)


class DropZoneWidget(QWidget):
    """Widget that accepts drag and drop for files."""

    files_dropped = pyqtSignal(list)  # Emits list of file paths

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Drop zone label
        self.label = QLabel("📁 Drag & Drop Files Here\n\nSupported: PDF, PNG, JPG, JPEG, TIFF")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("""
            QLabel {
                border: 2px dashed #555;
                border-radius: 10px;
                padding: 30px 40px;
                background-color: #2b2b2b;
                font-size: 14px;
                font-weight: bold;
                color: white;
                line-height: 1.8;
            }
        """)
        layout.addWidget(self.label)

        self.setMinimumHeight(150)
        self.setMaximumHeight(180)

    def dragEnterEvent(self, event):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.label.setStyleSheet("""
                QLabel {
                    border: 2px dashed #4CAF50;
                    border-radius: 10px;
                    padding: 30px 40px;
                    background-color: #e8f5e9;
                    font-size: 14px;
                    font-weight: bold;
                    color: #2e7d32;
                    line-height: 1.8;
                }
            """)

    def dragLeaveEvent(self, event):
        """Handle drag leave event."""
        self.label.setStyleSheet("""
            QLabel {
                border: 2px dashed #555;
                border-radius: 10px;
                padding: 30px 40px;
                background-color: #2b2b2b;
                font-size: 14px;
                font-weight: bold;
                color: white;
                line-height: 1.8;
            }
        """)

    def dropEvent(self, event):
        """Handle drop event."""
        files = []
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if Path(file_path).suffix.lower() in ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif']:
                files.append(file_path)

        if files:
            self.files_dropped.emit(files)

        # Reset style
        self.label.setStyleSheet("""
            QLabel {
                border: 2px dashed #555;
                border-radius: 10px;
                padding: 30px 40px;
                background-color: #2b2b2b;
                font-size: 14px;
                font-weight: bold;
                color: white;
                line-height: 1.8;
            }
        """)

        event.acceptProposedAction()


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
        self.files_processed = 0
        self.total_files = 0
        self.active_workers: List[PDFOCRWorker] = []
        self.output_directory: Optional[Path] = None
        self.is_processing = False

        # Performance statistics
        self.total_processing_time = 0.0
        self.page_processing_times: List[float] = []
        self.max_memory_usage = 0.0
        self.dpi_downshift_count = 0
        self.total_text_blocks = 0
        self.memory_samples: List[float] = []

        # Batch queue tracking
        self.file_queue_status: Dict[str, Dict] = {}  # file_path -> {status, progress, pages, confidence, worker}

        # Low-confidence page tracking for reprocessing
        self.low_confidence_pages: Dict[str, List[int]] = {}  # file_path -> [page_numbers]
        self.page_confidences: Dict[str, Dict[int, float]] = {}  # file_path -> {page_num: confidence}

        # Settings file path
        self.settings_file = Path.home() / '.hybrid_ocr_settings.json'

        self.init_ui()
        self.load_settings()

        # Timer for updating elapsed time during processing
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_elapsed_time)
        self.update_timer.setInterval(1000)  # Update every second

    def init_ui(self):
        """Initialize comprehensive UI."""
        self.setWindowTitle("Hybrid PDF OCR System")
        self.setGeometry(100, 100, 1600, 1000)

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
        main_layout = QVBoxLayout()
        tab.setLayout(main_layout)

        # Create scroll area for all content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        layout = QVBoxLayout()
        scroll_content.setLayout(layout)

        # File selection section (collapsible)
        file_group = CollapsibleBox("File Selection")
        file_layout = QVBoxLayout()

        # Drag & Drop zone
        self.drop_zone = DropZoneWidget()
        self.drop_zone.files_dropped.connect(self.handle_dropped_files)
        file_layout.addWidget(self.drop_zone)

        # File buttons
        file_btn_layout = QHBoxLayout()

        add_files_btn = QPushButton("📄 Add Files")
        add_files_btn.clicked.connect(self.add_files)
        add_files_btn.setToolTip("Select one or more files (PDF, PNG, JPG, TIFF)")
        file_btn_layout.addWidget(add_files_btn)

        add_folder_btn = QPushButton("📁 Add Folder")
        add_folder_btn.clicked.connect(self.add_folder)
        add_folder_btn.setToolTip("Select a folder to add all supported files")
        file_btn_layout.addWidget(add_folder_btn)

        file_layout.addLayout(file_btn_layout)

        # Batch Processing Queue table (integrated with file selection)
        queue_label = QLabel("Processing Queue:")
        queue_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        file_layout.addWidget(queue_label)

        # Queue table
        self.queue_table = QTableWidget()
        self.queue_table.setColumnCount(5)
        self.queue_table.setHorizontalHeaderLabels([
            "File", "Status", "Progress", "Pages", "Confidence"
        ])
        self.queue_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.queue_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.queue_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.queue_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.queue_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.queue_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.queue_table.setSelectionMode(QTableWidget.SingleSelection)
        self.queue_table.setMinimumHeight(200)
        self.queue_table.setMaximumHeight(300)
        self.queue_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #555;
                gridline-color: #444;
                background-color: #2b2b2b;
            }
            QTableWidget::item {
                padding: 5px;
                color: white;
            }
            QTableWidget::item:selected {
                background-color: #4CAF50;
                color: white;
            }
            QHeaderView::section {
                background-color: #3c3c3c;
                color: white;
                padding: 5px;
                border: 1px solid #555;
                font-weight: bold;
            }
        """)
        file_layout.addWidget(self.queue_table)

        # Queue controls
        queue_controls = QHBoxLayout()

        self.move_up_btn = QPushButton("⬆ Move Up")
        self.move_up_btn.clicked.connect(self.move_queue_item_up)
        self.move_up_btn.setEnabled(False)
        self.move_up_btn.setToolTip("Move selected file up in processing queue")
        queue_controls.addWidget(self.move_up_btn)

        self.move_down_btn = QPushButton("⬇ Move Down")
        self.move_down_btn.clicked.connect(self.move_queue_item_down)
        self.move_down_btn.setEnabled(False)
        self.move_down_btn.setToolTip("Move selected file down in processing queue")
        queue_controls.addWidget(self.move_down_btn)

        self.remove_queue_btn = QPushButton("✖ Remove Selected")
        self.remove_queue_btn.clicked.connect(self.remove_queue_item)
        self.remove_queue_btn.setEnabled(False)
        self.remove_queue_btn.setToolTip("Remove selected file from queue")
        queue_controls.addWidget(self.remove_queue_btn)

        queue_controls.addStretch()
        file_layout.addLayout(queue_controls)

        # Connect queue table selection
        self.queue_table.itemSelectionChanged.connect(self.on_queue_selection_changed)

        # Add file_layout to collapsible box
        file_group.addLayout(file_layout)
        layout.addWidget(file_group)

        # Processing options (collapsible)
        options_group = CollapsibleBox("Processing Options")
        options_group.toggle_button.setChecked(False)
        options_group.content_area.setVisible(False)
        options_layout = QFormLayout()

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

        # Engine selection (DocAI only)
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["docai"])
        self.engine_combo.setToolTip("Google Cloud Document AI (OCR + Form Parser + Layout Parser)")
        options_layout.addRow("Engine:", self.engine_combo)

        # Confidence threshold
        self.low_conf_threshold = QDoubleSpinBox()
        self.low_conf_threshold.setRange(0.0, 1.0)
        self.low_conf_threshold.setSingleStep(0.05)
        self.low_conf_threshold.setValue(self.config.get('thresholds', {}).get('low_confidence', 0.5))
        self.low_conf_threshold.setToolTip("Flag pages with confidence below this threshold")
        options_layout.addRow("Low Confidence:", self.low_conf_threshold)

        # Cache
        self.cache_check = QCheckBox("Use Cache")
        self.cache_check.setChecked(self.config.get('cache', {}).get('enabled', True))
        options_layout.addRow("Cache:", self.cache_check)

        options_group.addLayout(options_layout)
        layout.addWidget(options_group)

        # Advanced options (collapsible, initially collapsed)
        advanced_group = CollapsibleBox("Advanced Options")
        advanced_group.toggle_button.setChecked(False)
        advanced_group.content_area.setVisible(False)
        advanced_layout = QFormLayout()

        # Preprocessing options
        preproc_layout = QHBoxLayout()

        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(self.config.get('pdf', {}).get('dpi', 300))
        self.dpi_spin.setSuffix(" DPI")
        self.dpi_spin.setToolTip("Higher DPI = better quality but slower")
        preproc_layout.addWidget(QLabel("DPI:"))
        preproc_layout.addWidget(self.dpi_spin)

        self.auto_rotate_check = QCheckBox("Auto-rotate")
        self.auto_rotate_check.setChecked(self.config.get('preprocessing', {}).get('auto_rotate', True))
        preproc_layout.addWidget(self.auto_rotate_check)

        self.deskew_check = QCheckBox("Deskew")
        self.deskew_check.setChecked(self.config.get('preprocessing', {}).get('deskew', True))
        preproc_layout.addWidget(self.deskew_check)

        self.enhance_check = QCheckBox("Enhance")
        self.enhance_check.setChecked(self.config.get('preprocessing', {}).get('enhance_contrast', True))
        self.enhance_check.setToolTip("Enhance image contrast and sharpness")
        preproc_layout.addWidget(self.enhance_check)

        preproc_layout.addStretch()
        advanced_layout.addRow("Preprocessing:", preproc_layout)

        # Postprocessing options
        postproc_layout = QHBoxLayout()

        self.normalize_whitespace_check = QCheckBox("Normalize Whitespace")
        self.normalize_whitespace_check.setChecked(self.config.get('postprocessing', {}).get('normalize_whitespace', True))
        postproc_layout.addWidget(self.normalize_whitespace_check)

        self.fix_errors_check = QCheckBox("Fix Common Errors")
        self.fix_errors_check.setChecked(self.config.get('postprocessing', {}).get('fix_common_errors', True))
        postproc_layout.addWidget(self.fix_errors_check)

        postproc_layout.addStretch()
        advanced_layout.addRow("Postprocessing:", postproc_layout)

        # Performance options
        perf_layout = QHBoxLayout()

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 16)
        self.workers_spin.setValue(self.config.get('device', {}).get('num_workers', 4))
        self.workers_spin.setToolTip("Number of parallel workers")
        perf_layout.addWidget(QLabel("Workers:"))
        perf_layout.addWidget(self.workers_spin)

        perf_layout.addStretch()
        advanced_layout.addRow("Performance:", perf_layout)

        advanced_group.addLayout(advanced_layout)
        layout.addWidget(advanced_group)

        # Hardware Acceleration section (collapsible, initially collapsed)
        hardware_group = CollapsibleBox("Hardware Acceleration")
        hardware_group.toggle_button.setChecked(False)
        hardware_group.content_area.setVisible(False)
        hardware_layout = QVBoxLayout()

        # GPU Info display
        from util.device import get_device_manager
        device_manager = get_device_manager()

        gpu_info_layout = QHBoxLayout()
        gpu_label = QLabel("GPU:")
        gpu_label.setStyleSheet("font-weight: bold;")
        gpu_info_layout.addWidget(gpu_label)

        gpu_name = device_manager.get_gpu_name()
        is_mps = device_manager.is_mps()

        gpu_status = QLabel(f"{'✓' if is_mps else '✗'} {gpu_name}")
        if is_mps:
            gpu_status.setStyleSheet("color: green; font-weight: bold;")
        else:
            gpu_status.setStyleSheet("color: gray;")
        gpu_info_layout.addWidget(gpu_status)
        gpu_info_layout.addStretch()
        hardware_layout.addLayout(gpu_info_layout)

        # Metal acceleration status
        metal_layout = QHBoxLayout()
        metal_label = QLabel("Metal:")
        metal_label.setStyleSheet("font-weight: bold;")
        metal_layout.addWidget(metal_label)

        metal_status = QLabel("Enabled" if is_mps else "Disabled (CPU mode)")
        metal_status.setStyleSheet("color: green;" if is_mps else "color: gray;")
        metal_layout.addWidget(metal_status)
        metal_layout.addStretch()
        hardware_layout.addLayout(metal_layout)

        # Parallel page processing
        parallel_layout = QHBoxLayout()
        parallel_label = QLabel("Parallel Pages:")
        parallel_label.setStyleSheet("font-weight: bold;")
        parallel_layout.addWidget(parallel_label)

        self.parallel_pages_spin = QSpinBox()
        self.parallel_pages_spin.setRange(1, 4)
        self.parallel_pages_spin.setValue(1)
        self.parallel_pages_spin.setToolTip("Number of pages to process simultaneously (1-4)\nHigher values use more memory but are faster")
        parallel_layout.addWidget(self.parallel_pages_spin)

        parallel_help = QLabel("(1-4 pages simultaneously)")
        parallel_help.setStyleSheet("color: gray; font-style: italic;")
        parallel_layout.addWidget(parallel_help)
        parallel_layout.addStretch()
        hardware_layout.addLayout(parallel_layout)

        hardware_group.addLayout(hardware_layout)
        layout.addWidget(hardware_group)

        # System Options section (collapsible, initially collapsed)
        system_group = CollapsibleBox("System Options")
        system_group.toggle_button.setChecked(False)
        system_group.content_area.setVisible(False)
        system_layout = QVBoxLayout()

        # Language-specific spell correction
        spell_layout = QHBoxLayout()
        spell_label = QLabel("Spell Correction:")
        spell_label.setStyleSheet("font-weight: bold;")
        spell_layout.addWidget(spell_label)

        self.spell_correction_combo = QComboBox()
        self.spell_correction_combo.addItems([
            "Disabled",
            "English",
            "Korean",
            "Auto-detect"
        ])
        self.spell_correction_combo.setCurrentText("Disabled")
        self.spell_correction_combo.setToolTip(
            "Apply language-specific spell correction to OCR results\n"
            "• Disabled: No correction\n"
            "• English: English spell checker\n"
            "• Korean: Korean spell checker\n"
            "• Auto-detect: Detect language and apply appropriate correction"
        )
        spell_layout.addWidget(self.spell_correction_combo)
        spell_layout.addStretch()
        system_layout.addLayout(spell_layout)

        # Confidence-based reprocessing
        reprocess_layout = QVBoxLayout()

        self.enable_reprocessing_check = QCheckBox("Enable Low-Confidence Reprocessing")
        self.enable_reprocessing_check.setChecked(False)
        self.enable_reprocessing_check.setToolTip(
            "Automatically reprocess pages with low OCR confidence\n"
            "Uses alternative processing methods to improve quality"
        )
        self.enable_reprocessing_check.stateChanged.connect(self.on_reprocessing_toggled)
        reprocess_layout.addWidget(self.enable_reprocessing_check)

        # Confidence threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addSpacing(20)  # Indent

        threshold_label = QLabel("Confidence Threshold:")
        threshold_layout.addWidget(threshold_label)

        self.confidence_threshold_spin = QSpinBox()
        self.confidence_threshold_spin.setMinimum(50)
        self.confidence_threshold_spin.setMaximum(95)
        self.confidence_threshold_spin.setValue(70)
        self.confidence_threshold_spin.setSuffix("%")
        self.confidence_threshold_spin.setEnabled(False)
        self.confidence_threshold_spin.setToolTip(
            "Pages with confidence below this threshold will be reprocessed\n"
            "Recommended: 70% (balance between quality and speed)"
        )
        threshold_layout.addWidget(self.confidence_threshold_spin)

        threshold_help = QLabel("(Pages below this will be reprocessed)")
        threshold_help.setStyleSheet("color: gray; font-style: italic;")
        threshold_layout.addWidget(threshold_help)
        threshold_layout.addStretch()
        reprocess_layout.addLayout(threshold_layout)

        # Reprocessing method
        method_layout = QHBoxLayout()
        method_layout.addSpacing(20)  # Indent

        method_label = QLabel("Reprocessing Method:")
        method_layout.addWidget(method_label)

        self.reprocess_method_combo = QComboBox()
        self.reprocess_method_combo.addItems([
            "Alternative Processor",
            "Higher DPI",
            "Both (Processor + DPI)"
        ])
        self.reprocess_method_combo.setCurrentIndex(0)
        self.reprocess_method_combo.setEnabled(False)
        self.reprocess_method_combo.setToolTip(
            "Method to use for reprocessing low-confidence pages:\n"
            "• Alternative Processor: Try different DocAI processor\n"
            "• Higher DPI: Increase DPI for that page (300→400)\n"
            "• Both: Try alternative processor first, then higher DPI if still low"
        )
        method_layout.addWidget(self.reprocess_method_combo)
        method_layout.addStretch()
        reprocess_layout.addLayout(method_layout)

        system_layout.addLayout(reprocess_layout)

        # Auto DPI downshift
        dpi_downshift_layout = QHBoxLayout()

        self.auto_dpi_downshift_check = QCheckBox("Auto DPI Downshift")
        self.auto_dpi_downshift_check.setChecked(True)
        self.auto_dpi_downshift_check.setToolTip(
            "Automatically reduce DPI when memory is insufficient\n"
            "Helps prevent out-of-memory errors on large files\n"
            "DPI: 300 → 200 → 150 (if needed)"
        )
        dpi_downshift_layout.addWidget(self.auto_dpi_downshift_check)

        dpi_help = QLabel("(Prevents out-of-memory errors)")
        dpi_help.setStyleSheet("color: gray; font-style: italic;")
        dpi_downshift_layout.addWidget(dpi_help)
        dpi_downshift_layout.addStretch()
        system_layout.addLayout(dpi_downshift_layout)

        # Memory threshold for DPI downshift
        mem_threshold_layout = QHBoxLayout()
        mem_label = QLabel("Memory Threshold:")
        mem_label.setStyleSheet("font-weight: bold;")
        mem_threshold_layout.addWidget(mem_label)

        self.memory_threshold_spin = QSpinBox()
        self.memory_threshold_spin.setRange(50, 95)
        self.memory_threshold_spin.setSingleStep(5)
        self.memory_threshold_spin.setValue(85)
        self.memory_threshold_spin.setSuffix("%")
        self.memory_threshold_spin.setToolTip(
            "Trigger DPI downshift when memory usage exceeds this percentage"
        )
        mem_threshold_layout.addWidget(self.memory_threshold_spin)

        mem_help2 = QLabel("(Trigger downshift at this memory %)")
        mem_help2.setStyleSheet("color: gray; font-style: italic;")
        mem_threshold_layout.addWidget(mem_help2)
        mem_threshold_layout.addStretch()
        system_layout.addLayout(mem_threshold_layout)

        system_group.addLayout(system_layout)
        layout.addWidget(system_group)

        # Output options (collapsible)
        output_group = CollapsibleBox("Output Options")
        output_layout = QVBoxLayout()

        # Output directory selection
        output_dir_layout = QHBoxLayout()

        # Output directory label/input
        dir_input_layout = QVBoxLayout()
        dir_label = QLabel("Output Directory:")
        dir_input_layout.addWidget(dir_label)

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Not set (will save next to original files)")
        self.output_dir_edit.setReadOnly(True)
        dir_input_layout.addWidget(self.output_dir_edit)

        output_dir_layout.addLayout(dir_input_layout, stretch=3)

        # Buttons
        button_layout = QVBoxLayout()

        select_output_btn = QPushButton("Browse...")
        select_output_btn.clicked.connect(self.select_output_directory)
        button_layout.addWidget(select_output_btn)

        self.open_output_btn = QPushButton("Open Folder")
        self.open_output_btn.clicked.connect(self.open_output_directory)
        self.open_output_btn.setEnabled(False)
        button_layout.addWidget(self.open_output_btn)

        output_dir_layout.addLayout(button_layout, stretch=1)

        output_layout.addLayout(output_dir_layout)

        # Output formats
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Export Formats:"))

        self.export_txt_check = QCheckBox("TXT")
        self.export_txt_check.setChecked(True)
        format_layout.addWidget(self.export_txt_check)

        self.export_json_check = QCheckBox("JSON")
        self.export_json_check.setChecked(False)
        format_layout.addWidget(self.export_json_check)

        self.export_md_check = QCheckBox("Markdown")
        self.export_md_check.setChecked(False)
        format_layout.addWidget(self.export_md_check)

        self.export_searchable_pdf_check = QCheckBox("Searchable PDF")
        self.export_searchable_pdf_check.setChecked(True)
        format_layout.addWidget(self.export_searchable_pdf_check)

        format_layout.addStretch()
        output_layout.addLayout(format_layout)

        # Table extraction options
        table_layout = QVBoxLayout()

        self.extract_tables_check = QCheckBox("Extract Tables to CSV")
        self.extract_tables_check.setChecked(False)
        self.extract_tables_check.setToolTip(
            "Extract detected tables from documents and save as CSV files\n"
            "Each table will be saved as a separate CSV file\n"
            "Requires Form Parser processor"
        )
        self.extract_tables_check.stateChanged.connect(self.on_table_extraction_toggled)
        table_layout.addWidget(self.extract_tables_check)

        # Table extraction settings (indented)
        table_settings_layout = QHBoxLayout()
        table_settings_layout.addSpacing(20)  # Indent

        table_settings_label = QLabel("Min Rows:")
        table_settings_layout.addWidget(table_settings_label)

        self.table_min_rows_spin = QSpinBox()
        self.table_min_rows_spin.setMinimum(2)
        self.table_min_rows_spin.setMaximum(100)
        self.table_min_rows_spin.setValue(3)
        self.table_min_rows_spin.setEnabled(False)
        self.table_min_rows_spin.setToolTip(
            "Minimum number of rows to consider as a table\n"
            "Tables with fewer rows will be ignored"
        )
        table_settings_layout.addWidget(self.table_min_rows_spin)

        table_settings_label2 = QLabel("Min Cols:")
        table_settings_layout.addWidget(table_settings_label2)

        self.table_min_cols_spin = QSpinBox()
        self.table_min_cols_spin.setMinimum(2)
        self.table_min_cols_spin.setMaximum(50)
        self.table_min_cols_spin.setValue(2)
        self.table_min_cols_spin.setEnabled(False)
        self.table_min_cols_spin.setToolTip(
            "Minimum number of columns to consider as a table\n"
            "Tables with fewer columns will be ignored"
        )
        table_settings_layout.addWidget(self.table_min_cols_spin)

        table_settings_layout.addStretch()
        table_layout.addLayout(table_settings_layout)

        output_layout.addLayout(table_layout)

        output_group.addLayout(output_layout)
        layout.addWidget(output_group)

        # Progress section - Status Display
        progress_group = QGroupBox("Status")
        progress_layout = QVBoxLayout()
        progress_group.setLayout(progress_layout)

        # Progress bar with animation
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #555;
                border-radius: 5px;
                text-align: center;
                background-color: #2b2b2b;
                height: 30px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                  stop:0 #4CAF50, stop:1 #45a049);
                border-radius: 3px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)

        # Status text with file count and percentage
        self.status_label = QLabel("Status: Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-weight: bold; font-size: 13px; padding: 5px;")
        progress_layout.addWidget(self.status_label)

        # Detailed progress text (files processed)
        self.progress_label = QLabel("Files: 0 / 0 | Pages: 0 / 0")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("color: #888; font-size: 12px;")
        progress_layout.addWidget(self.progress_label)

        # Time information layout
        time_layout = QHBoxLayout()

        # Elapsed time
        self.elapsed_label = QLabel("Elapsed: 00:00:00")
        self.elapsed_label.setAlignment(Qt.AlignCenter)
        self.elapsed_label.setStyleSheet("color: #888; font-size: 11px;")
        time_layout.addWidget(self.elapsed_label)

        # Estimated remaining time
        self.time_label = QLabel("Remaining: --:--:--")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("color: #888; font-size: 11px;")
        time_layout.addWidget(self.time_label)

        progress_layout.addLayout(time_layout)

        # Error message display
        self.error_label = QLabel("")
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.setStyleSheet("color: #ff6b6b; font-weight: bold; padding: 5px;")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        progress_layout.addWidget(self.error_label)

        layout.addWidget(progress_group)

        # Performance Statistics section (collapsible, initially open)
        stats_group = CollapsibleBox("Performance Statistics")
        stats_layout = QFormLayout()

        # Total processing time
        self.stats_total_time = QLabel("--:--:--")
        self.stats_total_time.setStyleSheet("color: #4CAF50;")
        stats_layout.addRow("Total Processing Time:", self.stats_total_time)

        # Average page time
        self.stats_avg_page_time = QLabel("-- sec/page")
        stats_layout.addRow("Average Page Time:", self.stats_avg_page_time)

        # Maximum memory usage
        self.stats_max_memory = QLabel("-- MB")
        stats_layout.addRow("Peak Memory Usage:", self.stats_max_memory)

        # DPI downshift count
        self.stats_dpi_downshift = QLabel("0 times")
        stats_layout.addRow("DPI Downshifts:", self.stats_dpi_downshift)

        # Text blocks detected
        self.stats_text_blocks = QLabel("0 blocks")
        stats_layout.addRow("Text Blocks Detected:", self.stats_text_blocks)

        # Processing speed
        self.stats_processing_speed = QLabel("-- pages/min")
        stats_layout.addRow("Processing Speed:", self.stats_processing_speed)

        stats_group.addLayout(stats_layout)
        layout.addWidget(stats_group)

        # Action buttons - Execution Control
        btn_layout = QHBoxLayout()

        # Clear files button
        clear_files_btn = QPushButton("🗑️ Clear Files")
        clear_files_btn.clicked.connect(self.clear_all_files)
        clear_files_btn.setToolTip("Remove all files and reset")
        clear_files_btn.setStyleSheet("padding: 10px;")
        btn_layout.addWidget(clear_files_btn)

        # Cancel button (processing only)
        self.cancel_btn = QPushButton("✖ Cancel")
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setToolTip("Cancel current processing")
        self.cancel_btn.setStyleSheet("padding: 10px; color: #ff6b6b;")
        btn_layout.addWidget(self.cancel_btn)

        # Run OCR button
        self.process_btn = QPushButton("▶ Run OCR")
        self.process_btn.clicked.connect(self.process_ocr)
        self.process_btn.setEnabled(False)
        self.process_btn.setStyleSheet("font-size: 14px; padding: 10px; font-weight: bold; background-color: #4CAF50; color: white;")
        self.process_btn.setToolTip("Start OCR processing")
        btn_layout.addWidget(self.process_btn)

        btn_layout.addStretch()

        # Reprocess low confidence
        self.reprocess_low_conf_btn = QPushButton("🔄 Reprocess Low Confidence")
        self.reprocess_low_conf_btn.clicked.connect(self.reprocess_low_confidence)
        self.reprocess_low_conf_btn.setEnabled(False)
        self.reprocess_low_conf_btn.setToolTip("Reprocess pages with low confidence scores")
        btn_layout.addWidget(self.reprocess_low_conf_btn)

        # Save settings
        save_settings_btn = QPushButton("💾 Save Settings")
        save_settings_btn.clicked.connect(self.save_settings)
        save_settings_btn.setToolTip("Save current settings for next session")
        btn_layout.addWidget(save_settings_btn)

        layout.addLayout(btn_layout)

        # Set scroll content and add to main layout
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

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

    def on_reprocessing_toggled(self, state):
        """Toggle confidence-based reprocessing controls."""
        enabled = state == Qt.Checked
        self.confidence_threshold_spin.setEnabled(enabled)
        self.reprocess_method_combo.setEnabled(enabled)

    def on_table_extraction_toggled(self, state):
        """Toggle table extraction controls."""
        enabled = state == Qt.Checked
        self.table_min_rows_spin.setEnabled(enabled)
        self.table_min_cols_spin.setEnabled(enabled)

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
                    # Initialize queue status
                    self.file_queue_status[str(path)] = {
                        'status': '⏳ Pending',
                        'progress': 0,
                        'pages': '-',
                        'confidence': '-',
                    }

            self.process_btn.setEnabled(len(self.selected_files) > 0)
            self.update_queue_table()
            self.log(f"Added {len(files)} file(s)")

    def add_folder(self):
        """Add all supported files from a folder."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder",
            "",
        )

        if folder:
            folder_path = Path(folder)
            # Support multiple file types
            supported_files = []
            for ext in ['*.pdf', '*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif']:
                supported_files.extend(folder_path.glob(ext))
                # Also check uppercase
                supported_files.extend(folder_path.glob(ext.upper()))

            added_count = 0
            for file_path in supported_files:
                if file_path not in self.selected_files:
                    self.selected_files.append(file_path)
                    # Initialize queue status
                    self.file_queue_status[str(file_path)] = {
                        'status': '⏳ Pending',
                        'progress': 0,
                        'pages': '-',
                        'confidence': '-',
                    }
                    added_count += 1

            self.process_btn.setEnabled(len(self.selected_files) > 0)
            self.update_queue_table()
            self.log(f"Added {added_count} file(s) from folder")

    def handle_dropped_files(self, file_paths: List[str]):
        """Handle files dropped into the drop zone."""
        added_count = 0
        for file_path_str in file_paths:
            file_path = Path(file_path_str)
            if file_path not in self.selected_files:
                self.selected_files.append(file_path)
                # Initialize queue status
                self.file_queue_status[str(file_path)] = {
                    'status': '⏳ Pending',
                    'progress': 0,
                    'pages': '-',
                    'confidence': '-',
                }
                added_count += 1

        self.process_btn.setEnabled(len(self.selected_files) > 0)
        self.update_queue_table()
        self.log(f"Added {added_count} file(s) via drag & drop")

    def clear_files(self):
        """Clear all files."""
        self.selected_files.clear()
        self.file_queue_status.clear()
        self.process_btn.setEnabled(False)
        self.update_queue_table()
        self.log("Cleared all files")

    def clear_all_files(self):
        """Clear all files and reset progress."""
        # Clear file list and queue
        self.selected_files.clear()
        self.file_queue_status.clear()
        self.update_queue_table()

        # Reset progress
        self.progress_bar.setValue(0)
        self.status_label.setText("Status: Ready")
        self.progress_label.setText("Files: 0 / 0 | Pages: 0 / 0")
        self.elapsed_label.setText("Elapsed: 00:00:00")
        self.time_label.setText("Remaining: --:--:--")
        self.error_label.setVisible(False)

        # Reset counters
        self.pages_processed = 0
        self.total_pages = 0
        self.files_processed = 0
        self.total_files = 0

        # Reset performance statistics
        self.reset_performance_stats()

        # Update queue table
        self.update_queue_table()

        # Disable buttons
        self.process_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.reprocess_low_conf_btn.setEnabled(False)

        self.log("Cleared all files and reset")

    def update_elapsed_time(self):
        """Update elapsed time display."""
        if self.processing_start_time and self.is_processing:
            elapsed = time.time() - self.processing_start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            self.elapsed_label.setText(f"Elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}")

            # Sample memory usage periodically
            self.sample_memory_usage()

            # Update performance stats in real-time
            self.update_performance_stats()

    def update_status_display(self):
        """Update all status displays with current progress."""
        # Update progress bar
        if self.total_pages > 0:
            progress = int((self.pages_processed / self.total_pages) * 100)
            self.progress_bar.setValue(progress)

        # Update status label
        if self.is_processing:
            self.status_label.setText(f"Status: Processing... ({self.files_processed}/{self.total_files} files)")
        else:
            self.status_label.setText("Status: Ready")

        # Update progress label
        self.progress_label.setText(f"Files: {self.files_processed} / {self.total_files} | Pages: {self.pages_processed} / {self.total_pages}")

        # Calculate and update estimated remaining time
        if self.processing_start_time and self.pages_processed > 0 and self.is_processing:
            elapsed = time.time() - self.processing_start_time
            pages_per_second = self.pages_processed / elapsed
            if pages_per_second > 0:
                remaining_pages = self.total_pages - self.pages_processed
                remaining_seconds = remaining_pages / pages_per_second
                hours = int(remaining_seconds // 3600)
                minutes = int((remaining_seconds % 3600) // 60)
                seconds = int(remaining_seconds % 60)
                self.time_label.setText(f"Remaining: {hours:02d}:{minutes:02d}:{seconds:02d}")
            else:
                self.time_label.setText("Remaining: Calculating...")
        else:
            self.time_label.setText("Remaining: --:--:--")

    def show_error(self, message: str):
        """Display error message in status section."""
        self.error_label.setText(f"⚠ Error: {message}")
        self.error_label.setVisible(True)
        self.log(f"ERROR: {message}")

    def hide_error(self):
        """Hide error message."""
        self.error_label.setVisible(False)

    def update_performance_stats(self):
        """Update performance statistics display."""
        # Total processing time
        if self.processing_start_time:
            elapsed = time.time() - self.processing_start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            self.stats_total_time.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
            self.total_processing_time = elapsed

        # Average page time
        if self.page_processing_times:
            avg_time = sum(self.page_processing_times) / len(self.page_processing_times)
            self.stats_avg_page_time.setText(f"{avg_time:.2f} sec/page")
        elif self.pages_processed > 0 and self.total_processing_time > 0:
            avg_time = self.total_processing_time / self.pages_processed
            self.stats_avg_page_time.setText(f"{avg_time:.2f} sec/page")

        # Processing speed
        if self.total_processing_time > 0 and self.pages_processed > 0:
            pages_per_min = (self.pages_processed / self.total_processing_time) * 60
            self.stats_processing_speed.setText(f"{pages_per_min:.1f} pages/min")

        # Maximum memory usage
        if self.max_memory_usage > 0:
            self.stats_max_memory.setText(f"{self.max_memory_usage:.1f} MB")

        # DPI downshift count
        self.stats_dpi_downshift.setText(f"{self.dpi_downshift_count} times")

        # Text blocks detected
        self.stats_text_blocks.setText(f"{self.total_text_blocks:,} blocks")

    def sample_memory_usage(self):
        """Sample current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_samples.append(memory_mb)
            if memory_mb > self.max_memory_usage:
                self.max_memory_usage = memory_mb
        except ImportError:
            # psutil not available, skip memory tracking
            pass
        except Exception as e:
            logger.warning(f"Failed to sample memory: {e}")

    def reset_performance_stats(self):
        """Reset all performance statistics."""
        self.total_processing_time = 0.0
        self.page_processing_times.clear()
        self.max_memory_usage = 0.0
        self.dpi_downshift_count = 0
        self.total_text_blocks = 0
        self.memory_samples.clear()

        # Reset display
        self.stats_total_time.setText("--:--:--")
        self.stats_avg_page_time.setText("-- sec/page")
        self.stats_max_memory.setText("-- MB")
        self.stats_dpi_downshift.setText("0 times")
        self.stats_text_blocks.setText("0 blocks")
        self.stats_processing_speed.setText("-- pages/min")

    def select_output_directory(self):
        """Select output directory for results."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            str(self.output_directory) if self.output_directory else "",
        )

        if directory:
            self.output_directory = Path(directory)
            self.output_dir_edit.setText(str(self.output_directory))
            self.open_output_btn.setEnabled(True)
            self.log(f"Output directory: {self.output_directory}")

    def open_output_directory(self):
        """Open output directory in file explorer."""
        if not self.output_directory:
            QMessageBox.warning(self, "No Directory", "Please select an output directory first")
            return

        # Create directory if it doesn't exist
        self.output_directory.mkdir(parents=True, exist_ok=True)

        # Open directory in file explorer
        import subprocess
        import sys

        try:
            if sys.platform == 'darwin':  # macOS
                subprocess.run(['open', str(self.output_directory)])
            elif sys.platform == 'win32':  # Windows
                subprocess.run(['explorer', str(self.output_directory)])
            else:  # Linux
                subprocess.run(['xdg-open', str(self.output_directory)])

            self.log(f"Opened directory: {self.output_directory}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to open directory: {str(e)}")
            self.log(f"Failed to open directory: {str(e)}")

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

        # Prepare thresholds
        thresholds = {
            'low_confidence': self.low_conf_threshold.value(),
        }

        # Update config with GUI settings
        updated_config = self.config.copy()

        # Update PDF settings
        updated_config['pdf']['dpi'] = self.dpi_spin.value()

        # Update preprocessing settings
        updated_config['preprocessing']['auto_rotate'] = self.auto_rotate_check.isChecked()
        updated_config['preprocessing']['deskew'] = self.deskew_check.isChecked()
        updated_config['preprocessing']['enhance_contrast'] = self.enhance_check.isChecked()

        # Update postprocessing settings
        updated_config['postprocessing']['normalize_whitespace'] = self.normalize_whitespace_check.isChecked()
        updated_config['postprocessing']['fix_common_errors'] = self.fix_errors_check.isChecked()

        # Update system settings
        if 'system' not in updated_config:
            updated_config['system'] = {}
        updated_config['system']['spell_correction'] = self.spell_correction_combo.currentText()
        updated_config['system']['auto_dpi_downshift'] = self.auto_dpi_downshift_check.isChecked()
        updated_config['system']['memory_threshold'] = self.memory_threshold_spin.value()
        updated_config['system']['enable_reprocessing'] = self.enable_reprocessing_check.isChecked()
        updated_config['system']['confidence_threshold'] = self.confidence_threshold_spin.value()
        updated_config['system']['reprocess_method'] = self.reprocess_method_combo.currentText()

        # Update device settings
        updated_config['device']['num_workers'] = self.workers_spin.value()
        self.thread_pool.setMaxThreadCount(self.workers_spin.value())

        # Prepare for processing
        self.current_results.clear()
        self.failed_pages.clear()
        self.pages_processed = 0
        self.files_processed = 0
        self.total_files = len(self.selected_files)
        self.total_pages = 0  # Will be updated as we process
        self.processing_start_time = time.time()
        self.active_workers.clear()
        self.is_processing = True

        # Reset performance statistics
        self.reset_performance_stats()

        # Update UI
        self.process_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.hide_error()

        # Start timer for elapsed time updates
        self.update_timer.start()

        # Update status display
        self.status_label.setText(f"Status: Starting... (0/{self.total_files} files)")
        self.progress_label.setText(f"Files: 0 / {self.total_files} | Pages: 0 / 0")

        self.log(f"Starting OCR processing with engine: {engine}")
        self.log(f"Settings: DPI={self.dpi_spin.value()}, Workers={self.workers_spin.value()}")

        # Get selected export formats
        export_formats = []
        if self.export_txt_check.isChecked():
            export_formats.append('txt')
        if self.export_json_check.isChecked():
            export_formats.append('json')
        if self.export_md_check.isChecked():
            export_formats.append('markdown')
        if self.export_searchable_pdf_check.isChecked():
            export_formats.append('searchable_pdf')
        if self.extract_tables_check.isChecked():
            export_formats.append('tables_csv')

        # Table extraction settings
        table_settings = {
            'enabled': self.extract_tables_check.isChecked(),
            'min_rows': self.table_min_rows_spin.value(),
            'min_cols': self.table_min_cols_spin.value(),
        }

        # Process each file
        for file_path in self.selected_files:
            # Update queue status to processing
            self.update_file_queue_status(
                str(file_path),
                '🔄 Processing',
                progress=0,
                pages=0,
                confidence=0.0
            )

            worker = PDFOCRWorker(
                file_path=file_path,
                config=updated_config,
                cache_manager=self.cache_manager if self.cache_check.isChecked() else None,
                engine=engine,
                page_range=page_range,
                thresholds=thresholds,
                use_ensemble=True,  # Always use ensemble mode
                output_directory=self.output_directory,
                export_formats=export_formats,
                table_settings=table_settings,
            )

            # Connect signals
            worker.signals.progress.connect(self.on_progress)
            worker.signals.page_completed.connect(self.on_page_completed)
            worker.signals.page_failed.connect(self.on_page_failed)
            worker.signals.finished.connect(self.on_processing_finished)
            worker.signals.error.connect(self.on_processing_error)
            worker.signals.log_message.connect(self.log)
            worker.signals.status_update.connect(lambda msg: self.statusBar().showMessage(msg))

            # Store worker reference in queue status
            self.file_queue_status[str(file_path)]['worker'] = worker

            # Add to active workers and thread pool
            self.active_workers.append(worker)
            self.thread_pool.start(worker)

        self.log(f"Processing {len(self.selected_files)} file(s)...")
        self.statusBar().showMessage("Processing...")

    def reprocess_low_confidence(self):
        """Reprocess pages with low confidence (button callback)."""
        if not self.failed_pages and not self.low_confidence_pages:
            QMessageBox.information(
                self,
                "No Pages to Reprocess",
                "There are no failed or low-confidence pages to reprocess."
            )
            return

        # Count total pages to reprocess
        total_pages = len(self.failed_pages)
        for pages in self.low_confidence_pages.values():
            total_pages += len(pages)

        # Ask user for confirmation
        reply = QMessageBox.question(
            self,
            "Reprocess Pages",
            f"Found {total_pages} page(s) to reprocess.\n\n"
            f"Failed pages: {len(self.failed_pages)}\n"
            f"Low-confidence pages: {sum(len(p) for p in self.low_confidence_pages.values())}\n\n"
            "Do you want to reprocess them?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply == QMessageBox.No:
            return

        self.log(f"Starting reprocessing of {total_pages} page(s)...")

        # For now, just call the auto-reprocessing logic
        # In future, this could create workers for specific pages only
        if self.low_confidence_pages:
            self.reprocess_low_confidence_pages()

        if self.failed_pages:
            self.log(f"Note: Failed pages reprocessing not yet implemented")
            self.log(f"Failed pages: {len(self.failed_pages)}")

    def reprocess_low_confidence_pages(self):
        """Automatically reprocess pages with low confidence."""
        if not self.low_confidence_pages:
            return

        self.log("=== Starting Low-Confidence Page Reprocessing ===")
        method = self.reprocess_method_combo.currentText()
        self.log(f"Reprocessing method: {method}")

        # For now, we'll log the pages that would be reprocessed
        # Full implementation would require creating new workers with modified settings
        for file_path, page_nums in self.low_confidence_pages.items():
            if not page_nums:
                continue

            file_name = Path(file_path).name
            self.log(f"File: {file_name}")
            self.log(f"  Low-confidence pages: {', '.join(map(str, sorted(page_nums)))}")

            # Log average confidence for these pages
            if file_path in self.page_confidences:
                low_conf_values = [self.page_confidences[file_path][p] for p in page_nums if p in self.page_confidences[file_path]]
                if low_conf_values:
                    avg_conf = sum(low_conf_values) / len(low_conf_values)
                    self.log(f"  Average confidence: {avg_conf:.2%}")

            # TODO: Implement actual reprocessing logic
            # This would involve:
            # 1. For "Alternative Processor": Switch DocAI processor type
            # 2. For "Higher DPI": Increase DPI (e.g., 300 -> 400)
            # 3. For "Both": Try alternative processor first, then higher DPI if still low
            # 4. Create new worker with modified config for just these pages
            # 5. Merge reprocessed results back into original result

        self.log("Note: Full reprocessing implementation pending - currently logging only")
        self.log("=== Reprocessing Check Complete ===")

    def cancel_processing(self):
        """Cancel ongoing processing."""
        self.log("Cancelling processing...")

        # Stop timer
        self.is_processing = False
        self.update_timer.stop()

        # Update status
        self.status_label.setText(f"Status: Cancelled ({self.files_processed}/{self.total_files} files processed)")
        self.show_error("Processing cancelled by user")

        # Clear active workers
        for worker in self.active_workers:
            worker.cancel()

        # Update UI
        self.cancel_btn.setEnabled(False)
        self.process_btn.setEnabled(True)
        self.statusBar().showMessage("Cancelled")

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

        # Update total pages if we got more info
        if total_pages > self.total_pages:
            self.total_pages = total_pages

        # Update status display with all current info
        self.update_status_display()

    def on_progress(self, message: str, current: int, total: int):
        """Handle progress update."""
        self.update_progress(current, total)
        # Progress label is updated by update_status_display()

    def on_page_completed(self, page_num: int, result: dict):
        """Handle page completion."""
        # Collect statistics
        blocks = result.get('blocks', 0)
        self.total_text_blocks += blocks

        # Track page processing time if available
        if 'processing_time' in result:
            self.page_processing_times.append(result['processing_time'])

        # Track page confidence for potential reprocessing
        file_path = result.get('file_path')
        confidence = result.get('confidence', 0)

        if file_path:
            # Initialize tracking structures if needed
            if file_path not in self.page_confidences:
                self.page_confidences[file_path] = {}
                self.low_confidence_pages[file_path] = []

            self.page_confidences[file_path][page_num] = confidence

            # Check if reprocessing is enabled and confidence is below threshold
            if self.enable_reprocessing_check.isChecked():
                threshold = self.confidence_threshold_spin.value() / 100.0
                if confidence < threshold:
                    self.low_confidence_pages[file_path].append(page_num)
                    self.log(f"⚠ Page {page_num} marked for reprocessing (confidence: {confidence:.2%} < {threshold:.0%})")

        self.log(f"Page {page_num} completed: {blocks} blocks, "
                f"confidence: {confidence:.2%}")

    def on_page_failed(self, page_num: int, reason: str, thumbnail: QPixmap):
        """Handle page failure."""
        self.log(f"Page {page_num} failed: {reason}")
        # Add to error view
        # For now, we'll skip adding to UI since we need the file path
        # This can be enhanced later

    def on_processing_finished(self, result: dict):
        """Handle processing completion."""
        file_path = result.get('file_path')
        self.current_results[file_path] = result

        # Update file counter
        self.files_processed += 1

        # Update total pages if available
        pages_in_file = result.get('total_pages', 0)
        if pages_in_file > 0:
            self.total_pages += pages_in_file

        # Collect statistics from result
        if 'statistics' in result:
            stats = result['statistics']
            # DPI downshift count
            if 'dpi_downshifts' in stats:
                self.dpi_downshift_count += stats['dpi_downshifts']
            # Text blocks
            if 'text_blocks' in stats:
                self.total_text_blocks += stats['text_blocks']

        # Update queue status to completed
        avg_confidence = result.get('average_confidence', 0.0)
        self.update_file_queue_status(
            file_path,
            '✓ Completed',
            progress=100,
            pages=pages_in_file,
            confidence=avg_confidence
        )

        # Update status display
        self.update_status_display()

        # Enable "Open Folder" button if output directory is set
        if self.output_directory:
            self.open_output_btn.setEnabled(True)
        elif result.get('output_files'):
            # If no output directory set, enable button based on first output file's parent
            output_files = result.get('output_files', {})
            if output_files:
                first_output = list(output_files.values())[0]
                self.output_directory = Path(first_output).parent
                self.output_dir_edit.setText(str(self.output_directory))
                self.open_output_btn.setEnabled(True)

        # Check if all workers are done
        self.check_all_workers_done()

    def on_processing_error(self, error_message: str):
        """Handle processing error."""
        self.log(f"Error: {error_message}")
        self.show_error(error_message)

        # Update processing state
        self.is_processing = False
        self.update_timer.stop()

        # Re-enable process button
        self.process_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def check_all_workers_done(self):
        """Check if all workers are finished."""
        if self.thread_pool.activeThreadCount() == 0:
            self.log("All processing completed")
            self.statusBar().showMessage("Completed")

            # Stop timer and update final status
            self.is_processing = False
            self.update_timer.stop()

            # Update progress bar to 100%
            self.progress_bar.setValue(100)
            self.status_label.setText(f"Status: Completed ({self.files_processed}/{self.total_files} files)")

            # Calculate and display final statistics
            if self.processing_start_time:
                elapsed = time.time() - self.processing_start_time
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                seconds = int(elapsed % 60)
                self.log(f"Total processing time: {hours:02d}:{minutes:02d}:{seconds:02d}")

                # Update final performance statistics
                self.update_performance_stats()

                # Log statistics
                if self.pages_processed > 0:
                    avg_time = elapsed / self.pages_processed
                    self.log(f"Average page time: {avg_time:.2f} sec/page")
                    pages_per_min = (self.pages_processed / elapsed) * 60
                    self.log(f"Processing speed: {pages_per_min:.1f} pages/min")
                if self.max_memory_usage > 0:
                    self.log(f"Peak memory usage: {self.max_memory_usage:.1f} MB")
                if self.dpi_downshift_count > 0:
                    self.log(f"DPI downshifts: {self.dpi_downshift_count}")
                if self.total_text_blocks > 0:
                    self.log(f"Total text blocks: {self.total_text_blocks:,}")

            self.process_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)

            # Auto-reprocess low-confidence pages if enabled
            if self.enable_reprocessing_check.isChecked():
                total_low_conf = sum(len(pages) for pages in self.low_confidence_pages.values())
                if total_low_conf > 0:
                    self.log(f"Found {total_low_conf} low-confidence page(s) across {len(self.low_confidence_pages)} file(s)")
                    self.reprocess_low_confidence_pages()

            # Show reprocess button if there are low confidence pages
            if self.failed_pages:
                self.reprocess_low_conf_btn.setEnabled(True)

    def save_settings(self):
        """Save current GUI settings to file."""
        import json

        settings = {
            'output_directory': str(self.output_directory) if self.output_directory else None,
            'page_range': {
                'all_pages': self.all_pages_check.isChecked(),
                'start': self.page_start.value(),
                'end': self.page_end.value(),
            },
            'engine': self.engine_combo.currentText(),
            'thresholds': {
                'low_confidence': self.low_conf_threshold.value(),
            },
            'cache_enabled': self.cache_check.isChecked(),
            'export_formats': {
                'txt': self.export_txt_check.isChecked(),
                'json': self.export_json_check.isChecked(),
                'markdown': self.export_md_check.isChecked(),
                'searchable_pdf': self.export_searchable_pdf_check.isChecked(),
                'tables_csv': self.extract_tables_check.isChecked(),
            },
            'table_extraction': {
                'enabled': self.extract_tables_check.isChecked(),
                'min_rows': self.table_min_rows_spin.value(),
                'min_cols': self.table_min_cols_spin.value(),
            },
            'advanced_options': {
                'dpi': self.dpi_spin.value(),
                'auto_rotate': self.auto_rotate_check.isChecked(),
                'deskew': self.deskew_check.isChecked(),
                'enhance': self.enhance_check.isChecked(),
                'normalize_whitespace': self.normalize_whitespace_check.isChecked(),
                'fix_common_errors': self.fix_errors_check.isChecked(),
                'workers': self.workers_spin.value(),
            },
            'hardware': {
                'parallel_pages': self.parallel_pages_spin.value(),
            },
            'system': {
                'spell_correction': self.spell_correction_combo.currentText(),
                'auto_dpi_downshift': self.auto_dpi_downshift_check.isChecked(),
                'memory_threshold': self.memory_threshold_spin.value(),
                'enable_reprocessing': self.enable_reprocessing_check.isChecked(),
                'confidence_threshold': self.confidence_threshold_spin.value(),
                'reprocess_method': self.reprocess_method_combo.currentText(),
            },
            'window': {
                'geometry': {
                    'x': self.geometry().x(),
                    'y': self.geometry().y(),
                    'width': self.geometry().width(),
                    'height': self.geometry().height(),
                }
            }
        }

        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            self.log(f"Settings saved to {self.settings_file}")
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")

    def load_settings(self):
        """Load GUI settings from file."""
        import json

        if not self.settings_file.exists():
            return

        try:
            with open(self.settings_file, 'r') as f:
                settings = json.load(f)

            # Output directory
            if settings.get('output_directory'):
                self.output_directory = Path(settings['output_directory'])
                self.output_dir_edit.setText(str(self.output_directory))

            # Page range
            page_range = settings.get('page_range', {})
            self.all_pages_check.setChecked(page_range.get('all_pages', True))
            self.page_start.setValue(page_range.get('start', 1))
            self.page_end.setValue(page_range.get('end', 9999))

            # Engine
            engine = settings.get('engine', 'docai')
            index = self.engine_combo.findText(engine)
            if index >= 0:
                self.engine_combo.setCurrentIndex(index)

            # Thresholds
            thresholds = settings.get('thresholds', {})
            self.low_conf_threshold.setValue(thresholds.get('low_confidence', 0.5))

            # Cache
            self.cache_check.setChecked(settings.get('cache_enabled', True))

            # Export formats
            export_formats = settings.get('export_formats', {})
            self.export_txt_check.setChecked(export_formats.get('txt', False))
            self.export_json_check.setChecked(export_formats.get('json', False))
            self.export_md_check.setChecked(export_formats.get('markdown', False))
            self.export_searchable_pdf_check.setChecked(export_formats.get('searchable_pdf', True))
            self.extract_tables_check.setChecked(export_formats.get('tables_csv', False))

            # Table extraction settings
            table_settings = settings.get('table_extraction', {})
            self.table_min_rows_spin.setValue(table_settings.get('min_rows', 3))
            self.table_min_cols_spin.setValue(table_settings.get('min_cols', 2))

            # Advanced options
            advanced = settings.get('advanced_options', {})
            self.dpi_spin.setValue(advanced.get('dpi', 300))
            self.auto_rotate_check.setChecked(advanced.get('auto_rotate', True))
            self.deskew_check.setChecked(advanced.get('deskew', False))
            self.enhance_check.setChecked(advanced.get('enhance', False))
            self.normalize_whitespace_check.setChecked(advanced.get('normalize_whitespace', True))
            self.fix_errors_check.setChecked(advanced.get('fix_common_errors', True))
            self.workers_spin.setValue(advanced.get('workers', 4))

            # Hardware settings
            hardware = settings.get('hardware', {})
            self.parallel_pages_spin.setValue(hardware.get('parallel_pages', 1))

            # System settings
            system = settings.get('system', {})
            spell_correction = system.get('spell_correction', 'Disabled')
            index = self.spell_correction_combo.findText(spell_correction)
            if index >= 0:
                self.spell_correction_combo.setCurrentIndex(index)
            self.auto_dpi_downshift_check.setChecked(system.get('auto_dpi_downshift', True))
            self.memory_threshold_spin.setValue(system.get('memory_threshold', 85))

            # Reprocessing settings
            self.enable_reprocessing_check.setChecked(system.get('enable_reprocessing', False))
            self.confidence_threshold_spin.setValue(system.get('confidence_threshold', 70))
            reprocess_method = system.get('reprocess_method', 'Alternative Processor')
            index = self.reprocess_method_combo.findText(reprocess_method)
            if index >= 0:
                self.reprocess_method_combo.setCurrentIndex(index)

            # Window geometry
            window_settings = settings.get('window', {})
            geometry = window_settings.get('geometry', {})
            if geometry:
                self.setGeometry(
                    geometry.get('x', 100),
                    geometry.get('y', 100),
                    geometry.get('width', 1400),
                    geometry.get('height', 900)
                )

            self.log(f"Settings loaded from {self.settings_file}")
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")

    # ===== Batch Queue Management =====

    def update_queue_table(self):
        """Update the batch queue table with current status."""
        self.queue_table.setRowCount(len(self.selected_files))

        for idx, file_path in enumerate(self.selected_files):
            file_path_str = str(file_path)

            # File name
            file_item = QTableWidgetItem(file_path.name)
            file_item.setToolTip(file_path_str)
            self.queue_table.setItem(idx, 0, file_item)

            # Status
            status_info = self.file_queue_status.get(file_path_str, {})
            status = status_info.get('status', '⏳ Pending')
            status_item = QTableWidgetItem(status)

            # Color code status
            if '✓' in status or 'Completed' in status:
                status_item.setForeground(QColor('#4CAF50'))
            elif '🔄' in status or 'Processing' in status:
                status_item.setForeground(QColor('#2196F3'))
            elif '✗' in status or 'Failed' in status:
                status_item.setForeground(QColor('#f44336'))
            else:
                status_item.setForeground(QColor('#888'))

            self.queue_table.setItem(idx, 1, status_item)

            # Progress
            progress = status_info.get('progress', 0)
            progress_item = QTableWidgetItem(f"{progress}%")
            self.queue_table.setItem(idx, 2, progress_item)

            # Pages
            pages = status_info.get('pages', '-')
            pages_item = QTableWidgetItem(str(pages))
            self.queue_table.setItem(idx, 3, pages_item)

            # Confidence
            confidence = status_info.get('confidence', '-')
            if isinstance(confidence, float):
                confidence_item = QTableWidgetItem(f"{confidence:.1%}")
                # Color code confidence
                if confidence >= 0.9:
                    confidence_item.setForeground(QColor('#4CAF50'))
                elif confidence >= 0.7:
                    confidence_item.setForeground(QColor('#FFC107'))
                else:
                    confidence_item.setForeground(QColor('#f44336'))
            else:
                confidence_item = QTableWidgetItem(str(confidence))
            self.queue_table.setItem(idx, 4, confidence_item)

    def on_queue_selection_changed(self):
        """Handle queue table selection change."""
        has_selection = len(self.queue_table.selectedItems()) > 0
        self.move_up_btn.setEnabled(has_selection and not self.is_processing)
        self.move_down_btn.setEnabled(has_selection and not self.is_processing)
        self.remove_queue_btn.setEnabled(has_selection and not self.is_processing)

    def move_queue_item_up(self):
        """Move selected queue item up."""
        current_row = self.queue_table.currentRow()
        if current_row > 0:
            # Swap in selected_files list
            self.selected_files[current_row], self.selected_files[current_row - 1] = \
                self.selected_files[current_row - 1], self.selected_files[current_row]

            # Update queue table and select new position
            self.update_queue_table()
            self.queue_table.selectRow(current_row - 1)

            self.log(f"Moved {self.selected_files[current_row - 1].name} up in queue")

    def move_queue_item_down(self):
        """Move selected queue item down."""
        current_row = self.queue_table.currentRow()
        if current_row < len(self.selected_files) - 1:
            # Swap in selected_files list
            self.selected_files[current_row], self.selected_files[current_row + 1] = \
                self.selected_files[current_row + 1], self.selected_files[current_row]

            # Update queue table and select new position
            self.update_queue_table()
            self.queue_table.selectRow(current_row + 1)

            self.log(f"Moved {self.selected_files[current_row + 1].name} down in queue")

    def remove_queue_item(self):
        """Remove selected item from queue."""
        current_row = self.queue_table.currentRow()
        if current_row >= 0:
            removed_file = self.selected_files.pop(current_row)
            file_path_str = str(removed_file)

            # Remove from queue status
            if file_path_str in self.file_queue_status:
                del self.file_queue_status[file_path_str]

            self.update_queue_table()
            self.process_btn.setEnabled(len(self.selected_files) > 0)
            self.log(f"Removed {removed_file.name} from queue")

    def update_file_queue_status(self, file_path: str, status: str, progress: int = 0,
                                  pages: int = 0, confidence: float = 0.0):
        """Update status for a file in the queue."""
        self.file_queue_status[file_path] = {
            'status': status,
            'progress': progress,
            'pages': pages,
            'confidence': confidence,
        }
        self.update_queue_table()

    def closeEvent(self, event):
        """Handle window close event."""
        # Save settings before closing
        self.save_settings()

        # Cancel all workers
        for worker in self.active_workers:
            worker.cancel()

        # Clean up thread pool
        self.thread_pool.waitForDone()

        # Record a "recent exit" marker so that if the macOS
        # .app bundle or launcher tries to immediately relaunch
        # the process, the next startup can detect it and exit
        # without opening a new window.
        try:
            _recent_exit_marker.parent.mkdir(parents=True, exist_ok=True)
            _recent_exit_marker.write_text(str(time.time()))
        except Exception:
            # If writing the marker fails, we still proceed with normal quit.
            pass

        # Accept the close event and explicitly quit the application.
        # This helps ensure that when the main window is closed from
        # a bundled .app, the Qt event loop stops and the process
        # terminates instead of immediately re-opening a new window.
        event.accept()
        app = QApplication.instance()
        if app is not None:
            app.quit()


def main():
    """Main entry point."""
    # If the app was closed very recently, do not allow an immediate
    # relaunch. On some macOS setups, a bundled .app (or associated
    # launcher) can try to restart the process when it exits, which
    # manifests as the window reappearing right after the user quits.
    try:
        if _recent_exit_marker.exists():
            mtime = _recent_exit_marker.stat().st_mtime
            if time.time() - mtime < _recent_exit_timeout_seconds:
                # Treat this as an undesired automatic relaunch and exit quietly.
                return
    except Exception:
        # If anything goes wrong checking the marker, continue startup normally.
        pass

    # Prevent multiple GUI instances (helps avoid infinite-window behavior)
    if not acquire_single_instance_lock():
        # Another instance is already running – avoid opening another window
        # (In a bundled .app there is usually no visible console, so we just exit quietly.)
        return

    # Setup logging
    setup_logging(log_level='INFO')

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Hybrid PDF OCR")
    app.setStyle("Fusion")  # Modern cross-platform style
    # Ensure the app quits when the last window is closed
    app.setQuitOnLastWindowClosed(True)

    # Create and show main window
    config_path = Path("configs/app.yaml") if Path("configs/app.yaml").exists() else None
    window = HybridOCRApp(config_path)
    window.show()

    # Run event loop
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
