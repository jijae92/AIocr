"""
Logging utilities with PII filtering and JSONL audit trail support.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# PII patterns to filter
PII_PATTERNS = [
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),  # Email
    (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),  # SSN
    (r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]'),  # Phone
    (r'\b\d{16}\b', '[CARD]'),  # Credit card
    (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP]'),  # IP address
]


class PIIFilter(logging.Filter):
    """Filter to remove PII from log messages."""

    def __init__(self, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        self.patterns = [(re.compile(pattern), replacement) for pattern, replacement in PII_PATTERNS]

    def filter(self, record: logging.LogRecord) -> bool:
        if self.enabled and hasattr(record, 'msg'):
            msg = str(record.msg)
            for pattern, replacement in self.patterns:
                msg = pattern.sub(replacement, msg)
            record.msg = msg
        return True


class JSONLFormatter(logging.Formatter):
    """Formatter for JSONL output."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add extra fields
        if hasattr(record, 'extra'):
            log_data.update(record.extra)

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


class AuditLogger:
    """Audit logger for tracking all operations."""

    def __init__(self, log_dir: Path, filter_pii: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.filter_pii = filter_pii

        # Create audit log file
        self.audit_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"

        # Setup logger
        self.logger = logging.getLogger('audit')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # File handler with JSONL format
        file_handler = logging.FileHandler(self.audit_file, encoding='utf-8')
        file_handler.setFormatter(JSONLFormatter())
        if filter_pii:
            file_handler.addFilter(PIIFilter(enabled=True))

        self.logger.addHandler(file_handler)

    def log(self, operation: str, details: Optional[Dict[str, Any]] = None, **kwargs):
        """Log an audit event."""
        log_data = {
            'operation': operation,
            'details': details or {},
            **kwargs,
        }
        self.logger.info('', extra={'extra': log_data})

    def log_ocr_request(
        self,
        file_path: str,
        engine: str,
        page_count: int,
        **kwargs,
    ):
        """Log OCR request."""
        self.log(
            'ocr_request',
            details={
                'file_path': Path(file_path).name,  # Only log filename, not full path
                'engine': engine,
                'page_count': page_count,
                **kwargs,
            },
        )

    def log_ocr_result(
        self,
        file_path: str,
        engine: str,
        success: bool,
        duration: float,
        error: Optional[str] = None,
        **kwargs,
    ):
        """Log OCR result."""
        self.log(
            'ocr_result',
            details={
                'file_path': Path(file_path).name,
                'engine': engine,
                'success': success,
                'duration_seconds': duration,
                'error': error,
                **kwargs,
            },
        )


def setup_logging(
    log_level: str = 'INFO',
    log_dir: Optional[Path] = None,
    log_format: str = 'text',
    filter_pii: bool = True,
    rotation_size_mb: int = 10,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        log_format: Log format ('text' or 'json')
        filter_pii: Whether to filter PII from logs
        rotation_size_mb: Log file rotation size in MB
        backup_count: Number of backup files to keep

    Returns:
        Configured logger
    """
    logger = logging.getLogger('hybrid_ocr')
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    if log_format == 'json':
        console_handler.setFormatter(JSONLFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
            )
        )

    if filter_pii:
        console_handler.addFilter(PIIFilter(enabled=True))

    logger.addHandler(console_handler)

    # File handler (if log_dir provided)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            log_dir / 'app.log',
            maxBytes=rotation_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8',
        )

        if log_format == 'json':
            file_handler.setFormatter(JSONLFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                )
            )

        if filter_pii:
            file_handler.addFilter(PIIFilter(enabled=True))

        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(f'hybrid_ocr.{name}')
