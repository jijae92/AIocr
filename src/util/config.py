"""
Configuration loading and management utilities.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from util.logging import get_logger

logger = get_logger(__name__)


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load application configuration from YAML file.

    Args:
        config_path: Path to config file (default: configs/app.yaml)

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Try default locations
        default_paths = [
            Path("configs/app.yaml"),
            Path(__file__).parent.parent.parent / "configs" / "app.yaml",
        ]

        for path in default_paths:
            if path.exists():
                config_path = path
                break

    if config_path is None or not config_path.exists():
        logger.warning(f"Config file not found, using defaults")
        return get_default_config()

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Expand environment variables
        config = expand_env_vars(config)

        logger.info(f"Loaded configuration from {config_path}")
        return config

    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return get_default_config()


def expand_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively expand environment variables in config values.

    Supports ${VAR_NAME} syntax.

    Args:
        config: Configuration dictionary

    Returns:
        Config with expanded environment variables
    """
    if isinstance(config, dict):
        return {k: expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [expand_env_vars(item) for item in config]
    elif isinstance(config, str):
        # Expand environment variables
        if '${' in config:
            # Replace ${VAR} with environment variable value
            import re
            pattern = r'\$\{([^}]+)\}'

            def replacer(match):
                var_name = match.group(1)
                return os.environ.get(var_name, match.group(0))

            return re.sub(pattern, replacer, config)
        return config
    else:
        return config


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.

    Returns:
        Default configuration dictionary
    """
    return {
        'app': {
            'name': 'Hybrid PDF OCR',
            'version': '0.1.0',
            'cache_dir': '~/.cache/hybrid-pdf-ocr',
            'log_dir': '~/.cache/hybrid-pdf-ocr/logs',
            'temp_dir': '/tmp/hybrid-pdf-ocr',
        },
        'device': {
            'preferred': 'cpu',
            'fallback': 'cpu',
            'num_workers': 4,
            'num_processes': 2,
        },
        'pdf': {
            'dpi': 300,
            'max_pages_per_batch': 10,
            'page_range': None,
            'image_format': 'PNG',
        },
        'thresholds': {
            'docai_confidence': 0.85,
            'low_confidence': 0.5,
            'high_confidence': 0.95,
            'simple_table_cells': 50,
            'multi_column_threshold': 2,
            'formula_confidence': 0.7,
            'numeric_content_ratio': 0.7,
        },
        'routing': {
            'strategy': 'adaptive',
            'parallel_execution': False,
            'max_parallel_models': 3,
            'block_level_routing': True,
        },
        'ensemble': {
            'enabled': True,
            'strategy': 'weighted_voting',
            'min_models': 2,
            'max_models': 3,
        },
        'preprocessing': {
            'auto_rotate': True,
            'deskew': True,
            'enhance_contrast': True,
            'denoise': False,
        },
        'postprocessing': {
            'normalize_whitespace': True,
            'fix_common_errors': True,
            'merge_blocks': True,
            'preserve_layout': True,
        },
        'export': {
            'formats': ['json', 'txt', 'searchable_pdf'],
            'include_confidence': True,
            'include_bboxes': True,
            'pdf_text_opacity': 0,
        },
        'cache': {
            'enabled': True,
            'strategy': 'content_hash',
            'ttl': 3600,
            'max_size_mb': 1024,
        },
        'logging': {
            'level': 'INFO',
            'log_routing_decisions': True,
            'log_confidence_scores': True,
            'log_timing': True,
        },
    }


def save_config(config: Dict[str, Any], output_path: Path):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Saved configuration to {output_path}")

    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        raise
