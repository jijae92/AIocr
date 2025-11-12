"""
Content-addressable cache for OCR results.
"""

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from util.logging import get_logger

logger = get_logger(__name__)


class CacheStore:
    """Content-addressable cache store."""

    def __init__(
        self,
        cache_dir: Path,
        max_size_mb: Optional[int] = None,
        ttl_seconds: Optional[int] = None,
    ):
        """
        Initialize cache store.

        Args:
            cache_dir: Directory for cache storage
            max_size_mb: Maximum cache size in MB (None for unlimited)
            ttl_seconds: Time-to-live in seconds (None for no expiration)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.ttl_seconds = ttl_seconds

        # Metadata file
        self.metadata_file = self.cache_dir / '_metadata.json'
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        return {}

    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")

    def _compute_hash(self, content: bytes) -> str:
        """
        Compute SHA-256 hash of content.

        Args:
            content: Content bytes

        Returns:
            Hex digest of hash
        """
        return hashlib.sha256(content).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """
        Get cache file path for key.

        Args:
            cache_key: Cache key (hash)

        Returns:
            Path to cache file
        """
        # Use first 2 characters as subdirectory for distribution
        subdir = cache_key[:2]
        cache_subdir = self.cache_dir / subdir
        cache_subdir.mkdir(exist_ok=True)
        return cache_subdir / f"{cache_key}.json"

    def compute_key(self, file_path: Path, **params) -> str:
        """
        Compute cache key for file and parameters.

        Args:
            file_path: Path to file
            **params: Additional parameters (engine, settings, etc.)

        Returns:
            Cache key (hash)
        """
        # Read file content
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read file for cache key: {e}")
            raise

        # Create composite key from content + params
        key_data = content + json.dumps(params, sort_keys=True).encode('utf-8')
        return self._compute_hash(key_data)

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached result.

        Args:
            cache_key: Cache key

        Returns:
            Cached result or None if not found/expired
        """
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            logger.debug(f"Cache miss: {cache_key}")
            return None

        # Check TTL
        if self.ttl_seconds is not None:
            cached_time = self.metadata.get(cache_key, {}).get('timestamp')
            if cached_time:
                cached_dt = datetime.fromisoformat(cached_time)
                age_seconds = (datetime.now() - cached_dt).total_seconds()
                if age_seconds > self.ttl_seconds:
                    logger.debug(f"Cache expired: {cache_key}")
                    self.delete(cache_key)
                    return None

        # Load cached result
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            logger.debug(f"Cache hit: {cache_key}")

            # Update access time
            if cache_key in self.metadata:
                self.metadata[cache_key]['last_access'] = datetime.now().isoformat()
                self._save_metadata()

            return result
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return None

    def put(self, cache_key: str, result: Dict[str, Any], metadata: Optional[Dict] = None):
        """
        Store result in cache.

        Args:
            cache_key: Cache key
            result: Result to cache
            metadata: Optional metadata
        """
        cache_path = self._get_cache_path(cache_key)

        try:
            # Write cache file
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            # Update metadata
            self.metadata[cache_key] = {
                'timestamp': datetime.now().isoformat(),
                'last_access': datetime.now().isoformat(),
                'size_bytes': cache_path.stat().st_size,
                'metadata': metadata or {},
            }
            self._save_metadata()

            logger.debug(f"Cached result: {cache_key}")

            # Check cache size
            if self.max_size_mb is not None:
                self._enforce_size_limit()

        except Exception as e:
            logger.error(f"Failed to cache result: {e}")

    def delete(self, cache_key: str):
        """
        Delete cached result.

        Args:
            cache_key: Cache key
        """
        cache_path = self._get_cache_path(cache_key)

        try:
            if cache_path.exists():
                cache_path.unlink()
            if cache_key in self.metadata:
                del self.metadata[cache_key]
                self._save_metadata()
            logger.debug(f"Deleted cache: {cache_key}")
        except Exception as e:
            logger.error(f"Failed to delete cache: {e}")

    def _enforce_size_limit(self):
        """Enforce cache size limit by deleting oldest entries."""
        if self.max_size_mb is None:
            return

        # Calculate total size
        total_size = sum(
            meta.get('size_bytes', 0) for meta in self.metadata.values()
        )
        max_size_bytes = self.max_size_mb * 1024 * 1024

        if total_size <= max_size_bytes:
            return

        logger.info(f"Cache size {total_size / 1024 / 1024:.2f}MB exceeds limit, cleaning up")

        # Sort by last access time
        sorted_keys = sorted(
            self.metadata.keys(),
            key=lambda k: self.metadata[k].get('last_access', ''),
        )

        # Delete oldest until under limit
        for key in sorted_keys:
            if total_size <= max_size_bytes:
                break

            size = self.metadata[key].get('size_bytes', 0)
            self.delete(key)
            total_size -= size

        logger.info(f"Cache cleaned up, new size: {total_size / 1024 / 1024:.2f}MB")

    def clear(self):
        """Clear entire cache."""
        try:
            # Remove all cache files
            for cache_path in self.cache_dir.glob('*/*.json'):
                cache_path.unlink()

            # Remove subdirectories
            for subdir in self.cache_dir.glob('*'):
                if subdir.is_dir():
                    subdir.rmdir()

            # Clear metadata
            self.metadata.clear()
            self._save_metadata()

            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_size = sum(
            meta.get('size_bytes', 0) for meta in self.metadata.values()
        )

        return {
            'num_entries': len(self.metadata),
            'total_size_mb': total_size / 1024 / 1024,
            'max_size_mb': self.max_size_mb,
            'ttl_seconds': self.ttl_seconds,
        }


class CacheManager:
    """High-level cache manager."""

    def __init__(
        self,
        cache_dir: Path,
        enabled: bool = True,
        max_size_mb: Optional[int] = None,
        ttl_seconds: Optional[int] = None,
    ):
        """
        Initialize cache manager.

        Args:
            cache_dir: Cache directory
            enabled: Whether cache is enabled
            max_size_mb: Max cache size in MB
            ttl_seconds: TTL in seconds
        """
        self.enabled = enabled
        self.store = CacheStore(cache_dir, max_size_mb, ttl_seconds) if enabled else None

    def get_or_compute(
        self,
        file_path: Path,
        compute_fn,
        engine: str,
        **params,
    ) -> Dict[str, Any]:
        """
        Get cached result or compute it.

        Args:
            file_path: Input file path
            compute_fn: Function to compute result if not cached
            engine: OCR engine name
            **params: Additional parameters

        Returns:
            OCR result
        """
        if not self.enabled or self.store is None:
            return compute_fn()

        # Compute cache key
        cache_params = {'engine': engine, **params}
        cache_key = self.store.compute_key(file_path, **cache_params)

        # Try to get from cache
        cached_result = self.store.get(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached result for {file_path.name}")
            return cached_result

        # Compute result
        logger.info(f"Computing result for {file_path.name}")
        result = compute_fn()

        # Store in cache
        metadata = {
            'file_name': file_path.name,
            'engine': engine,
            'params': params,
        }
        self.store.put(cache_key, result, metadata)

        return result

    def clear(self):
        """Clear cache."""
        if self.enabled and self.store is not None:
            self.store.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.enabled and self.store is not None:
            return self.store.get_stats()
        return {'enabled': False}
