"""
Offline-first utilities for hybrid cloud/local storage and database switching.
Provides internet detection, storage mode detection, and data synchronization.
"""

import logging
import os
import socket
from datetime import datetime, timedelta
from typing import Tuple

from django.conf import settings
from django.db import DatabaseError, OperationalError, connection, connections

logger = logging.getLogger(__name__)

# Cache for connectivity checks (prevents excessive network calls)
_CONNECTIVITY_CACHE = {"timestamp": None, "is_online": None}
CONNECTIVITY_CACHE_DURATION = 30  # seconds


def is_online() -> bool:
    """
    Check if the system has internet connectivity.
    Uses cached results for 30 seconds to avoid excessive network calls.
    
    Returns:
        bool: True if online, False if offline (includes FORCE_OFFLINE check).
    """
    global _CONNECTIVITY_CACHE
    
    # Check for forced offline mode
    if os.getenv("FORCE_OFFLINE", "").lower() in {"1", "true", "yes", "on"}:
        logger.info("FORCE_OFFLINE environment variable set. Forcing offline mode.")
        return False
    
    # Check cache validity
    if _CONNECTIVITY_CACHE["timestamp"] is not None:
        cache_age = datetime.now() - _CONNECTIVITY_CACHE["timestamp"]
        if cache_age.total_seconds() < CONNECTIVITY_CACHE_DURATION:
            return _CONNECTIVITY_CACHE["is_online"]
    
    # Perform connectivity check
    is_connected = _check_connectivity()
    _CONNECTIVITY_CACHE["timestamp"] = datetime.now()
    _CONNECTIVITY_CACHE["is_online"] = is_connected
    
    return is_connected


def _check_connectivity() -> bool:
    """
    Perform actual connectivity test using multiple strategies.
    
    Returns:
        bool: True if internet is available.
    """
    # Strategy 1: Test DNS resolution (lightweight)
    if _test_dns():
        return True
    
    # Strategy 2: Test socket connection to public DNS
    if _test_socket_connection():
        return True
    
    # Strategy 3: Test connection to a public HTTP endpoint (if available)
    try:
        import urllib.request
        urllib.request.urlopen("http://8.8.8.8", timeout=3)
        return True
    except Exception:
        pass
    
    logger.debug("No internet connectivity detected")
    return False


def _test_dns() -> bool:
    """Test DNS resolution to 8.8.8.8 (Google DNS)."""
    try:
        socket.gethostbyname("8.8.8.8")
        logger.debug("DNS connectivity check passed")
        return True
    except Exception as e:
        logger.debug(f"DNS check failed: {e}")
        return False


def _test_socket_connection() -> bool:
    """Test socket connection to 8.8.8.8:53 (Google DNS)."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex(("8.8.8.8", 53))
        sock.close()
        if result == 0:
            logger.debug("Socket connectivity check passed")
            return True
    except Exception as e:
        logger.debug(f"Socket check failed: {e}")
    return False


def get_app_mode() -> Tuple[str, str]:
    """
    Get the current application mode (database and storage).
    
    Returns:
        Tuple[str, str]: (database_mode, storage_mode) where each is "cloud" or "offline"
    """
    db_mode = get_database_mode()
    storage_mode = get_storage_mode()
    return db_mode, storage_mode


def get_database_mode() -> str:
    """
    Determine the current database mode.
    
    Returns:
        str: "cloud" if using PostgreSQL/NeonDB, "offline" if using SQLite
    """
    if hasattr(settings, "DATABASES") and settings.DATABASES:
        engine = settings.DATABASES["default"].get("ENGINE", "")
        return "cloud" if "postgres" in engine.lower() else "offline"
    return "offline"


def get_storage_mode() -> str:
    """
    Determine the current storage mode.
    
    Returns:
        str: "cloud" if using Cloudinary, "offline" if using local storage
    """
    if hasattr(settings, "STORAGES"):
        default_backend = settings.STORAGES.get("default", {}).get("BACKEND", "")
        return "cloud" if "cloudinary" in default_backend.lower() else "offline"
    return "offline"


def is_database_online() -> bool:
    """
    Check if the cloud database (NeonDB) is currently accessible.
    
    Returns:
        bool: True if database is reachable, False otherwise
    """
    try:
        connection.ensure_connection()
        return True
    except (DatabaseError, OperationalError) as e:
        logger.warning(f"Database connectivity check failed: {e}")
        return False


def get_unsynced_records(model_class):
    """
    Retrieve all unsync­ed records for a given model.
    
    Args:
        model_class: The Django model class
    
    Returns:
        QuerySet: Records with is_synced=False
    """
    try:
        if hasattr(model_class, "is_synced"):
            return model_class.objects.filter(is_synced=False)
    except Exception as e:
        logger.error(f"Error retrieving unsynced records: {e}")
    return model_class.objects.none()


def mark_records_synced(records, batch_size=100):
    """
    Mark a batch of records as synced.
    
    Args:
        records: QuerySet of records to mark as synced
        batch_size: Number of records to update in each batch
    
    Returns:
        int: Number of records updated
    """
    try:
        count = 0
        for record in records.iterator(chunk_size=batch_size):
            if hasattr(record, "is_synced"):
                record.is_synced = True
                record.save(update_fields=["is_synced"])
                count += 1
        return count
    except Exception as e:
        logger.error(f"Error marking records as synced: {e}")
        return 0


def log_mode_change(from_mode: str, to_mode: str, component: str):
    """
    Log a mode change for auditing and debugging.
    
    Args:
        from_mode: Previous mode ("cloud" or "offline")
        to_mode: New mode ("cloud" or "offline")
        component: Component that changed (e.g., "database", "storage")
    """
    logger.info(f"[MODE CHANGE] {component}: {from_mode} → {to_mode}")
    
    # Also print to console for visibility
    mode_emoji = "☁️" if to_mode == "cloud" else "💾"
    print(f"\n{mode_emoji} {component.upper()} MODE: {to_mode.upper()}\n")


def log_startup_mode():
    """Log the application startup mode."""
    db_mode, storage_mode = get_app_mode()
    
    if is_online():
        status = "🟢 ONLINE - Using Cloud Mode"
        message = f"Using Cloud Mode (NeonDB + Cloudinary)"
    else:
        status = "🔴 OFFLINE - Using Offline Mode"
        message = f"Using Offline Mode (SQLite + Local Storage)"
    
    logger.info(f"{status}: {message}")
    print(f"\n{'=' * 60}")
    print(f"  {status}")
    print(f"  Database: {db_mode.upper()}")
    print(f"  Storage: {storage_mode.upper()}")
    print(f"{'=' * 60}\n")


def handle_sync_error(error: Exception, context: str = ""):
    """
    Gracefully handle synchronization errors.
    
    Args:
        error: The exception that occurred
        context: Additional context about where the error occurred
    """
    error_msg = f"Sync error in {context}: {str(error)[:100]}"
    logger.error(error_msg)
    
    # Log but don't raise - allow offline operation to continue


def ensure_media_root_exists():
    """Ensure the local MEDIA_ROOT directory exists for offline storage."""
    try:
        media_root = getattr(settings, "MEDIA_ROOT", None)
        if media_root:
            os.makedirs(media_root, exist_ok=True)
            logger.debug(f"Ensured MEDIA_ROOT exists: {media_root}")
    except Exception as e:
        logger.warning(f"Could not ensure MEDIA_ROOT: {e}")


def get_connectivity_status_dict() -> dict:
    """
    Get a complete connectivity status dictionary for use in templates/API responses.
    
    Returns:
        dict: Status information including modes, connectivity, and sync info
    """
    db_mode, storage_mode = get_app_mode()
    
    return {
        "is_online": is_online(),
        "database_mode": db_mode,
        "storage_mode": storage_mode,
        "database_reachable": is_database_online(),
        "timestamp": datetime.now().isoformat(),
        "status_message": (
            "System is online - using cloud services"
            if is_online()
            else "System is offline - using local storage"
        ),
    }
