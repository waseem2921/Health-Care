"""
Offline error handlers and recovery mechanisms for graceful degradation.
Provides exception classes and handler functions for offline scenarios.
"""

import logging
from typing import Dict, Optional

from django.http import JsonResponse

logger = logging.getLogger(__name__)


class OfflineError(Exception):
    """Base exception for offline-related errors."""
    pass


class StorageUnavailableError(OfflineError):
    """Raised when storage backend (Cloudinary) is unavailable."""
    pass


class DatabaseUnavailableError(OfflineError):
    """Raised when database is unavailable."""
    pass


class SyncError(OfflineError):
    """Raised when sync operation fails."""
    pass


def handle_storage_error(error: Exception, context: str = "", fallback_to_local: bool = True):
    """
    Handle storage-related errors gracefully.
    
    Args:
        error: The exception that occurred
        context: Additional context about where the error occurred
        fallback_to_local: Whether to fall back to local storage
    
    Returns:
        Dict: Error information and recovery suggestion
    """
    error_msg = f"Storage error in {context}: {str(error)[:100]}"
    logger.error(error_msg)
    
    recovery_msg = "Using local storage fallback" if fallback_to_local else "Storage temporarily unavailable"
    
    return {
        "error": True,
        "message": error_msg,
        "recovery": recovery_msg,
        "can_recover": fallback_to_local,
        "error_type": "storage",
    }


def handle_database_error(error: Exception, context: str = "", can_retry: bool = True):
    """
    Handle database-related errors gracefully.
    
    Args:
        error: The exception that occurred
        context: Additional context about where the error occurred
        can_retry: Whether the operation can be retried
    
    Returns:
        Dict: Error information and recovery suggestion
    """
    error_msg = f"Database error in {context}: {str(error)[:100]}"
    logger.error(error_msg)
    
    if can_retry:
        recovery_msg = "Using local database fallback. Your changes will sync when online."
    else:
        recovery_msg = "Database error - some functionality may be limited"
    
    return {
        "error": True,
        "message": error_msg,
        "recovery": recovery_msg,
        "can_recover": can_retry,
        "error_type": "database",
    }


def get_error_response(error_dict: Dict, status_code: int = 503) -> JsonResponse:
    """
    Create a JSON error response for API requests.
    
    Args:
        error_dict: Error information dictionary
        status_code: HTTP status code
    
    Returns:
        JsonResponse: Formatted error response
    """
    return JsonResponse(
        {
            "success": False,
            "error": True,
            "message": error_dict.get("message", "An error occurred"),
            "recovery": error_dict.get("recovery", ""),
            "can_recover": error_dict.get("can_recover", False),
            "error_type": error_dict.get("error_type", "unknown"),
        },
        status=status_code,
    )


def log_sync_event(
    event_type: str,
    model_name: str,
    record_id: int,
    success: bool = True,
    message: str = "",
):
    """
    Log a synchronization event.
    
    Args:
        event_type: Type of event ("save", "sync_attempt", "auto_sync", etc.)
        model_name: Name of the model
        record_id: ID of the record
        success: Whether the operation was successful
        message: Additional message
    """
    status = "✓" if success else "✗"
    log_msg = f"{status} [{event_type}] {model_name}(id={record_id})"
    
    if message:
        log_msg += f" - {message}"
    
    if success:
        logger.debug(log_msg)
    else:
        logger.warning(log_msg)


def handle_data_save_offline(instance, field_updates: Dict):
    """
    Handle data save operation when offline.
    
    Args:
        instance: The Django model instance being saved
        field_updates: Dictionary of fields being updated
    
    Returns:
        Dict: Status information
    """
    # Mark record as not synced
    if hasattr(instance, "is_synced"):
        instance.is_synced = False
    
    try:
        instance.save()
        log_sync_event(
            "save_offline",
            instance.__class__.__name__,
            instance.pk,
            success=True,
            message="Saved locally, marked for sync",
        )
        return {
            "success": True,
            "message": "Data saved locally. Will sync when online.",
            "is_synced": getattr(instance, "is_synced", True),
        }
    except Exception as e:
        logger.error(f"Failed to save data offline: {e}")
        return {
            "success": False,
            "message": "Failed to save data locally",
            "error": str(e),
        }


def create_offline_notification(
    notification_type: str, title: str, message: str, details: Optional[Dict] = None
) -> Dict:
    """
    Create a notification for offline status changes.
    
    Args:
        notification_type: Type of notification ("offline", "online", "sync_complete", etc.)
        title: Notification title
        message: Notification message
        details: Additional details
    
    Returns:
        Dict: Notification data
    """
    return {
        "type": notification_type,
        "title": title,
        "message": message,
        "details": details or {},
        "dismissible": True,
        "priority": "info" if "online" in notification_type else "warning",
    }


def get_offline_status_badge() -> Dict:
    """
    Get a badge/indicator for displaying offline status in UI.
    
    Returns:
        Dict: Badge configuration
    """
    return {
        "visible": True,
        "label": "OFFLINE MODE",
        "color": "danger",  # Bootstrap color
        "icon": "offline",
        "tooltip": "System is operating in offline mode. Changes will sync when online.",
    }


def get_cloud_status_badge() -> Dict:
    """
    Get a badge/indicator for displaying cloud mode status in UI.
    
    Returns:
        Dict: Badge configuration
    """
    return {
        "visible": True,
        "label": "CLOUD SYNC ENABLED",
        "color": "success",  # Bootstrap color
        "icon": "cloud-check",
        "tooltip": "System is connected to cloud services.",
    }


class OfflineDataQueue:
    """
    Simple in-memory queue for tracking unsync­ed data operations.
    Useful for UI feedback and debugging.
    """
    
    def __init__(self, max_size: int = 1000):
        self.queue = []
        self.max_size = max_size
    
    def add_pending_operation(self, model_name: str, record_id: int, operation: str):
        """
        Add a pending operation to the queue.
        
        Args:
            model_name: Name of the model
            record_id: ID of the record
            operation: Type of operation ("create", "update", "delete")
        """
        item = {
            "model": model_name,
            "record_id": record_id,
            "operation": operation,
            "timestamp": datetime.now().isoformat(),
        }
        
        self.queue.append(item)
        
        # Keep queue bounded
        if len(self.queue) > self.max_size:
            self.queue = self.queue[-self.max_size :]
    
    def get_pending_operations(self) -> list:
        """Get all pending operations."""
        return self.queue.copy()
    
    def clear_pending_operations(self):
        """Clear the pending operations queue."""
        self.queue.clear()


# Global instance
offline_data_queue = OfflineDataQueue()


from datetime import datetime
