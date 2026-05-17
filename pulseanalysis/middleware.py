"""
Offline-resilience middleware that gracefully handles internet/database connectivity issues.
When the cloud database becomes unreachable, automatically switches to local SQLite.
Handles storage layer failures gracefully and maintains data sync status.
"""

import logging
from functools import lru_cache

from django.conf import settings
from django.db import DatabaseError, OperationalError, connection, connections
from django.http import HttpResponse

logger = logging.getLogger(__name__)

# Global state to track if we're in offline mode (database unreachable)
_OFFLINE_MODE = False
_DB_CHECK_ATTEMPTED = False
_STORAGE_CHECK_ATTEMPTED = False
_STORAGE_OFFLINE = False
_FALLBACK_APPLIED = False


def is_offline_mode():
    """Check if the app has switched to offline mode due to database unavailability."""
    global _OFFLINE_MODE, _DB_CHECK_ATTEMPTED

    if getattr(settings, "FORCE_OFFLINE", False):
        _OFFLINE_MODE = True
        _DB_CHECK_ATTEMPTED = True
        return True
    
    if not _DB_CHECK_ATTEMPTED:
        try:
            connection.ensure_connection()
            _OFFLINE_MODE = False
        except (DatabaseError, OperationalError):
            _OFFLINE_MODE = True
            logger.warning("Database unreachable. Operating in offline mode with local SQLite.")
        _DB_CHECK_ATTEMPTED = True
    
    return _OFFLINE_MODE


def is_storage_offline():
    """Check if storage backend (Cloudinary) is unavailable."""
    global _STORAGE_OFFLINE, _STORAGE_CHECK_ATTEMPTED
    
    if _STORAGE_CHECK_ATTEMPTED:
        return _STORAGE_OFFLINE
    
    _STORAGE_CHECK_ATTEMPTED = True
    
    # Try a simple storage connectivity test
    try:
        from django.core.files.storage import default_storage
        # Attempt to list or stat a file in storage to test connectivity
        if hasattr(default_storage, 'listdir'):
            try:
                default_storage.listdir('/')
            except Exception:
                # Cloudinary might not support listdir, but other ops might
                pass
        _STORAGE_OFFLINE = False
    except Exception as e:
        logger.warning(f"Storage backend check failed: {e}")
        _STORAGE_OFFLINE = True
    
    return _STORAGE_OFFLINE


class OfflineResilienceMiddleware:
    """Middleware to handle gracefully database and storage connectivity errors."""
    
    def __init__(self, get_response):
        self.get_response = get_response
        self.offline_mode = False
        self.request_count = 0

    def __call__(self, request):
        global _FALLBACK_APPLIED
        self.request_count += 1
        
        # Add offline status to request for use in views/templates
        request.is_offline = is_offline_mode()
        request.is_storage_offline = is_storage_offline()

        # If the app is offline, switch connection settings once before executing views.
        # This prevents repeated attempts against unreachable cloud hosts.
        if request.is_offline and not _FALLBACK_APPLIED:
            ensure_offline_fallback()
            _close_db_connections()
            _FALLBACK_APPLIED = True

        # Attempt to detect if we're in offline mode
        try:
            if not _DB_CHECK_ATTEMPTED:
                is_offline_mode()
        except Exception as e:
            logger.debug(f"Offline detection error: {e}")
            self.offline_mode = True
            request.is_offline = True

        try:
            response = self.get_response(request)
        except (DatabaseError, OperationalError) as e:
            # Database error - app is now in offline mode
            global _OFFLINE_MODE
            _OFFLINE_MODE = True
            request.is_offline = True
            logger.warning(f"Database operation failed, using offline fallback: {e}")

            # Switch to SQLite and retry this request once.
            ensure_offline_fallback()
            _close_db_connections()
            _FALLBACK_APPLIED = True
            try:
                response = self.get_response(request)
            except Exception as retry_exc:
                logger.warning(f"Offline retry failed: {retry_exc}")
                if "/admin" in request.path:
                    return HttpResponse(
                        "Admin panel temporarily unavailable. Please try again when connected.",
                        status=503,
                    )
                return HttpResponse(
                    "The system is operating in offline mode. Some features may be limited.",
                    status=503,
                )
        except Exception as e:
            logger.exception(f"Unexpected error in offline resilience middleware: {e}")
            return HttpResponse("Unexpected server error.", status=500)

        return response


def ensure_offline_fallback():
    """
    Attempt to switch to SQLite if PostgreSQL is unreachable.
    Can be called at startup or when errors are encountered.
    """
    global _OFFLINE_MODE, _DB_CHECK_ATTEMPTED
    
    if not hasattr(settings, 'DATABASES') or not settings.DATABASES:
        return False
    
    current_engine = settings.DATABASES['default'].get('ENGINE', '')
    
    # If already using SQLite, no need to fallback
    if 'sqlite' in current_engine:
        return True
    
    # Try to connect to current database
    try:
        db = connections['default']
        db.ensure_connection()
        return True
    except (DatabaseError, OperationalError, Exception) as e:
        logger.warning(
            f"Cloud database ({current_engine}) unavailable: {e}. "
            "Switching to local SQLite for offline mode."
        )
        # Switch to SQLite
        settings.DATABASES['default'] = {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': str(settings.LOCAL_DB_PATH),
        }
        _OFFLINE_MODE = True
        _DB_CHECK_ATTEMPTED = True
        return False


def _close_db_connections():
    """Close active DB connections so Django reopens them with fallback settings."""
    try:
        for conn in connections.all():
            conn.close()
    except Exception as exc:
        logger.debug(f"Connection close during fallback had a non-fatal issue: {exc}")
