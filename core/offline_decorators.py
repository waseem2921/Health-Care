"""
Decorators and utilities for adding offline support to Django views.
Provides automatic error handling and graceful degradation for offline scenarios.
"""

import functools
import logging
from typing import Callable, Optional

from django.http import JsonResponse
from django.shortcuts import render

from core.offline_errors import get_error_response, handle_database_error, handle_storage_error

logger = logging.getLogger(__name__)


def offline_safe(
    json_response: bool = False,
    fallback_view: Optional[str] = None,
    allow_offline: bool = True,
):
    """
    Decorator to make a view offline-safe with automatic error handling.
    
    Args:
        json_response: If True, return JSON error responses (for APIs)
        fallback_view: Optional fallback template for offline mode
        allow_offline: If True, allow view to run in offline mode; if False, show error
    
    Usage:
        @offline_safe(json_response=False)
        def my_view(request):
            # View code
            pass
        
        @offline_safe(json_response=True)
        def my_api_view(request):
            # API view code
            pass
    """
    def decorator(view_func: Callable):
        @functools.wraps(view_func)
        def wrapper(request, *args, **kwargs):
            # Add offline status to request object
            request.is_offline = getattr(request, "is_offline", False)
            request.is_storage_offline = getattr(request, "is_storage_offline", False)
            
            # If offline and not allowed, show error
            if request.is_offline and not allow_offline:
                if json_response:
                    error_dict = handle_database_error(
                        Exception("System is offline"),
                        context=view_func.__name__,
                        can_retry=True,
                    )
                    return get_error_response(error_dict)
                else:
                    context = {
                        "error": "System is currently in offline mode.",
                        "message": "This feature is not available offline. Please try again when connected.",
                    }
                    return render(request, "error.html", context, status=503)
            
            try:
                response = view_func(request, *args, **kwargs)
                return response
            except Exception as e:
                logger.exception(f"Error in offline-safe view {view_func.__name__}: {e}")
                
                error_dict = handle_database_error(e, context=view_func.__name__)
                
                if json_response:
                    return get_error_response(error_dict)
                else:
                    context = {
                        "error": "An error occurred",
                        "message": error_dict.get("recovery", "Please try again later."),
                    }
                    return render(request, "error.html", context, status=503)
        
        return wrapper
    return decorator


def handle_storage_gracefully(fallback_method: Optional[Callable] = None):
    """
    Decorator to gracefully handle storage-related errors.
    Falls back to local storage if Cloudinary is unavailable.
    
    Args:
        fallback_method: Optional method to call if primary storage fails
    
    Usage:
        @handle_storage_gracefully()
        def upload_file(request):
            # File upload code
            pass
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(request, *args, **kwargs):
            try:
                return func(request, *args, **kwargs)
            except Exception as e:
                # Check if it's a storage-related error
                error_msg = str(e).lower()
                if any(
                    keyword in error_msg
                    for keyword in ["cloudinary", "storage", "upload", "connection", "timeout"]
                ):
                    logger.warning(f"Storage error in {func.__name__}: {e}")
                    
                    if fallback_method:
                        return fallback_method(request, *args, **kwargs)
                    
                    error_dict = handle_storage_error(e, context=func.__name__)
                    return JsonResponse(error_dict, status=503)
                else:
                    raise
        
        return wrapper
    return decorator


def sync_on_save(model_class):
    """
    Decorator to handle marking records as unsynced when in offline mode.
    Should wrap model save methods.
    
    Args:
        model_class: The Django model class
    
    Usage:
        @sync_on_save(MyModel)
        def save_data(request):
            instance = MyModel.objects.create(...)
            return response
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(request, *args, **kwargs):
            is_offline = getattr(request, "is_offline", False)
            
            # Execute the view
            response = func(request, *args, **kwargs)
            
            # If offline, mark any created/updated records as unsynced
            if is_offline and hasattr(model_class, "objects"):
                try:
                    # Mark recent records as unsynced
                    from datetime import timedelta
                    from django.utils import timezone
                    
                    recent_cutoff = timezone.now() - timedelta(minutes=1)
                    model_class.objects.filter(
                        created_at__gte=recent_cutoff,
                        is_synced=True
                    ).update(is_synced=False)
                    
                    logger.debug(
                        f"Marked recent {model_class.__name__} records as unsynced "
                        "(offline mode)"
                    )
                except Exception as e:
                    logger.warning(f"Could not mark records unsynced: {e}")
            
            return response
        
        return wrapper
    return decorator


def add_offline_context(template_name: str):
    """
    Decorator to add offline status context to template renders.
    
    Args:
        template_name: Name of template to render
    
    Usage:
        @add_offline_context("mytemplate.html")
        def my_view(request, context):
            return context
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(request, *args, **kwargs):
            context = func(request, *args, **kwargs) or {}
            
            # Add offline information
            context["is_offline"] = getattr(request, "is_offline", False)
            context["is_storage_offline"] = getattr(request, "is_storage_offline", False)
            context["offline_mode_active"] = (
                getattr(request, "is_offline", False) or getattr(request, "is_storage_offline", False)
            )
            
            if context.get("offline_mode_active"):
                context["warning_message"] = (
                    "System is operating in offline mode. "
                    "Your changes will sync when connectivity is restored."
                )
            
            return context
        
        return wrapper
    return decorator


def retry_on_database_error(max_retries: int = 3, delay: float = 0.1):
    """
    Decorator to retry a function if database errors occur.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
    
    Usage:
        @retry_on_database_error(max_retries=3)
        def database_operation():
            # Database code
            pass
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from django.db import DatabaseError, OperationalError
            import time
            
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (DatabaseError, OperationalError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Database error (attempt {attempt + 1}), retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"Database error after {max_retries} attempts: {e}")
            
            # Raise the last exception if all retries failed
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator
