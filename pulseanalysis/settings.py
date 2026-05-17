import logging
import os
import re
import socket
from pathlib import Path

import dj_database_url
from django.core.exceptions import ImproperlyConfigured
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent


def _env_bool(name, default=False):
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name, default=""):
    raw_value = os.getenv(name, default)
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _staticfiles_backend():
    if DEBUG:
        return "django.contrib.staticfiles.storage.StaticFilesStorage"

    try:
        import whitenoise.storage  # noqa: F401
    except ImportError:
        return "django.contrib.staticfiles.storage.StaticFilesStorage"

    return "whitenoise.storage.CompressedManifestStaticFilesStorage"


SECRET_KEY = os.getenv("SECRET_KEY", "django-insecure-pulseanalysis-local-key")
DEBUG = _env_bool("DEBUG", False)
RENDER_EXTERNAL_HOSTNAME = os.getenv("RENDER_EXTERNAL_HOSTNAME", "").strip()
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "").strip()

# Build ALLOWED_HOSTS from env or sensible defaults and include render host when present
ALLOWED_HOSTS = _env_list("ALLOWED_HOSTS", "127.0.0.1,localhost")
if RENDER_EXTERNAL_HOSTNAME:
    ALLOWED_HOSTS.append(RENDER_EXTERNAL_HOSTNAME)
ALLOWED_HOSTS = list(dict.fromkeys(ALLOWED_HOSTS))

# CSRF trusted origins
CSRF_TRUSTED_ORIGINS = _env_list("CSRF_TRUSTED_ORIGINS", "")
if RENDER_EXTERNAL_HOSTNAME:
    CSRF_TRUSTED_ORIGINS.extend([
        f"https://{RENDER_EXTERNAL_HOSTNAME}",
        f"http://{RENDER_EXTERNAL_HOSTNAME}",
    ])
if RENDER_EXTERNAL_URL:
    CSRF_TRUSTED_ORIGINS.append(RENDER_EXTERNAL_URL)
CSRF_TRUSTED_ORIGINS = list(dict.fromkeys(CSRF_TRUSTED_ORIGINS))

USE_X_FORWARDED_HOST = _env_bool("USE_X_FORWARDED_HOST", True)
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SESSION_COOKIE_SECURE = _env_bool("SESSION_COOKIE_SECURE", not DEBUG)
CSRF_COOKIE_SECURE = _env_bool("CSRF_COOKIE_SECURE", not DEBUG)
# ────────────────────────────────────────────────────────────────────────────────
# Offline-First Configuration
# ────────────────────────────────────────────────────────────────────────────────
# Offline mode: when True, always use local SQLite. When False, try cloud DB with fallback.
USE_LOCAL_DB = _env_bool("USE_LOCAL_DB", False)
# Always allow SQLite fallback for seamless offline operation
ALLOW_SQLITE_FALLBACK = _env_bool("ALLOW_SQLITE_FALLBACK", True)
# Force offline mode regardless of connectivity
FORCE_OFFLINE = _env_bool("FORCE_OFFLINE", False)
# Use local storage even if Cloudinary is available
USE_LOCAL_STORAGE = _env_bool("USE_LOCAL_STORAGE", False)
# FORCE_OFFLINE must force both DB and storage to local mode.
if FORCE_OFFLINE:
    USE_LOCAL_DB = True
    USE_LOCAL_STORAGE = True
# Path to local offline database
LOCAL_DB_PATH = os.getenv("LOCAL_DB_PATH", str(BASE_DIR / "local_offline.db"))
# Maximum time to cache connectivity checks (seconds)
CONNECTIVITY_CACHE_DURATION = int(os.getenv("CONNECTIVITY_CACHE_DURATION", "30"))

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "core",
    "cloudinary",
    "cloudinary_storage",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "pulseanalysis.middleware.OfflineResilienceMiddleware",
]

try:
    import whitenoise.middleware  # noqa: F401

    MIDDLEWARE.insert(1, "whitenoise.middleware.WhiteNoiseMiddleware")
except ImportError:
    pass

ROOT_URLCONF = "pulseanalysis.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
    {
        "BACKEND": "django.template.backends.jinja2.Jinja2",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "environment": "pulseanalysis.jinja2.environment",
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "pulseanalysis.wsgi.application"
ASGI_APPLICATION = "pulseanalysis.asgi.application"
raw_database_url = os.getenv("DATABASE_URL", "").strip()
DATABASE_URL = raw_database_url.strip('"').strip("'")
embedded_match = re.search(r"(postgres(?:ql)?://[^\s'\"]+)", DATABASE_URL)
if embedded_match:
    DATABASE_URL = embedded_match.group(1)

DATABASE_MODE = "sqlite"

if USE_LOCAL_DB:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": LOCAL_DB_PATH,
        }
    }
else:
    database_url_valid = bool(DATABASE_URL) and DATABASE_URL not in {"://", "postgresql://", "postgres://"} and "://" in DATABASE_URL

    if database_url_valid:
        try:
            DATABASES = {
                "default": dj_database_url.parse(
                    DATABASE_URL,
                    conn_max_age=600,
                    ssl_require=True,
                )
            }
            DATABASES["default"]["CONN_HEALTH_CHECKS"] = True
            DATABASE_MODE = "postgres"
        except Exception as exc:
            if not ALLOW_SQLITE_FALLBACK:
                raise ImproperlyConfigured(f"Invalid DATABASE_URL configuration: {exc}") from exc
            DATABASES = {
                "default": {
                    "ENGINE": "django.db.backends.sqlite3",
                    "NAME": LOCAL_DB_PATH,
                }
            }
    else:
        if not ALLOW_SQLITE_FALLBACK:
            raise ImproperlyConfigured(
                "DATABASE_URL is missing/malformed and ALLOW_SQLITE_FALLBACK is disabled."
            )
        DATABASES = {
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": LOCAL_DB_PATH,
            }
        }

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

CLOUDINARY_STORAGE = {
    "CLOUD_NAME": os.getenv("CLOUDINARY_CLOUD_NAME", ""),
    "API_KEY": os.getenv("CLOUDINARY_API_KEY", ""),
    "API_SECRET": os.getenv("CLOUDINARY_API_SECRET", ""),
}

if all(CLOUDINARY_STORAGE.values()) and not os.getenv("CLOUDINARY_URL"):
    os.environ["CLOUDINARY_URL"] = (
        f"cloudinary://{CLOUDINARY_STORAGE['API_KEY']}:{CLOUDINARY_STORAGE['API_SECRET']}@{CLOUDINARY_STORAGE['CLOUD_NAME']}"
    )

cloudinary_available = all(CLOUDINARY_STORAGE.values())
require_cloudinary = _env_bool("REQUIRE_CLOUDINARY", False)

if require_cloudinary and not cloudinary_available:
    raise ImproperlyConfigured(
        "REQUIRE_CLOUDINARY is enabled but Cloudinary credentials are missing."
    )

# Configure storage: use Cloudinary if available and not forced to local, otherwise use local filesystem
if cloudinary_available and not USE_LOCAL_STORAGE and not FORCE_OFFLINE:
    STORAGES = {
        "default": {
            "BACKEND": "cloudinary_storage.storage.MediaCloudinaryStorage",
        },
        "staticfiles": {
            "BACKEND": _staticfiles_backend(),
        },
    }
    STORAGE_MODE = "cloudinary"
else:
    STORAGES = {
        "default": {
            "BACKEND": "django.core.files.storage.FileSystemStorage",
        },
        "staticfiles": {
            "BACKEND": _staticfiles_backend(),
        },
    }
    STORAGE_MODE = "local"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ────────────────────────────────────────────────────────────────────────────────
# Offline-Resilience Initialization
# ────────────────────────────────────────────────────────────────────────────────
# On startup, test cloud database connectivity. If unreachable, auto-switch to SQLite.
if not USE_LOCAL_DB and database_url_valid:
    try:
        # Quick connectivity test to database host
        db_host = DATABASES['default'].get('HOST', '')
        if db_host:
            socket.create_connection((db_host, 5432), timeout=3)
            logger.info(f"✅ Cloud database reachable: {db_host}")
    except Exception as e:
        logger.warning(
            f"Cloud database unreachable at startup: {str(e)[:50]}. "
            "Switching to offline SQLite mode."
        )
        DATABASES['default'] = {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': LOCAL_DB_PATH,
        }

# Ensure local media directory exists
try:
    (BASE_DIR / "media").mkdir(exist_ok=True)
    logger.debug("Ensured /media directory exists for offline storage")
except Exception as e:
    logger.warning(f"Could not create media directory: {e}")

# Log startup mode
try:
    from core.offline_utils import log_startup_mode
    log_startup_mode()
except Exception as e:
    logger.debug(f"Could not initialize offline utils logging: {e}")


# ────────────────────────────────────────────────────────────────────────────────
# Logging Configuration for Offline-First System
# ────────────────────────────────────────────────────────────────────────────────
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "[{levelname}] {asctime} {module} {funcName}:{lineno} - {message}",
            "style": "{",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {
            "format": "{levelname} {message}",
            "style": "{",
        },
        "offline_format": {
            "format": "[{levelname}] {asctime} [OFFLINE] {name} - {message}",
            "style": "{",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "filters": {
        "require_offline": {
            "()": "django.utils.log.RequireDebugTrue",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "verbose",
            "filename": str(BASE_DIR / "logs" / "pulseanalysis.log"),
            "maxBytes": 1024 * 1024 * 10,  # 10 MB
            "backupCount": 10,
        },
        "offline_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "offline_format",
            "filename": str(BASE_DIR / "logs" / "offline_operations.log"),
            "maxBytes": 1024 * 1024 * 5,  # 5 MB
            "backupCount": 5,
        },
        "sync_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "verbose",
            "filename": str(BASE_DIR / "logs" / "sync_operations.log"),
            "maxBytes": 1024 * 1024 * 5,  # 5 MB
            "backupCount": 5,
        },
    },
    "loggers": {
        "django": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False,
        },
        "django.db": {
            "handlers": ["file"],
            "level": "WARNING",
            "propagate": False,
        },
        "django.request": {
            "handlers": ["console", "file"],
            "level": "WARNING",
            "propagate": False,
        },
        "core": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "core.offline_utils": {
            "handlers": ["console", "offline_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "core.offline_decorators": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "core.management.commands.sync_to_cloud": {
            "handlers": ["console", "sync_file"],
            "level": "INFO",
            "propagate": False,
        },
        "pulseanalysis": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "pulseanalysis.middleware": {
            "handlers": ["console", "offline_file"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}

# Ensure logs directory exists
try:
    logs_dir = BASE_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)
except Exception as e:
    logger.warning(f"Could not create logs directory: {e}")
