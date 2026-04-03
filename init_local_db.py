#!/usr/bin/env python
"""Initialize local SQLite database with all migrations."""
import os
import sys
import django
from pathlib import Path

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pulseanalysis.settings')
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
django.setup()

from django.core.management import call_command
from django.conf import settings
from django.db import connections

# Get the local_offline.db path
local_db_path = getattr(settings, "LOCAL_DB_PATH", settings.BASE_DIR / "local_offline.db")

# Add the sync_local database to DATABASES if not present
if 'sync_local' not in settings.DATABASES:
    settings.DATABASES['sync_local'] = {
        "ATOMIC_REQUESTS": False,
        "AUTOCOMMIT": True,
        "CONN_HEALTH_CHECKS": False,
        "CONN_MAX_AGE": 0,
        "ENGINE": "django.db.backends.sqlite3",
        "HOST": "",
        "NAME": str(local_db_path),
        "OPTIONS": {},
        "PASSWORD": "",
        "PORT": "",
        "TIME_ZONE": None,
        "USER": "",
        "TEST": {
            "CHARSET": None,
            "COLLATION": None,
            "MIGRATE": True,
        },
    }

print(f"📦 Initializing local database at: {local_db_path}\n")

# Ensure connection is closed and fresh
if 'sync_local' in connections:
    connections['sync_local'].close()

# Run migrations on sync_local
try:
    call_command('migrate', database='sync_local', verbosity=2)
    print("\n✅ Local database initialized successfully!")
except Exception as e:
    print(f"\n❌ Error initializing local database: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

