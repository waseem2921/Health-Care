# PulseAnalysis AI - Health Risk Prediction System (Django)

A Django-based Healthcare Analytics Web Application designed for pulse analysis, healthcare monitoring, dashboard visualization, and data-driven insights.

## Live Streaming 
https://preventive-healthcare-intelligence-system.onrender.com/

Admin Panel : Admin Password : Admin
User Panel: waseem Password : 1234

## Cloud Setup (NeonDB + Cloudinary)

This project is configured for a shared cloud database and cloud media storage.

1. Create a `.env` file in the project root.
2. Add the following environment variables:

```env
SECRET_KEY=replace-with-your-secret-key
DEBUG=False
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com

DATABASE_URL=postgresql://<user>:<password>@<host>/<dbname>?sslmode=require

CLOUDINARY_CLOUD_NAME=your-cloud-name
CLOUDINARY_API_KEY=your-api-key
CLOUDINARY_API_SECRET=your-api-secret
```

3. Install dependencies and run migrations:

```bash
pip install -r requirements.txt
python manage.py migrate
```

4. Seed shared starter data:

```bash
python manage.py seed_shared_data
```

5. Verify NeonDB + Cloudinary integration:

```bash
python manage.py verify_cloud_stack
```

6. Start the app:

```bash
python manage.py runserver 127.0.0.1:5000
```

All users now use the same NeonDB data, and uploaded media files are stored in Cloudinary.

### One-Click Run for Any User

Double-click [RUN_APP.bat](RUN_APP.bat) on Windows.

It automatically:
1. Creates/uses `.venv`
2. Installs dependencies (or continues offline if already installed)
3. Creates `.env` from `.env.example` if missing
4. Tries NeonDB first
5. Falls back to local offline DB when cloud DB is unreachable
6. Runs migrations and seed data
7. Starts the app at `http://127.0.0.1:5000`

For strict cloud-only mode, set `ALLOW_SQLITE_FALLBACK=False` and `REQUIRE_CLOUDINARY=True`.

## 🚀 Quick Start

Run the Django server:
```bash
python manage.py runserver 127.0.0.1:5000
```

The server starts at: **http://localhost:5000**

---

## 📋 Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🔐 Login Credentials

**Admin:**
- Username: `Admin`
- Password: `Admin`

**Users:** Register first at `/register`

---

## ⚠️ Important: Model Training Required

Before users can make predictions, you must:

1. **Login as Admin**
2. Go to **Preprocessing** (processes the dataset)
3. Go to **Train Models** (trains ML models)
4. Wait for training to complete

---

## 🎯 Features

- **User Predictions** with SHAP & LIME explanations
- **Admin Model Management** with accuracy metrics
- **Model Retraining** with new data
- **Performance Dashboard**

---

## 📁 Project Structure

```
├── manage.py              # Django entrypoint
├── pulseanalysis/         # Django project settings/urls
├── core/                  # Django application views
├── requirements.txt       # Dependencies
├── .env                  # Environment variables for NeonDB + Cloudinary
├── Dataset/              # Training dataset
├── Models/               # Trained ML models
├── templates/            # HTML templates
└── static/               # CSS, JS, images
```

---

## 🔧 Troubleshooting

**Port 5000 already in use?**
- Change run command: `python manage.py runserver 127.0.0.1:5001`

**Models not found?**
- Train models first via Admin panel

**Database errors?**
- Database auto-initializes on startup

## 🚀 Render Deployment

Use these settings on Render:

- Build command: `pip install -r requirements.txt && python manage.py collectstatic --noinput && python manage.py migrate`
- Start command: `gunicorn pulseanalysis.wsgi:application`

Set these environment variables:

```env
SECRET_KEY=your-production-secret-key
DEBUG=False
ALLOWED_HOSTS=your-service.onrender.com
DATABASE_URL=postgresql://<user>:<password>@<host>/<dbname>?sslmode=require
RENDER_EXTERNAL_HOSTNAME=your-service.onrender.com
```

If you use Cloudinary for uploads, also set:

```env
CLOUDINARY_CLOUD_NAME=your-cloud-name
CLOUDINARY_API_KEY=your-api-key
CLOUDINARY_API_SECRET=your-api-secret
```

---

**Last Updated:** 2025-01-05

