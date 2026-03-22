from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, re_path
from django.views.static import serve

from core import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.home, name="home"),
    path("adminlogin", views.admin_login_page, name="adminlogin"),
    path("AdminAction", views.admin_action, name="AdminAction"),
    path("AdminHome", views.admin_home, name="AdminHome"),
    path("management-analytics", views.management_analytics_dashboard, name="management_analytics"),
    path("Upload", views.upload_page, name="Upload"),
    path("UploadAction", views.upload_action, name="UploadAction"),
    path("preprocess", views.preprocess_view, name="preprocess"),
    path("trainmodels", views.train_models, name="trainmodels"),
    path("comparison", views.comparison, name="comparison"),
    path("userlogin", views.user_login_page, name="userlogin"),
    path("login/patient", views.user_login_page, name="login_patient"),
    path("register", views.register_page, name="register"),
    path("RegAction", views.register_action, name="RegAction"),
    path("UserAction", views.user_action, name="UserAction"),
    path("ManageUsers", views.manage_users, name="ManageUsers"),
    path("delete_user/<str:username>", views.delete_user, name="delete_user"),
    path("Detect", views.detect_page, name="Detect"),
    path("UserHome", views.user_home, name="UserHome"),
    path("DetectAction", views.detect_action, name="DetectAction"),
    path("detect_action/", views.detect_action, name="detect_action"),
    path("connect-google-fit/", views.connect_google_fit, name="connect_google_fit"),
    path("google-fit-callback/", views.google_fit_callback, name="google_fit_callback"),
    path("fetch-google-fit/", views.fetch_google_fit_data, name="fetch_google_fit_data"),
    path("retrain", views.retrain, name="retrain"),
    path("model_performance", views.model_performance, name="model_performance"),
    path("login/doctor", views.doctor_login, name="login_doctor"),
    path("login/admin", views.admin_login_page, name="login_admin"),
    path("doctor/login/", views.doctor_login, name="doctor_login"),
    path("doctor/dashboard/", views.doctor_dashboard, name="doctor_dashboard"),
    path("doctor/patient/<int:patient_id>/", views.doctor_patient_detail, name="doctor_patient_detail"),
    path("doctor/logout/", views.doctor_logout, name="doctor_logout"),
    path("chatbot/query/", views.chatbot_query, name="chatbot_query"),
]

urlpatterns += static(settings.STATIC_URL, document_root=settings.BASE_DIR / "static")

# Keep local static/media assets available on runserver even with DEBUG=False.
# This prevents the UI from losing Bootstrap/theme styling in local/offline mode.
urlpatterns += [
    re_path(r"^static/(?P<path>.*)$", serve, {"document_root": settings.BASE_DIR / "static"}),
    re_path(r"^media/(?P<path>.*)$", serve, {"document_root": settings.BASE_DIR / "media"}),
]
