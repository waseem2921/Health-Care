import glob
import json
import os
import re
from datetime import date, datetime, timedelta
import time
from collections import Counter, defaultdict
from typing import Any, Dict, cast
from pathlib import Path
from urllib.parse import urlparse

import joblib
import lime
import lime.lime_tabular
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
import requests
from django.conf import settings
from django.db import OperationalError, ProgrammingError, connection, transaction
from django.db.models import Avg, Count, OuterRef, Q, Subquery
from django.db.models.functions import Coalesce, TruncDate
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from google_auth_oauthlib.flow import Flow
from xgboost import data

from .models import ChatbotInteraction, Doctor, DoctorNote, HealthMetrics, HealthTrend, Patient, Prediction
from .offline_utils import is_online


def _resolve_google_client_secret_file():
    for file_name in ("client_secret.json", "client-secret.json"):
        candidate = BASE_DIR / file_name
        if candidate.exists():
            return str(candidate)
    return None


def _load_registered_redirect_uris(client_secret_file):
    try:
        with open(client_secret_file, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        web_cfg = payload.get("web", {})
        redirect_uris = web_cfg.get("redirect_uris", [])
        return [uri.rstrip("/") + "/" for uri in redirect_uris]
    except Exception:
        return []
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
#tells Matplotlib to run without a GUI.
matplotlib.use("Agg") #Agg  allows image generation without screen rendering

BASE_DIR = Path(settings.BASE_DIR) #Base Project Directory : This gets the root directory of your Django project.
UPLOAD_FOLDER = BASE_DIR / "uploads" #Folder Paths : These define where uploaded files, trained models, static assets, and datasets will be stored within your project structure.
MODEL_FOLDER = BASE_DIR / "Models" #Models Folder : This is where the trained machine learning models and related artifacts will be saved.
STATIC_FOLDER = BASE_DIR / "static" #Static Folder : This is where static assets like CSS, JavaScript, and images will be stored.
DATASET_FOLDER = BASE_DIR / "Dataset" #Dataset Folder : This is where your datasets will be stored.

#Auto Creating Folders This loop ensures the folders exist automatically.
for folder in (UPLOAD_FOLDER, MODEL_FOLDER, STATIC_FOLDER, DATASET_FOLDER):
    folder.mkdir(parents=True, exist_ok=True)

#the input features used to train the model.
FEATURE_COLUMNS = [
    "age",
    "gender",
    "resting_heart_rate",
    #"heart_rate_variability",
    #"pulse_amplitude",
    "stress_level",
    "sleep_duration_hours",
    "sleep_quality_score",
    "steps_per_day",
    "calories_burned",
    "blood_oxygen_level",
    "activity_level",
]
#Target Column (What the Model Predicts) : This is the output label the model tries to predict.
TARGET_COLUMN = "lifestyle_disorder_risk"
#Categorical Columns: These features contain text categories, not numbers.
CATEGORICAL_COLUMNS = ["gender", "activity_level"]
#These are continuous numerical features.
NUMERIC_COLUMNS = [
    "age",
    "resting_heart_rate",
    #"heart_rate_variability",
    #"pulse_amplitude",
    "stress_level",
    "sleep_duration_hours",
    "sleep_quality_score",
    "steps_per_day",
    "calories_burned",
    "blood_oxygen_level",
]
REQUIRED_FIELDS = FEATURE_COLUMNS.copy()
GOOGLE_FIT_SCOPES = [
    "https://www.googleapis.com/auth/fitness.activity.read",
    "https://www.googleapis.com/auth/fitness.heart_rate.read",
    "https://www.googleapis.com/auth/fitness.sleep.read",
]
#Model Accuracy Variables : These store accuracy results for different algorithms.
xgb_acc = None
rf_acc = None
dec_acc = None
#Dataset Variables : These will hold the dataset and the training/testing splits in memory.
dataset = None
filepath = None
#Train/Test Split Variables : These will hold the features and labels for training and testing the machine learning models.
X_train = None
X_test = None
y_train = None
y_test = None


def render_jinja(request, template_name, context=None):
    return render(request, template_name, context or {}, using="jinja2")


BASELINES = {
    "steps": 8000,
    "sleep_hours": 7,
    "sleep_quality": 7,
    "heart_rate": 72,
    "stress_level": 3,
    "blood_oxygen": 97,
    "calories_burned": 2000,
}

RISK_TYPE_TESTS = {
    "Cardiovascular Strain": ["ECG", "Blood Pressure Monitoring", "Lipid Profile"],
    "Stress-Sleep Dysregulation": ["Sleep Study", "Cortisol Test", "Autonomic Function Assessment"],
    "Metabolic Lifestyle Strain": ["HbA1c", "Fasting Blood Glucose", "Waist Circumference Review"],
    "Lifestyle Imbalance": ["Physical Activity Screening", "Body Composition Review", "Sleep Hygiene Assessment"],
}

GENERAL_PHYSICIAN_SPECIALIZATIONS = [
    "General Physician",
    "General Practitioner",
    "General Medicine",
]


def calculate_priority(score):
    if score >= 80:
        return Prediction.PRIORITY_HIGH
    if score >= 50:
        return Prediction.PRIORITY_MEDIUM
    return Prediction.PRIORITY_LOW


def suggest_disease(metrics):
    age = float(metrics.get("age") or 0)
    heart_rate = float(metrics.get("heart_rate") or 0)
    steps = float(metrics.get("steps") or 0)
    sleep_hours = float(metrics.get("sleep") or 0)
    sleep_quality = float(metrics.get("sleep_quality") or 0)
    stress_level = float(metrics.get("stress_level") or 0)
    blood_oxygen = float(metrics.get("blood_oxygen") or 0)
    calories = float(metrics.get("calories") or 0)

    disease_scores = {
        "Cardiovascular Strain": 0,
        "Stress-Sleep Dysregulation": 0,
        "Metabolic Lifestyle Strain": 0,
        "Lifestyle Imbalance": 0,
    }

    if heart_rate >= 100:
        disease_scores["Cardiovascular Strain"] += 4
        disease_scores["Stress-Sleep Dysregulation"] += 1
    elif heart_rate >= 90:
        disease_scores["Cardiovascular Strain"] += 3
        disease_scores["Lifestyle Imbalance"] += 1

    if steps < 4000:
        disease_scores["Metabolic Lifestyle Strain"] += 3
        disease_scores["Lifestyle Imbalance"] += 3
    elif steps < 7000:
        disease_scores["Metabolic Lifestyle Strain"] += 1
        disease_scores["Lifestyle Imbalance"] += 1

    if sleep_hours < 5.5:
        disease_scores["Stress-Sleep Dysregulation"] += 4
        disease_scores["Cardiovascular Strain"] += 2
    elif sleep_hours < 6.5:
        disease_scores["Stress-Sleep Dysregulation"] += 2

    if sleep_quality and sleep_quality <= 4:
        disease_scores["Stress-Sleep Dysregulation"] += 3
    elif sleep_quality and sleep_quality <= 6:
        disease_scores["Stress-Sleep Dysregulation"] += 1

    if stress_level >= 8:
        disease_scores["Stress-Sleep Dysregulation"] += 4
    elif stress_level >= 6:
        disease_scores["Stress-Sleep Dysregulation"] += 2

    if calories < 1400:
        disease_scores["Metabolic Lifestyle Strain"] += 2
    elif calories < 1800:
        disease_scores["Metabolic Lifestyle Strain"] += 1

    if blood_oxygen and blood_oxygen < 94:
        disease_scores["Cardiovascular Strain"] += 3
    elif blood_oxygen and blood_oxygen < 96:
        disease_scores["Cardiovascular Strain"] += 1

    if age >= 55:
        disease_scores["Cardiovascular Strain"] += 2
        disease_scores["Metabolic Lifestyle Strain"] += 2
    elif age >= 40:
        disease_scores["Cardiovascular Strain"] += 1
        disease_scores["Metabolic Lifestyle Strain"] += 1

    disease_name = max(disease_scores, key=lambda name: disease_scores[name])
    top_score = disease_scores[disease_name]

    if top_score >= 7:
        risk_level = "High"
    elif top_score >= 4:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    return {
        "disease_name": disease_name,
        "risk_level": risk_level,
        "suggested_tests": RISK_TYPE_TESTS[disease_name],
    }


def _metric_value(metrics, key):
    try:
        return float(metrics.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0


def _required_specialization_from_metrics(metrics):
    heart_rate = _metric_value(metrics, "heart_rate")
    sleep_hours = _metric_value(metrics, "sleep_hours")
    sleep_quality = _metric_value(metrics, "sleep_quality")
    steps = _metric_value(metrics, "steps")
    calories = _metric_value(metrics, "calories")
    stress_level = _metric_value(metrics, "stress_level")
    blood_oxygen = _metric_value(metrics, "blood_oxygen")
    age = _metric_value(metrics, "age")

    if heart_rate > 100 or blood_oxygen < 94:
        return ["Cardiologist"]

    if stress_level >= 7 and (sleep_hours < 6 or sleep_quality <= 5):
        return ["Neurologist", "Sleep Specialist", "Sleep Medicine"]

    if steps < 3500 and calories < 1700 and age > 35:
        return ["Endocrinologist"]

    return GENERAL_PHYSICIAN_SPECIALIZATIONS


def _find_balanced_doctor(specializations):
    query = Q()
    for specialization in specializations:
        query |= Q(specialization__iexact=specialization) | Q(specialization__icontains=specialization)

    candidate_qs = (
        Doctor.objects.filter(query)
        .annotate(current_patient_count=Count("auto_assigned_patients", distinct=True))
        .order_by("current_patient_count", "id")
    )

    doctor = candidate_qs.first()
    if doctor is not None:
        return doctor

    if specializations != GENERAL_PHYSICIAN_SPECIALIZATIONS:
        return _find_balanced_doctor(GENERAL_PHYSICIAN_SPECIALIZATIONS)

    return None


def _get_session_patient(request):
    patient_id = request.session.get("patient_id")
    if patient_id:
        patient = Patient.objects.select_related("doctor", "assigned_doctor").filter(pk=patient_id).first()
        if patient is not None:
            return patient

    username = request.session.get("username")
    if not username:
        return None
    return Patient.objects.select_related("doctor", "assigned_doctor").filter(username=username).first()


def _normalize_label(label):
    return str(label).strip().lower().replace("_", " ")


def _label_probability_map(model, probabilities, target_encoder):
    decoded_labels = target_encoder.inverse_transform(model.classes_.astype(int))
    return {decoded_label: float(probability) for decoded_label, probability in zip(decoded_labels, probabilities)}


def _safe_transform_label(encoder, raw_value):
    normalized_value = str(raw_value).strip()
    known_classes = [str(item) for item in encoder.classes_]
    if normalized_value in known_classes:
        return int(encoder.transform([normalized_value])[0])

    # Common fallback for gender values not present in the training set.
    if normalized_value.lower() == "other":
        for fallback in ("Female", "Male"):
            if fallback in known_classes:
                return int(encoder.transform([fallback])[0])

    return int(encoder.transform([known_classes[0]])[0])


def _get_probability(probability_map, keyword):
    for label, probability in probability_map.items():
        if keyword in _normalize_label(label):
            return probability
    return 0.0


def _calculate_risk_score(probability_map):
    high_probability = _get_probability(probability_map, "high")
    moderate_probability = _get_probability(probability_map, "moderate")
    low_probability = _get_probability(probability_map, "low")
    return round((high_probability * 100) + (moderate_probability * 60) + (low_probability * 20), 2)


def _prediction_badge(priority):
    return {
        Prediction.PRIORITY_HIGH: "danger",
        Prediction.PRIORITY_MEDIUM: "warning",
        Prediction.PRIORITY_LOW: "success",
    }.get(priority, "secondary")


def _prediction_recommendation(priority):
    if priority == Prediction.PRIORITY_HIGH:
        return "Immediate medical review is recommended. Focus on sleep recovery, stress control, and close monitoring of cardiovascular indicators."
    if priority == Prediction.PRIORITY_MEDIUM:
        return "A clinical follow-up is advisable. Improve sleep consistency, increase daily activity, and keep tracking vitals."
    return "Current risk is comparatively low. Maintain healthy activity, sleep, and routine monitoring habits."


def _extract_lime_feature_names(lime_list):
    feature_names = []
    for rule_text, _ in lime_list:
        normalized_rule = rule_text.lower()
        for feature_name in FEATURE_COLUMNS:
            if feature_name in normalized_rule and feature_name not in feature_names:
                feature_names.append(feature_name)
    return feature_names


def _feature_display_name(feature_name):
    return {
        "resting_heart_rate": "resting heart rate",
        "sleep_duration_hours": "sleep duration",
        "steps_per_day": "daily steps",
        "calories_burned": "calories burned",
        "blood_oxygen_level": "blood oxygen level",
        "stress_level": "stress level",
        "sleep_quality_score": "sleep quality",
        "activity_level": "activity level",
    }.get(feature_name, feature_name.replace("_", " "))


def _build_textual_explanation(raw_input, feature_importance, lime_list, risk_score, priority):
    insights = []

    try:
        sleep_value = float(raw_input.get("sleep_duration_hours") or 0)
        if sleep_value and sleep_value < BASELINES["sleep_hours"]:
            insights.append(f"low sleep duration by {BASELINES['sleep_hours'] - sleep_value:.1f} hours")
    except (TypeError, ValueError):
        pass

    try:
        heart_rate_value = float(raw_input.get("resting_heart_rate") or 0)
        if heart_rate_value and heart_rate_value > BASELINES["heart_rate"]:
            insights.append(f"high resting heart rate at {heart_rate_value:.0f} bpm")
    except (TypeError, ValueError):
        pass

    try:
        steps_value = float(raw_input.get("steps_per_day") or 0)
        if steps_value and steps_value < BASELINES["steps"]:
            insights.append(f"reduced daily activity with {steps_value:.0f} steps")
    except (TypeError, ValueError):
        pass

    try:
        oxygen_value = float(raw_input.get("blood_oxygen_level") or 0)
        if oxygen_value and oxygen_value < 95:
            insights.append(f"lower blood oxygen level at {oxygen_value:.1f}%")
    except (TypeError, ValueError):
        pass

    top_feature_names = []
    if feature_importance is not None and not feature_importance.empty:
        top_feature_names.extend(feature_importance.head(3)["Feature"].tolist())
    top_feature_names.extend(_extract_lime_feature_names(lime_list))

    seen = set()
    ranked_features = []
    for feature_name in top_feature_names:
        if feature_name not in seen:
            seen.add(feature_name)
            ranked_features.append(_feature_display_name(feature_name))

    if not insights and ranked_features:
        insights = ranked_features[:2]

    if insights:
        if len(insights) == 1:
            leading_text = insights[0]
        else:
            leading_text = ", ".join(insights[:-1]) + f" and {insights[-1]}"
    else:
        leading_text = "the combined lifestyle and wearable indicators"

    evidence_text = ""
    if ranked_features:
        evidence_text = f" SHAP and LIME highlighted {', '.join(ranked_features[:3])} as the most influential drivers."

    return (
        f"{leading_text.capitalize()} significantly influenced the patient's health risk. "
        f"The computed risk score is {risk_score:.1f}, which places the patient in the {priority.lower()} priority group."
        f"{evidence_text}"
    )


def _parse_float(value, default=0.0):
    try:
        if value in [None, "", "null"]:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _derive_data_source(watch_data, raw_input):
    watch_fields = {key for key, value in (watch_data or {}).items() if value not in [None, "", "null"]}
    if not watch_fields:
        return "Manual entry"

    manual_supported_fields = {field for field in REQUIRED_FIELDS if field not in watch_fields and raw_input.get(field) not in [None, "", "null"]}
    if manual_supported_fields:
        return "Wearable + manual"
    return "Wearable"


def _activity_irregularity_score(metrics, recent_metrics):
    steps = _parse_float(metrics.get("steps_per_day"))
    activity_level = str(metrics.get("activity_level") or "").strip().lower()
    irregularity = 0.0

    if recent_metrics:
        recent_steps = [metric.steps for metric in recent_metrics if metric.steps is not None]
        if len(recent_steps) >= 3:
            average_steps = float(np.mean(recent_steps))
            if average_steps:
                coefficient = float(np.std(recent_steps)) / average_steps
                irregularity = min(12.0, coefficient * 18.0)

    if activity_level == "high" and steps < 5000:
        irregularity = max(irregularity, 8.0)
    elif activity_level == "low" and steps > 11000:
        irregularity = max(irregularity, 6.0)

    return irregularity


def _analyze_lifestyle_disorder(metrics, recent_metrics=None, model_risk_score=None):
    stress_level = _parse_float(metrics.get("stress_level"))
    steps = _parse_float(metrics.get("steps_per_day"))
    sleep_hours = _parse_float(metrics.get("sleep_duration_hours"))
    sleep_quality = _parse_float(metrics.get("sleep_quality_score"))
    heart_rate = _parse_float(metrics.get("resting_heart_rate"))
    activity_level = str(metrics.get("activity_level") or "").strip().lower()

    stress_component = min(24.0, (stress_level / 10.0) * 24.0)
    activity_component = 0.0
    if steps < 3000:
        activity_component += 18.0
    elif steps < 5000:
        activity_component += 14.0
    elif steps < BASELINES["steps"]:
        activity_component += 8.0

    if activity_level == "low":
        activity_component += 4.0
    elif activity_level == "moderate":
        activity_component += 1.5

    sleep_component = 0.0
    if sleep_hours < 5:
        sleep_component += 12.0
    elif sleep_hours < 6.5:
        sleep_component += 8.0
    elif sleep_hours < BASELINES["sleep_hours"]:
        sleep_component += 4.0

    if sleep_quality <= 4:
        sleep_component += 10.0
    elif sleep_quality <= 6:
        sleep_component += 6.0
    elif sleep_quality < BASELINES["sleep_quality"]:
        sleep_component += 3.0

    heart_rate_component = 0.0
    if heart_rate >= 100:
        heart_rate_component = 18.0
    elif heart_rate >= 90:
        heart_rate_component = 12.0
    elif heart_rate > BASELINES["heart_rate"]:
        heart_rate_component = 6.0

    irregular_activity_component = _activity_irregularity_score(metrics, recent_metrics)
    rule_based_score = min(
        100.0,
        stress_component + activity_component + sleep_component + heart_rate_component + irregular_activity_component,
    )
    final_score = round(rule_based_score, 1)
    if model_risk_score is not None:
        final_score = round(min(100.0, (rule_based_score * 0.65) + (float(model_risk_score) * 0.35)), 1)

    factors = []
    if stress_level >= 6:
        factors.append("high stress")
    if activity_component >= 8:
        factors.append("low physical activity")
    if sleep_component >= 8:
        factors.append("poor sleep quality")
    if heart_rate_component >= 6:
        factors.append("elevated heart rate")
    if irregular_activity_component >= 6:
        factors.append("irregular activity patterns")

    if final_score >= 70:
        risk_level = Prediction.PRIORITY_HIGH
    elif final_score >= 40:
        risk_level = Prediction.PRIORITY_MEDIUM
    else:
        risk_level = Prediction.PRIORITY_LOW

    return {
        "risk_score": final_score,
        "risk_level": risk_level,
        "factors": factors,
        "components": {
            "stress": round(stress_component, 1),
            "activity": round(activity_component, 1),
            "sleep": round(sleep_component, 1),
            "heart_rate": round(heart_rate_component, 1),
            "irregularity": round(irregular_activity_component, 1),
        },
    }


def _nervous_system_recovery_insight(metrics, recent_metrics=None):
    heart_rate = _parse_float(metrics.get("resting_heart_rate"))
    stress_level = _parse_float(metrics.get("stress_level"))
    sleep_hours = _parse_float(metrics.get("sleep_duration_hours"))
    sleep_quality = _parse_float(metrics.get("sleep_quality_score"))
    recent_heart_rate = [metric.heart_rate for metric in recent_metrics or [] if metric.heart_rate is not None]

    recovery_score = 100.0
    recovery_score -= min(30.0, max(0.0, heart_rate - BASELINES["heart_rate"]) * 1.2)
    recovery_score -= min(28.0, stress_level * 2.8)
    recovery_score -= min(22.0, max(0.0, BASELINES["sleep_hours"] - sleep_hours) * 6.0)
    recovery_score -= min(18.0, max(0.0, BASELINES["sleep_quality"] - sleep_quality) * 3.5)

    if recent_heart_rate:
        recent_average = float(np.mean(recent_heart_rate))
        if heart_rate > recent_average + 5:
            recovery_score -= 6.0

    recovery_score = round(max(0.0, min(100.0, recovery_score)), 1)

    explanation_parts = []
    if stress_level >= 7:
        explanation_parts.append("frequent stress spikes")
    if sleep_hours < 6 or sleep_quality <= 5:
        explanation_parts.append("poor sleep recovery")
    if heart_rate >= 90:
        explanation_parts.append("elevated resting pulse")

    if explanation_parts:
        explanation = f"{' and '.join(explanation_parts).capitalize()} may indicate early lifestyle imbalance."
    elif recovery_score >= 75:
        explanation = "Recovery signals look stable with balanced pulse, stress, and sleep patterns."
    else:
        explanation = "Moderate autonomic strain is present. Continued monitoring of stress and sleep recovery is advisable."

    return {
        "score": recovery_score,
        "explanation": explanation,
    }


def _stress_sleep_interaction_insight(metrics, recent_metrics=None):
    stress_level = _parse_float(metrics.get("stress_level"))
    sleep_hours = _parse_float(metrics.get("sleep_duration_hours"))
    sleep_quality = _parse_float(metrics.get("sleep_quality_score"))
    recent_metrics = list(recent_metrics or [])
    matched_days = 0

    ordered_metrics = sorted(recent_metrics, key=lambda metric: metric.date)
    for index, metric in enumerate(ordered_metrics[:-1]):
        next_metric = ordered_metrics[index + 1]
        if (metric.stress_level or 0) >= 7 and ((next_metric.sleep_quality or 0) <= 5 or (next_metric.sleep_hours or 0) < 6):
            matched_days += 1

    if matched_days:
        patient_text = f"High-stress days were followed by poor sleep on {matched_days} occasion(s) in the recent trend."
        doctor_text = f"Stress-sleep coupling detected: {matched_days} high-stress day(s) were followed by short or poor-quality sleep."
    elif stress_level >= 7 and (sleep_hours < 6 or sleep_quality <= 5):
        patient_text = "Today's high stress and poor sleep suggest a developing stress-sleep imbalance."
        doctor_text = "Current reading suggests same-day stress-sleep dysregulation despite limited historical confirmation."
    else:
        patient_text = "Recent stress and sleep trends do not show a strong adverse interaction pattern."
        doctor_text = "No clear repeated stress-to-poor-sleep sequence was detected in the available history."

    return {
        "patient_text": patient_text,
        "doctor_text": doctor_text,
    }


def _build_ai_explanation(factors, feature_importance, lime_list, risk_score):
    factor_text = ", ".join(factors[:3]) if factors else "the combined wearable and lifestyle indicators"
    ranked_features = []
    if feature_importance is not None and not feature_importance.empty:
        ranked_features.extend(feature_importance.head(3)["Feature"].tolist())
    ranked_features.extend(_extract_lime_feature_names(lime_list))

    seen = []
    for feature_name in ranked_features:
        display_name = _feature_display_name(feature_name)
        if display_name not in seen:
            seen.append(display_name)

    feature_text = ", ".join(seen[:3]) if seen else "stress, activity, and sleep features"
    return f"{factor_text.capitalize()} contributed most to the elevated risk score of {risk_score:.1f}. SHAP and LIME highlighted {feature_text} as the strongest drivers."


def _build_lifestyle_recommendations(factors, recovery_score, risk_level):
    recommendations = []
    if "high stress" in factors:
        recommendations.append("Schedule short daily stress-reset intervals and reduce late-evening stimulation.")
    if "low physical activity" in factors or "irregular activity patterns" in factors:
        recommendations.append("Target consistent daily movement with a gradual step increase toward 8,000 steps.")
    if "poor sleep quality" in factors:
        recommendations.append("Stabilize sleep timing and protect a 7-hour sleep window for recovery.")
    if "elevated heart rate" in factors:
        recommendations.append("Limit caffeine overload and monitor resting pulse trends over the next week.")
    if recovery_score < 60:
        recommendations.append("Prioritize recovery for 72 hours before increasing exercise intensity.")
    if not recommendations:
        recommendations.append("Maintain current healthy routines and continue weekly wearable tracking.")
    if risk_level == Prediction.PRIORITY_HIGH:
        recommendations.append("Arrange a clinician follow-up soon if symptoms persist or trends worsen.")
    return recommendations[:4]


def _recommended_follow_up_tests(triage_result, factors, recovery_score):
    tests = list(triage_result.get("suggested_tests", []))
    if "elevated heart rate" in factors and "ECG" not in tests:
        tests.append("ECG")
    if "poor sleep quality" in factors and "Sleep Study" not in tests:
        tests.append("Sleep Study")
    if "high stress" in factors and "Cortisol Test" not in tests:
        tests.append("Cortisol Test")
    if recovery_score < 55 and "Autonomic Function Assessment" not in tests:
        tests.append("Autonomic Function Assessment")
    return tests[:5]


def _split_text_list(value):
    return [item.strip() for item in (value or "").split("|") if item.strip()]


def _build_trend_payload(metrics):
    ordered_metrics = list(metrics)
    chart_labels = [metric.date.strftime("%b %d") for metric in ordered_metrics]
    return {
        "chart_labels": chart_labels,
        "steps_values": [metric.steps for metric in ordered_metrics],
        "sleep_values": [metric.sleep_hours for metric in ordered_metrics],
        "heart_rate_values": [metric.heart_rate for metric in ordered_metrics],
        "calories_values": [metric.calories_burned for metric in ordered_metrics],
        "stress_values": [metric.stress_level for metric in ordered_metrics],
        "steps_baseline": [BASELINES["steps"] for _ in ordered_metrics],
        "sleep_baseline": [BASELINES["sleep_hours"] for _ in ordered_metrics],
        "heart_rate_baseline": [BASELINES["heart_rate"] for _ in ordered_metrics],
        "calories_baseline": [BASELINES["calories_burned"] for _ in ordered_metrics],
        "stress_baseline": [BASELINES["stress_level"] for _ in ordered_metrics],
    }


def _upsert_health_metric(
    patient,
    metric_date,
    steps,
    sleep_hours,
    heart_rate,
    calories_burned,
    stress_level=None,
    sleep_quality=None,
    blood_oxygen=None,
    activity_level="",
    data_source="Manual entry",
):
    defaults = {
        "steps": int(float(steps or 0)),
        "sleep_hours": _parse_float(sleep_hours) if sleep_hours not in [None, "", "null"] else None,
        "sleep_quality": _parse_float(sleep_quality) if sleep_quality not in [None, "", "null"] else None,
        "heart_rate": _parse_float(heart_rate) if heart_rate not in [None, "", "null"] else None,
        "stress_level": _parse_float(stress_level) if stress_level not in [None, "", "null"] else None,
        "blood_oxygen": _parse_float(blood_oxygen) if blood_oxygen not in [None, "", "null"] else None,
        "activity_level": str(activity_level or ""),
        "calories_burned": float(calories_burned or 0),
        "data_source": data_source,
    }
    HealthMetrics.objects.update_or_create(patient=patient, date=metric_date, defaults=defaults)


def _persist_patient_metrics(patient, raw_input, fit_daily, data_source):
    today = timezone.localdate()
    _upsert_health_metric(
        patient,
        today,
        raw_input.get("steps_per_day"),
        raw_input.get("sleep_duration_hours"),
        raw_input.get("resting_heart_rate"),
        raw_input.get("calories_burned"),
        raw_input.get("stress_level"),
        raw_input.get("sleep_quality_score"),
        raw_input.get("blood_oxygen_level"),
        raw_input.get("activity_level"),
        data_source,
    )

    day_dates = fit_daily.get("day_dates", [])
    daily_steps = fit_daily.get("daily_steps", [])
    daily_calories = fit_daily.get("daily_calories", [])
    daily_heart_rate = fit_daily.get("daily_heart_rate", [])

    for index, day_value in enumerate(day_dates):
        try:
            metric_date = date.fromisoformat(day_value)
        except ValueError:
            continue

        sleep_hours = raw_input.get("sleep_duration_hours") if metric_date == today else None
        _upsert_health_metric(
            patient,
            metric_date,
            daily_steps[index] if index < len(daily_steps) else 0,
            sleep_hours,
            daily_heart_rate[index] if index < len(daily_heart_rate) else None,
            daily_calories[index] if index < len(daily_calories) else 0,
            data_source="Wearable",
        )


def _build_local_fit_payload(patient):
    if patient is None:
        return None

    metrics = list(HealthMetrics.objects.filter(patient=patient).order_by("-date")[:7])
    if not metrics:
        return None

    metrics.reverse()

    day_labels = [metric.date.strftime("%b %d") for metric in metrics]
    day_dates = [metric.date.isoformat() for metric in metrics]
    daily_steps = [int(metric.steps or 0) for metric in metrics]
    daily_calories = [round(float(metric.calories_burned or 0)) for metric in metrics]
    daily_heart_rate = [round(float(metric.heart_rate), 1) if metric.heart_rate is not None else None for metric in metrics]
    daily_stress = [round(float(metric.stress_level), 1) if metric.stress_level is not None else None for metric in metrics]

    latest_metric = metrics[-1]
    avg_steps = round(sum(daily_steps) / len(daily_steps)) if daily_steps else 0
    avg_calories = round(sum(daily_calories) / len(daily_calories)) if daily_calories else 0
    latest_heart_rate = next((value for value in reversed(daily_heart_rate) if value is not None), None)
    latest_stress = next((metric.stress_level for metric in reversed(metrics) if metric.stress_level is not None), None)
    latest_sleep_hours = next((metric.sleep_hours for metric in reversed(metrics) if metric.sleep_hours is not None), None)
    latest_sleep_quality = next((metric.sleep_quality for metric in reversed(metrics) if metric.sleep_quality is not None), None)
    latest_blood_oxygen = next((metric.blood_oxygen for metric in reversed(metrics) if metric.blood_oxygen is not None), None)
    latest_activity_level = next((metric.activity_level for metric in reversed(metrics) if metric.activity_level), "")

    watch_data = {
        "age": patient.age if patient is not None else "",
        "gender": patient.gender if patient is not None else "",
        "resting_heart_rate": latest_heart_rate if latest_heart_rate is not None else "",
        "stress_level": latest_stress if latest_stress is not None else "",
        "sleep_duration_hours": latest_sleep_hours if latest_sleep_hours is not None else "",
        "sleep_quality_score": latest_sleep_quality if latest_sleep_quality is not None else "",
        "steps_per_day": avg_steps,
        "calories_burned": avg_calories,
        "blood_oxygen_level": latest_blood_oxygen if latest_blood_oxygen is not None else "",
        "activity_level": latest_activity_level,
    }

    return {
        "watch_data": watch_data,
        "fit_daily_data": {
            "day_labels": day_labels,
            "day_dates": day_dates,
            "daily_steps": daily_steps,
            "daily_calories": daily_calories,
            "daily_heart_rate": daily_heart_rate,
            "daily_stress": daily_stress,
        },
    }


def _prime_fit_session(request, fit_payload):
    request.session["fit_daily_data"] = fit_payload["fit_daily_data"]
    request.session["watch_data"] = fit_payload["watch_data"]
    return fit_payload["watch_data"]


def _metric_interpretation(label, average_value, baseline_value):
    if average_value is None:
        return f"Patient {label.lower()} data is not available for the last 7 days."

    delta = average_value - baseline_value
    if label == "Sleep":
        if delta < 0:
            return f"Patient sleep is below recommended level by {abs(delta):.1f} hours."
        if delta > 0:
            return f"Patient sleep is above the 7-hour baseline by {delta:.1f} hours."
        return "Patient sleep is aligned with the recommended baseline."

    if label == "Heart Rate":
        if delta > 0:
            return f"Patient heart rate is higher than average by {delta:.1f} bpm."
        if delta < 0:
            return f"Patient heart rate is lower than the baseline by {abs(delta):.1f} bpm."
        return "Patient heart rate is aligned with the healthy baseline."

    if delta < 0:
        return f"Patient {label.lower()} is below the baseline by {abs(delta):.1f}."
    if delta > 0:
        return f"Patient {label.lower()} is above the baseline by {delta:.1f}."
    return f"Patient {label.lower()} matches the baseline target."

#File Upload Function : This function saves uploaded files from Django forms.
def _save_uploaded_file(uploaded_file, target_path: Path):
    with target_path.open("wb+") as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)

#Function Definition and Global Variables : This function preprocesses the dataset by handling missing values, encoding categorical variables, scaling numeric features, and splitting the data into training and testing sets. It also saves the encoders, scaler, and training data for later use in predictions and model retraining. Additionally, it cleans up old explanation files to ensure that new explanations are generated fresh for each prediction. The function returns the total number of samples, as well as the counts for training and testing sets.
def preprocess_data():
    global dataset, X_train, X_test, y_train, y_test
    #Load Dataset : This reads the dataset from a CSV file, which is expected to be located in the Dataset folder. It also drops any rows with missing values to ensure clean data for training.
    dataset_path = DATASET_FOLDER / "lifestyle_disorder_wearable_dataset.csv"
    dataset = pd.read_csv(dataset_path)
    dataset.dropna(inplace=True)
    #Encode Categorical Columns : This loop iterates through the defined categorical columns, applies label encoding to convert text categories into numeric values, and saves the encoders for later use during prediction.
    label_encoders = {}
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        dataset[col] = pd.Series(le.fit_transform(dataset[col]), index=dataset.index)
        label_encoders[col] = le
    #Save Encoders : This saves the fitted label encoders to disk using joblib, allowing them to be loaded later for consistent encoding during prediction and retraining.
    joblib.dump(label_encoders, MODEL_FOLDER / "encoders.joblib")
    #Encode Target Variable : This encodes the target variable (the label the model will predict) using label encoding and saves the encoder for later use during prediction.
    target_encoder = LabelEncoder()
    dataset[TARGET_COLUMN] = pd.Series(
        target_encoder.fit_transform(dataset[TARGET_COLUMN]), index=dataset.index
    )
    joblib.dump(target_encoder, MODEL_FOLDER / "target_encoder.joblib")
    #Scale Numeric Features : This applies standard scaling to the numeric features to normalize their values, which can improve model performance. The fitted scaler is also saved for later use during prediction and retraining.
    scaler = StandardScaler()
    dataset[NUMERIC_COLUMNS] = scaler.fit_transform(dataset[NUMERIC_COLUMNS])
    joblib.dump(scaler, MODEL_FOLDER / "scaler.joblib")
    #Separate Features and Target
    X = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]
    #Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    #Save Full Training Data Cache
    X_train_full = dataset[FEATURE_COLUMNS]
    joblib.dump(X_train_full, MODEL_FOLDER / "X_train_full_cache.joblib")
    #Clear Old Explainability Files
    for pattern in [str(STATIC_FOLDER / "lime_user_*.html"), str(STATIC_FOLDER / "shap_*.png")]:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
            except OSError:
                pass
    #Return Dataset Statistics
    return len(X), len(X_train), len(X_test)

#Home Page View
def home(request):
    return render_jinja(request, "index.html", _build_health_trend_context())


def _risk_css_class(risk_level):
    return {
        "High": "risk-high",
        "Moderate": "risk-moderate",
        "Low": "risk-low",
    }.get(str(risk_level), "risk-low")


def _disease_icon(disease_name):
    disease_text = str(disease_name or "").lower()
    if "dengue" in disease_text:
        return "fa-solid fa-mosquito"
    if "tuberculosis" in disease_text or "respiratory" in disease_text:
        return "fa-solid fa-lungs"
    if "nipah" in disease_text:
        return "fa-solid fa-virus"
    if "heat" in disease_text:
        return "fa-solid fa-sun"
    if "viral" in disease_text or "flu" in disease_text:
        return "fa-solid fa-head-side-cough"
    return "fa-solid fa-shield-virus"


def _risk_rank_value(risk_level):
    return {"High": 3, "Moderate": 2, "Low": 1}.get(str(risk_level), 1)


def _build_health_trend_context():
    empty_context = {
        "trend_cards": [],
        "most_critical": None,
        "most_critical_icon": "fa-solid fa-triangle-exclamation",
        "smart_insights": ["Trending insights are temporarily unavailable in offline local mode."],
        "trend_timeline_labels_json": json.dumps([]),
        "trend_timeline_series_json": json.dumps([]),
        "disease_frequency_json": json.dumps([]),
        "risk_distribution_json": json.dumps([]),
    }

    try:
        table_names = connection.introspection.table_names()
        if HealthTrend._meta.db_table not in table_names:
            return empty_context
    except (OperationalError, ProgrammingError):
        return empty_context

    today = timezone.now().date()
    start_day = today - timedelta(days=6)
    try:
        trends_qs = HealthTrend.objects.filter(created_at__date__gte=start_day).order_by("-created_at")
    except (OperationalError, ProgrammingError):
        return empty_context

    if not trends_qs.exists():
        trends_qs = HealthTrend.objects.order_by("-created_at")

    latest_local_trends = list(trends_qs.filter(is_local=True)[:40])
    if not latest_local_trends:
        latest_local_trends = list(trends_qs[:40])

    unique_local_trends = []
    seen_diseases = set()
    for trend in latest_local_trends:
        normalized_disease = str(trend.disease_name or "").strip().lower()
        if normalized_disease in seen_diseases:
            continue
        seen_diseases.add(normalized_disease)
        unique_local_trends.append(trend)
        if len(unique_local_trends) >= 10:
            break

    trend_cards = []
    for trend in unique_local_trends:
        trend_cards.append(
            {
                "title": trend.title,
                "disease_name": trend.disease_name,
                "risk_level": trend.risk_level,
                "confidence_score": round(float(trend.confidence_score or 0.0), 1),
                "description": trend.description,
                "preventive_advice": trend.preventive_advice,
                "source_url": trend.source_url,
                "created_at": trend.created_at,
                "risk_class": _risk_css_class(trend.risk_level),
                "icon_class": _disease_icon(trend.disease_name),
            }
        )

    # Aggregates are based on records from the last 7 days for trend analytics.
    recent_qs = HealthTrend.objects.filter(created_at__date__gte=start_day)
    if not recent_qs.exists():
        recent_qs = trends_qs

    disease_counts = Counter(recent_qs.values_list("disease_name", flat=True))
    risk_counts = Counter(recent_qs.values_list("risk_level", flat=True))

    disease_frequency = [
        {"label": label, "count": count}
        for label, count in sorted(disease_counts.items(), key=lambda item: item[1], reverse=True)
    ][:6]
    risk_distribution = [
        {"label": "High", "count": int(risk_counts.get("High", 0))},
        {"label": "Moderate", "count": int(risk_counts.get("Moderate", 0))},
        {"label": "Low", "count": int(risk_counts.get("Low", 0))},
    ]

    labels = [(start_day + timedelta(days=offset)).isoformat() for offset in range(7)]
    top_diseases = [item["label"] for item in disease_frequency[:3]]
    timeline_rows = (
        recent_qs.annotate(day=TruncDate("created_at"))
        .values("day", "disease_name")
        .annotate(count=Count("id"))
        .order_by("day")
    )

    timeline_matrix = {disease: {day: 0 for day in labels} for disease in top_diseases}
    for row in timeline_rows:
        disease_name = row.get("disease_name")
        day_val = row.get("day")
        if disease_name not in timeline_matrix or day_val is None:
            continue
        timeline_matrix[disease_name][day_val.isoformat()] = int(row.get("count", 0))

    timeline_series = [
        {
            "disease": disease,
            "counts": [timeline_matrix[disease][day] for day in labels],
        }
        for disease in top_diseases
    ]

    most_critical = None
    ranked_recent = list(recent_qs)
    if ranked_recent:
        most_critical = sorted(
            ranked_recent,
            key=lambda trend: (
                _risk_rank_value(trend.risk_level),
                float(trend.confidence_score or 0.0),
                trend.created_at,
            ),
            reverse=True,
        )[0]

    smart_insights = []
    if disease_counts.get("Dengue", 0) > 0:
        smart_insights.append("Dengue cases trending upwards in recent health headlines.")
    if disease_counts.get("Heatwave / Heatstroke", 0) > 0:
        smart_insights.append("Heat-related risks increasing due to weather-related coverage.")
    if disease_counts.get("Viral Flu", 0) > 0:
        smart_insights.append("Viral and flu-like illness mentions are active in current trends.")
    if not smart_insights and recent_qs.exists():
        smart_insights.append("General health risk signals are active. Continue preventive monitoring.")

    return {
        "trend_cards": trend_cards,
        "most_critical": most_critical,
        "most_critical_icon": _disease_icon(most_critical.disease_name) if most_critical else "fa-solid fa-triangle-exclamation",
        "smart_insights": smart_insights,
        "trend_timeline_labels_json": json.dumps(labels),
        "trend_timeline_series_json": json.dumps(timeline_series),
        "disease_frequency_json": json.dumps(disease_frequency),
        "risk_distribution_json": json.dumps(risk_distribution),
    }

#Admin Login Page View : This view renders the admin login page when accessed. It does not perform any authentication logic itself; it simply serves the HTML template for the admin login interface.
def admin_login_page(request):
    return render_jinja(request, "AdminApp/AdminLogin.html")

#Admin Action View : This view handles the POST request from the admin login form. It checks the provided username and password against hardcoded credentials ("Admin" for both). If the credentials are correct, it renders the admin home page; otherwise, it re-renders the login page with an error message indicating that the login failed. If the request method is not POST, it redirects to the admin login page.
def admin_action(request):
    if request.method == "POST":
        username = request.POST.get("username", "")
        password = request.POST.get("password", "")

        if username == "Admin" and password == "Admin":
            return render_jinja(request, "AdminApp/AdminHome.html")
        return render_jinja(request, "AdminApp/AdminLogin.html", {"msg": "Login Failed..!!"})

    return redirect("adminlogin")

#Admin Home Page View : This view renders the admin home page when accessed. It serves the HTML template for the admin dashboard interface, allowing administrators to access various functionalities such as uploading datasets, training models, and managing users.
def admin_home(request):
    return render_jinja(request, "AdminApp/AdminHome.html")


def _safe_round(value, digits=2):
    if value is None:
        return None
    return round(float(value), digits)


def _is_wearable_source(source):
    source_text = str(source or "").lower()
    return any(keyword in source_text for keyword in ["wearable", "watch", "google fit"])


def _is_manual_source(source):
    source_text = str(source or "").lower()
    return "manual" in source_text


def _load_global_feature_importance_from_model():
    feature_labels = {
        "sleep_quality_score": "Sleep Quality",
        "stress_level": "Stress Level",
        "steps_per_day": "Daily Steps",
        "sleep_duration_hours": "Sleep Hours",
        "resting_heart_rate": "Heart Rate",
        "calories_burned": "Calories Burned",
        "blood_oxygen_level": "Blood Oxygen",
        "activity_level": "Activity Level",
        "age": "Age",
        "gender": "Gender",
    }
    for model_name in ["RFModel.joblib", "XGModel.joblib", "DecModel.joblib"]:
        model_path = MODEL_FOLDER / model_name
        if not model_path.exists():
            continue
        try:
            model = joblib.load(model_path)
        except Exception:
            continue
        if not hasattr(model, "feature_importances_"):
            continue
        ranked = sorted(
            zip(FEATURE_COLUMNS, list(model.feature_importances_)),
            key=lambda item: item[1],
            reverse=True,
        )
        top = []
        for feature_name, score in ranked[:6]:
            top.append(
                {
                    "feature": feature_labels.get(feature_name, feature_name.replace("_", " ").title()),
                    "score": _safe_round(score, 4),
                }
            )
        if top:
            return top
    return []


def _compute_lifestyle_score(metrics):
    """Compute a composite 0–100 Lifestyle Disorder Score from recent HealthMetrics records."""
    if not metrics:
        return {"score": None, "interpretation": "Insufficient data for lifestyle score computation.", "components": {}}

    recent = list(metrics[-7:]) if len(metrics) >= 7 else list(metrics)

    sq_vals = [float(m.sleep_quality) for m in recent if m.sleep_quality is not None]
    sl_vals = [float(m.sleep_hours) for m in recent if m.sleep_hours is not None]
    st_vals = [float(m.stress_level) for m in recent if m.stress_level is not None]
    hr_vals = [float(m.heart_rate) for m in recent if m.heart_rate is not None]
    step_vals = [float(m.steps) for m in recent]
    cal_vals = [float(m.calories_burned) for m in recent]

    avg_sq = sum(sq_vals) / len(sq_vals) if sq_vals else 6.0
    avg_sl = sum(sl_vals) / len(sl_vals) if sl_vals else 6.5
    avg_st = sum(st_vals) / len(st_vals) if st_vals else 5.0
    avg_hr = sum(hr_vals) / len(hr_vals) if hr_vals else 72.0
    avg_steps = sum(step_vals) / len(step_vals) if step_vals else 0.0
    avg_cal = sum(cal_vals) / len(cal_vals) if cal_vals else 1500.0

    sleep_quality_sc = min(100.0, (avg_sq / 10.0) * 100.0)
    sleep_hours_sc = min(100.0, (avg_sl / 8.0) * 100.0)
    stress_sc = max(0.0, 100.0 - (avg_st / 10.0) * 100.0)
    hr_sc = max(0.0, 100.0 - (abs(avg_hr - 72.0) / 72.0) * 150.0)
    steps_sc = min(100.0, (avg_steps / 8000.0) * 100.0)
    cal_sc = min(100.0, (avg_cal / 2000.0) * 100.0)

    score = int(round(
        0.20 * sleep_quality_sc
        + 0.15 * sleep_hours_sc
        + 0.25 * stress_sc
        + 0.15 * hr_sc
        + 0.15 * steps_sc
        + 0.10 * cal_sc
    ))

    low_factors = []
    if sleep_quality_sc < 60:
        low_factors.append("reduced sleep quality")
    if sleep_hours_sc < 60:
        low_factors.append("insufficient sleep hours")
    if stress_sc < 60:
        low_factors.append("increasing stress levels")
    if steps_sc < 60:
        low_factors.append("low physical activity")
    if hr_sc < 60:
        low_factors.append("irregular heart rate")

    if score >= 75:
        interpretation = "Good lifestyle balance — healthy activity, sleep quality, and stress management detected."
    elif score >= 50:
        factor_text = ", ".join(low_factors[:2]) if low_factors else "suboptimal metric balance"
        interpretation = f"Moderate lifestyle risk detected due to {factor_text}."
    else:
        factor_text = ", ".join(low_factors[:3]) if low_factors else "multiple lifestyle imbalances"
        interpretation = f"Elevated lifestyle risk — {factor_text} require immediate attention."

    return {
        "score": score,
        "interpretation": interpretation,
        "components": {
            "sleep_quality": int(round(sleep_quality_sc)),
            "sleep_hours": int(round(sleep_hours_sc)),
            "stress": int(round(stress_sc)),
            "heart_rate": int(round(hr_sc)),
            "activity": int(round(steps_sc)),
            "calories": int(round(cal_sc)),
        },
    }


def _generate_risk_projection(metrics, current_risk_score):
    """Analyse 7–14 day metric trends and project the 30-day lifestyle disorder risk."""
    if not metrics or current_risk_score is None:
        return None

    recent = list(metrics[-14:]) if len(metrics) >= 14 else list(metrics)
    if len(recent) < 3:
        return None

    def _linear_slope(values):
        n = len(values)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        numer = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denom = sum((i - x_mean) ** 2 for i in range(n))
        return numer / denom if denom else 0.0

    risk_delta = 0.0

    stress_vals = [float(m.stress_level) for m in recent if m.stress_level is not None]
    if len(stress_vals) >= 3:
        risk_delta += _linear_slope(stress_vals) * 2.5

    sleep_vals = [float(m.sleep_hours) for m in recent if m.sleep_hours is not None]
    if len(sleep_vals) >= 3:
        risk_delta -= _linear_slope(sleep_vals) * 2.0

    sq_vals = [float(m.sleep_quality) for m in recent if m.sleep_quality is not None]
    if len(sq_vals) >= 3:
        risk_delta -= _linear_slope(sq_vals) * 1.5

    step_vals = [float(m.steps) for m in recent]
    if len(step_vals) >= 3:
        risk_delta -= (_linear_slope(step_vals) / 1000.0) * 1.5

    projected = int(round(min(100.0, max(0.0, float(current_risk_score) + risk_delta * 30))))

    trend_parts = []
    if len(stress_vals) >= 3:
        s = _linear_slope(stress_vals)
        if s > 0.1:
            trend_parts.append("rising stress")
        elif s < -0.1:
            trend_parts.append("decreasing stress")
    if len(sleep_vals) >= 3:
        s = _linear_slope(sleep_vals)
        if s < -0.05:
            trend_parts.append("declining sleep")
        elif s > 0.05:
            trend_parts.append("improving sleep")
    if len(step_vals) >= 3:
        s = _linear_slope(step_vals)
        if s < -100:
            trend_parts.append("declining activity")
        elif s > 100:
            trend_parts.append("increasing activity")

    trend_description = ", ".join(trend_parts) if trend_parts else "stable recent habits"
    risk_change = projected - int(round(float(current_risk_score)))

    if risk_change > 5:
        trend_direction = "increasing"
    elif risk_change < -5:
        trend_direction = "decreasing"
    else:
        trend_direction = "stable"

    return {
        "current_risk": round(float(current_risk_score), 1),
        "projected_risk": projected,
        "trend_direction": trend_direction,
        "trend_description": trend_description,
        "risk_change": risk_change,
    }


def _generate_ai_insight_cards(metrics):
    """Auto-generate short health insight cards from weekly metric trends."""
    if not metrics or len(metrics) < 2:
        return []

    recent = list(metrics[-7:]) if len(metrics) >= 7 else list(metrics)
    prior = list(metrics[-14:-7]) if len(metrics) >= 14 else []

    cards = []

    def _pct_change(r_vals, p_vals):
        if not r_vals or not p_vals:
            return None
        r_avg = sum(r_vals) / len(r_vals)
        p_avg = sum(p_vals) / len(p_vals)
        if p_avg == 0:
            return None
        return (r_avg - p_avg) / abs(p_avg) * 100

    if prior:
        r_sleep = [float(m.sleep_hours) for m in recent if m.sleep_hours is not None]
        p_sleep = [float(m.sleep_hours) for m in prior if m.sleep_hours is not None]
        pct = _pct_change(r_sleep, p_sleep)
        if pct is not None:
            if pct <= -10:
                cards.append({"type": "warning", "icon": "fa-moon", "title": "Sleep Declined", "text": f"Sleep hours decreased {abs(pct):.0f}% this week compared to last week."})
            elif pct >= 10:
                cards.append({"type": "success", "icon": "fa-moon", "title": "Sleep Improved", "text": f"Sleep hours increased {pct:.0f}% this week."})

        r_steps = [float(m.steps) for m in recent]
        p_steps = [float(m.steps) for m in prior]
        pct = _pct_change(r_steps, p_steps)
        if pct is not None:
            if pct <= -15:
                cards.append({"type": "warning", "icon": "fa-walking", "title": "Activity Reduced", "text": f"Activity (steps) reduced {abs(pct):.0f}% this week."})
            elif pct >= 15:
                cards.append({"type": "success", "icon": "fa-walking", "title": "Activity Increased", "text": f"Walking activity improved {pct:.0f}% this week."})

        r_stress = [float(m.stress_level) for m in recent if m.stress_level is not None]
        p_stress = [float(m.stress_level) for m in prior if m.stress_level is not None]
        pct = _pct_change(r_stress, p_stress)
        if pct is not None:
            if pct >= 15:
                cards.append({"type": "danger", "icon": "fa-brain", "title": "Stress Increasing", "text": f"Stress level increased {pct:.0f}% this week."})
            elif pct <= -15:
                cards.append({"type": "success", "icon": "fa-brain", "title": "Stress Easing", "text": f"Stress level reduced {abs(pct):.0f}% this week."})

        r_sq = [float(m.sleep_quality) for m in recent if m.sleep_quality is not None]
        p_sq = [float(m.sleep_quality) for m in prior if m.sleep_quality is not None]
        pct = _pct_change(r_sq, p_sq)
        if pct is not None and pct <= -15:
            cards.append({"type": "warning", "icon": "fa-bed", "title": "Sleep Quality Falling", "text": f"Sleep quality dropped {abs(pct):.0f}% this week."})

    sq_vals = [float(m.sleep_quality) for m in recent if m.sleep_quality is not None]
    if sq_vals:
        avg_sq = sum(sq_vals) / len(sq_vals)
        if avg_sq < 5 and not any(c["title"] == "Sleep Quality Falling" for c in cards):
            cards.append({"type": "danger", "icon": "fa-bed", "title": "Poor Sleep Quality", "text": f"Average sleep quality is {avg_sq:.1f}/10. Improving rest could reduce lifestyle risk."})

    avg_steps_recent = sum(float(m.steps) for m in recent) / len(recent) if recent else 0.0
    if avg_steps_recent < 5000 and not any(c["title"] == "Activity Reduced" for c in cards):
        cards.append({"type": "warning", "icon": "fa-shoe-prints", "title": "Low Activity Alert", "text": f"Average steps ({avg_steps_recent:,.0f}/day) are well below the 8,000-step daily target."})

    warning_count = sum(1 for c in cards if c["type"] in ["warning", "danger"])
    if warning_count >= 2:
        cards.append({"type": "info", "icon": "fa-chart-line", "title": "Risk Outlook", "text": "Multiple declining health indicators detected this week. Lifestyle disorder risk is likely to increase without habit changes."})
    elif not cards:
        cards.append({"type": "success", "icon": "fa-check-circle", "title": "Healthy Trend", "text": "No significant declining patterns detected this week. Keep up your current healthy routines."})

    return cards[:5]


def _compute_population_risk_stats():
    """Aggregate population-level risk factor prevalence across all patients."""
    from django.db.models import Avg as DbAvg  # local alias to avoid shadowing outer import
    total_patients = Patient.objects.count()
    if total_patients == 0:
        return [], [], []

    patient_avgs = list(
        HealthMetrics.objects
        .values("patient_id")
        .annotate(
            avg_steps=DbAvg("steps"),
            avg_sleep=DbAvg("sleep_hours"),
            avg_stress=DbAvg("stress_level"),
            avg_sq=DbAvg("sleep_quality"),
            avg_hr=DbAvg("heart_rate"),
        )
    )

    counts: Dict[str, int] = {
        "Low Activity": 0,
        "Sleep Deprivation": 0,
        "High Stress": 0,
        "Poor Sleep Quality": 0,
        "Elevated Heart Rate": 0,
    }
    for row in patient_avgs:
        if row["avg_steps"] is not None and row["avg_steps"] < 5000:
            counts["Low Activity"] += 1
        if row["avg_sleep"] is not None and row["avg_sleep"] < 6.0:
            counts["Sleep Deprivation"] += 1
        if row["avg_stress"] is not None and row["avg_stress"] > 6.0:
            counts["High Stress"] += 1
        if row["avg_sq"] is not None and row["avg_sq"] < 5.0:
            counts["Poor Sleep Quality"] += 1
        if row["avg_hr"] is not None and row["avg_hr"] > 85:
            counts["Elevated Heart Rate"] += 1

    denominator = len(patient_avgs) if patient_avgs else total_patients
    rows = sorted(
        [{"factor": k, "count": v, "pct": _safe_round(v / denominator * 100, 1) if denominator else 0.0} for k, v in counts.items()],
        key=lambda r: int(r["count"]),  # type: ignore[arg-type]
        reverse=True,
    )
    return rows, [r["factor"] for r in rows], [r["pct"] for r in rows]


def management_analytics_dashboard(request):
    today = timezone.localdate()
    fortnight_ago = today - timedelta(days=13)

    latest_prediction_subquery = Prediction.objects.filter(patient=OuterRef("pk")).order_by("-created_at")
    latest_metric_subquery = HealthMetrics.objects.filter(patient=OuterRef("pk")).order_by("-date", "-id")

    patients = list(
        Patient.objects.select_related("doctor", "assigned_doctor")
        .annotate(
            latest_priority=Subquery(latest_prediction_subquery.values("priority")[:1]),
            latest_risk_score=Subquery(latest_prediction_subquery.values("risk_score")[:1]),
            latest_data_source=Subquery(latest_prediction_subquery.values("data_source")[:1]),
            latest_metric_source=Subquery(latest_metric_subquery.values("data_source")[:1]),
        )
    )
    doctors = list(Doctor.objects.all())
    predictions = Prediction.objects.select_related("patient", "patient__doctor", "patient__assigned_doctor")
    metrics = list(HealthMetrics.objects.select_related("patient", "patient__doctor", "patient__assigned_doctor"))

    risk_label_map = {
        Prediction.PRIORITY_LOW: "Low Risk",
        Prediction.PRIORITY_MEDIUM: "Moderate Risk",
        Prediction.PRIORITY_HIGH: "High Risk",
    }
    risk_order = ["Low Risk", "Moderate Risk", "High Risk"]

    hospital_stats: defaultdict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "patients": 0,
            "doctors": 0,
            "high_risk_cases": 0,
            "risk_distribution": {"Low Risk": 0, "Moderate Risk": 0, "High Risk": 0},
            "stress_sum": 0.0,
            "stress_count": 0,
            "sleep_hours_sum": 0.0,
            "sleep_hours_count": 0,
            "sleep_quality_sum": 0.0,
            "sleep_quality_count": 0,
        }
    )

    doctor_workload = {
        doctor.id: {
            "doctor": doctor,
            "hospital": doctor.hospital or "Unassigned Hospital",
            "patients_assigned": 0,
            "high_risk_patients": 0,
        }
        for doctor in doctors
    }

    system_risk_distribution = {"Low Risk": 0, "Moderate Risk": 0, "High Risk": 0}
    source_patient_counts = {"wearable": 0, "manual": 0, "mixed": 0, "unknown": 0}
    reliability_by_source = {
        "wearable": {"patients": 0, "high_risk": 0, "risk_sum": 0.0, "risk_count": 0},
        "manual": {"patients": 0, "high_risk": 0, "risk_sum": 0.0, "risk_count": 0},
        "mixed": {"patients": 0, "high_risk": 0, "risk_sum": 0.0, "risk_count": 0},
        "unknown": {"patients": 0, "high_risk": 0, "risk_sum": 0.0, "risk_count": 0},
    }

    for doctor in doctors:
        hospital_bucket = cast(Dict[str, Any], hospital_stats[doctor.hospital or "Unassigned Hospital"])
        hospital_bucket["doctors"] = int(hospital_bucket["doctors"]) + 1

    for patient in patients:
        primary_doctor = patient.assigned_doctor if patient.assigned_doctor is not None else patient.doctor
        hospital_name = (
            primary_doctor.hospital
            if primary_doctor is not None and primary_doctor.hospital
            else "Unassigned Hospital"
        )

        hospital_bucket = cast(Dict[str, Any], hospital_stats[hospital_name])
        hospital_bucket["patients"] = int(hospital_bucket["patients"]) + 1
        latest_priority = getattr(patient, "latest_priority", None)
        if latest_priority in risk_label_map:
            risk_label = risk_label_map[latest_priority]
            risk_distribution = cast(Dict[str, int], hospital_bucket["risk_distribution"])
            risk_distribution[risk_label] = int(risk_distribution.get(risk_label, 0)) + 1
            system_risk_distribution[risk_label] += 1
            if latest_priority == Prediction.PRIORITY_HIGH:
                hospital_bucket["high_risk_cases"] = int(hospital_bucket["high_risk_cases"]) + 1

        if primary_doctor is not None and primary_doctor.id in doctor_workload:
            doctor_workload[primary_doctor.id]["patients_assigned"] += 1
            if latest_priority == Prediction.PRIORITY_HIGH:
                doctor_workload[primary_doctor.id]["high_risk_patients"] += 1

        source_text = getattr(patient, "latest_data_source", None) or getattr(patient, "latest_metric_source", None)
        source_key = "unknown"
        if source_text:
            if _is_wearable_source(source_text) and _is_manual_source(source_text):
                source_key = "mixed"
            elif _is_wearable_source(source_text):
                source_key = "wearable"
            elif _is_manual_source(source_text):
                source_key = "manual"

        source_patient_counts[source_key] += 1
        reliability = reliability_by_source[source_key]
        reliability["patients"] += 1
        risk_score = getattr(patient, "latest_risk_score", None)
        if latest_priority == Prediction.PRIORITY_HIGH:
            reliability["high_risk"] += 1
        if risk_score is not None:
            reliability["risk_sum"] += float(risk_score)
            reliability["risk_count"] += 1

    activity_distribution = Counter()
    sedentary_keywords = {"sedentary", "low"}
    daily_totals = {fortnight_ago + timedelta(days=index): {"total": 0, "sedentary": 0} for index in range(14)}

    for metric in metrics:
        primary_doctor = metric.patient.assigned_doctor if metric.patient.assigned_doctor is not None else metric.patient.doctor
        hospital_name = (
            primary_doctor.hospital
            if primary_doctor is not None and primary_doctor.hospital
            else "Unassigned Hospital"
        )
        hospital_bucket = cast(Dict[str, Any], hospital_stats[hospital_name])
        if metric.stress_level is not None:
            hospital_bucket["stress_sum"] = float(hospital_bucket["stress_sum"]) + float(metric.stress_level)
            hospital_bucket["stress_count"] = int(hospital_bucket["stress_count"]) + 1
        if metric.sleep_hours is not None:
            hospital_bucket["sleep_hours_sum"] = float(hospital_bucket["sleep_hours_sum"]) + float(metric.sleep_hours)
            hospital_bucket["sleep_hours_count"] = int(hospital_bucket["sleep_hours_count"]) + 1
        if metric.sleep_quality is not None:
            hospital_bucket["sleep_quality_sum"] = float(hospital_bucket["sleep_quality_sum"]) + float(metric.sleep_quality)
            hospital_bucket["sleep_quality_count"] = int(hospital_bucket["sleep_quality_count"]) + 1

        activity_label = (metric.activity_level or "Unknown").strip() or "Unknown"
        activity_distribution[activity_label] += 1

        if metric.date in daily_totals:
            daily_totals[metric.date]["total"] += 1
            activity_norm = activity_label.lower()
            if activity_norm in sedentary_keywords or metric.steps < 5000:
                daily_totals[metric.date]["sedentary"] += 1

    system_sleep_stress = HealthMetrics.objects.aggregate(
        avg_stress=Avg("stress_level"),
        avg_sleep_hours=Avg("sleep_hours"),
        avg_sleep_quality=Avg("sleep_quality"),
        avg_steps=Avg("steps"),
        avg_calories=Avg("calories_burned"),
    )

    hospital_rows = []
    for hospital_name, values in sorted(
        hospital_stats.items(),
        key=lambda item: int(cast(Dict[str, Any], item[1])["high_risk_cases"]),
        reverse=True,
    ):
        hospital_rows.append(
            {
                "hospital": hospital_name,
                "patients": values["patients"],
                "doctors": values["doctors"],
                "high_risk_cases": values["high_risk_cases"],
                "risk_distribution": values["risk_distribution"],
                "avg_stress": _safe_round(values["stress_sum"] / values["stress_count"], 2) if values["stress_count"] else None,
                "avg_sleep_hours": _safe_round(values["sleep_hours_sum"] / values["sleep_hours_count"], 2) if values["sleep_hours_count"] else None,
                "avg_sleep_quality": _safe_round(values["sleep_quality_sum"] / values["sleep_quality_count"], 2) if values["sleep_quality_count"] else None,
            }
        )

    hospital_stress_sorted = [row for row in hospital_rows if row["avg_stress"] is not None and row["avg_sleep_quality"] is not None]
    hospital_stress_sorted.sort(key=lambda row: row["avg_stress"], reverse=True)

    cross_hospital_insight = "Not enough complete hospital sleep-stress data yet."
    if len(hospital_stress_sorted) >= 2:
        highest_stress = hospital_stress_sorted[0]
        lowest_stress = hospital_stress_sorted[-1]
        if highest_stress["avg_sleep_quality"] < lowest_stress["avg_sleep_quality"]:
            cross_hospital_insight = (
                f"Hospitals with higher stress averages show lower sleep quality: "
                f"{highest_stress['hospital']} stress {highest_stress['avg_stress']} vs sleep quality {highest_stress['avg_sleep_quality']}, "
                f"while {lowest_stress['hospital']} stress {lowest_stress['avg_stress']} vs sleep quality {lowest_stress['avg_sleep_quality']}."
            )
        else:
            cross_hospital_insight = (
                "Cross-hospital stress and sleep quality are mixed; monitor units with rising stress trends for early sleep-quality decline."
            )

    doctor_workload_rows = sorted(
        [
            {
                "doctor_name": payload["doctor"].name,
                "hospital": payload["hospital"],
                "patients_assigned": payload["patients_assigned"],
                "high_risk_patients": payload["high_risk_patients"],
            }
            for payload in doctor_workload.values()
        ],
        key=lambda row: (row["high_risk_patients"], row["patients_assigned"]),
        reverse=True,
    )

    hospital_workload_distribution = defaultdict(lambda: {"doctor_count": 0, "patient_load": 0})
    for row in doctor_workload_rows:
        hospital_workload_distribution[row["hospital"]]["doctor_count"] += 1
        hospital_workload_distribution[row["hospital"]]["patient_load"] += row["patients_assigned"]

    hospital_workload_rows = []
    for hospital_name, values in sorted(hospital_workload_distribution.items()):
        doctor_count = values["doctor_count"]
        patient_load = values["patient_load"]
        hospital_workload_rows.append(
            {
                "hospital": hospital_name,
                "doctor_count": doctor_count,
                "patient_load": patient_load,
                "avg_patients_per_doctor": _safe_round(patient_load / doctor_count, 2) if doctor_count else 0,
            }
        )

    total_predictions = predictions.count()
    high_risk_alerts = predictions.filter(priority=Prediction.PRIORITY_HIGH).count()
    lifestyle_warnings = predictions.filter(priority__in=[Prediction.PRIORITY_MEDIUM, Prediction.PRIORITY_HIGH]).count()

    factor_counter = Counter()
    for prediction in predictions:
        for factor in _split_text_list(prediction.risk_factors):
            normalized = factor.strip().lower()
            if normalized:
                factor_counter[normalized] += 1

    xai_feature_importance = [
        {
            "feature": key.replace("_", " ").title(),
            "score": value,
        }
        for key, value in factor_counter.most_common(6)
    ]
    if not xai_feature_importance:
        xai_feature_importance = _load_global_feature_importance_from_model()

    total_registered_patients = Patient.objects.count()
    active_patient_ids = set(HealthMetrics.objects.filter(date=today).values_list("patient_id", flat=True))
    active_patient_ids.update(Prediction.objects.filter(created_at__date=today).values_list("patient_id", flat=True))

    active_doctor_ids = set(DoctorNote.objects.filter(created_at__date=today).values_list("doctor_id", flat=True))
    for doctor_id, assigned_id in Prediction.objects.filter(created_at__date=today).values_list("patient__doctor_id", "patient__assigned_doctor_id"):
        if assigned_id:
            active_doctor_ids.add(assigned_id)
        elif doctor_id:
            active_doctor_ids.add(doctor_id)

    wearable_patient_ids = set(
        HealthMetrics.objects.filter(
            Q(data_source__icontains="wearable")
            | Q(data_source__icontains="watch")
            | Q(data_source__icontains="google fit")
        )
        .values_list("patient_id", flat=True)
        .distinct()
    )

    data_quality = {
        "missing_sleep_metrics": HealthMetrics.objects.filter(
            Q(sleep_hours__isnull=True) | Q(sleep_quality__isnull=True)
        ).count(),
        "missing_wearable_data": max(total_registered_patients - len(wearable_patient_ids), 0),
        "abnormal_heart_rate_values": HealthMetrics.objects.filter(
            Q(heart_rate__lt=40) | Q(heart_rate__gt=120)
        ).count(),
    }

    sedentary_dates = sorted(daily_totals.keys())
    sedentary_rate_series = []
    for day in sedentary_dates:
        day_total = daily_totals[day]["total"]
        if day_total == 0:
            sedentary_rate_series.append(0)
            continue
        sedentary_rate_series.append(_safe_round((daily_totals[day]["sedentary"] / day_total) * 100, 2))

    sedentary_alert_days = [
        day.strftime("%b %d")
        for day in sedentary_dates
        if daily_totals[day]["total"] and (daily_totals[day]["sedentary"] / daily_totals[day]["total"]) >= 0.6
    ]

    reliability_rows = []
    source_display = {
        "wearable": "Wearable",
        "manual": "Manual",
        "mixed": "Wearable + Manual",
        "unknown": "Unknown",
    }
    for source_key in ["wearable", "manual", "mixed", "unknown"]:
        values = reliability_by_source[source_key]
        patient_count = values["patients"]
        reliability_rows.append(
            {
                "source": source_display[source_key],
                "patients": patient_count,
                "avg_risk_score": _safe_round(values["risk_sum"] / values["risk_count"], 2) if values["risk_count"] else None,
                "high_risk_rate": _safe_round((values["high_risk"] / patient_count) * 100, 2) if patient_count else None,
            }
        )

    pop_risk_rows, pop_factor_labels, pop_factor_pcts = _compute_population_risk_stats()

    chart_payload = {
        "hospital_labels": [row["hospital"] for row in hospital_rows],
        "hospital_patient_counts": [row["patients"] for row in hospital_rows],
        "hospital_doctor_counts": [row["doctors"] for row in hospital_rows],
        "hospital_high_risk_counts": [row["high_risk_cases"] for row in hospital_rows],
        "risk_labels": risk_order,
        "risk_values": [system_risk_distribution[label] for label in risk_order],
        "stress_sleep_hospitals": [row["hospital"] for row in hospital_rows],
        "stress_values": [row["avg_stress"] or 0 for row in hospital_rows],
        "sleep_quality_values": [row["avg_sleep_quality"] or 0 for row in hospital_rows],
        "doctor_workload_labels": [row["doctor_name"] for row in doctor_workload_rows[:10]],
        "doctor_patient_counts": [row["patients_assigned"] for row in doctor_workload_rows[:10]],
        "doctor_high_risk_counts": [row["high_risk_patients"] for row in doctor_workload_rows[:10]],
        "activity_labels": list(activity_distribution.keys()),
        "activity_values": list(activity_distribution.values()),
        "sedentary_labels": [day.strftime("%b %d") for day in sedentary_dates],
        "sedentary_values": sedentary_rate_series,
        "xai_features": [row["feature"] for row in xai_feature_importance],
        "xai_scores": [row["score"] for row in xai_feature_importance],
        "source_labels": ["Wearable", "Manual", "Mixed", "Unknown"],
        "source_counts": [
            source_patient_counts["wearable"],
            source_patient_counts["manual"],
            source_patient_counts["mixed"],
            source_patient_counts["unknown"],
        ],
        "pop_factor_labels": pop_factor_labels,
        "pop_factor_values": pop_factor_pcts,
    }

    context = {
        "generated_at": timezone.now(),
        "hospital_rows": hospital_rows,
        "system_risk_distribution": system_risk_distribution,
        "cross_hospital_insight": cross_hospital_insight,
        "avg_stress": _safe_round(system_sleep_stress.get("avg_stress"), 2),
        "avg_sleep_hours": _safe_round(system_sleep_stress.get("avg_sleep_hours"), 2),
        "avg_sleep_quality": _safe_round(system_sleep_stress.get("avg_sleep_quality"), 2),
        "avg_steps": _safe_round(system_sleep_stress.get("avg_steps"), 0),
        "avg_calories": _safe_round(system_sleep_stress.get("avg_calories"), 0),
        "activity_distribution": dict(activity_distribution),
        "doctor_workload_rows": doctor_workload_rows,
        "hospital_workload_rows": hospital_workload_rows,
        "total_predictions": total_predictions,
        "high_risk_alerts": high_risk_alerts,
        "lifestyle_warnings": lifestyle_warnings,
        "xai_feature_importance": xai_feature_importance,
        "wearable_patients": source_patient_counts["wearable"],
        "manual_patients": source_patient_counts["manual"],
        "mixed_source_patients": source_patient_counts["mixed"],
        "unknown_source_patients": source_patient_counts["unknown"],
        "reliability_rows": reliability_rows,
        "total_registered_patients": total_registered_patients,
        "active_patients_today": len(active_patient_ids),
        "total_chatbot_interactions": ChatbotInteraction.objects.count(),
        "active_doctors_today": len(active_doctor_ids),
        "data_quality": data_quality,
        "sedentary_alert_days": sedentary_alert_days,
        "population_risk_factors": pop_risk_rows,
        "chart_data": chart_payload,
    }
    return render_jinja(request, "AdminApp/ManagementAnalytics.html", context)


#Admin Upload Page View : This view renders the dataset upload page for administrators. It serves the HTML template that contains the form for uploading a new dataset, allowing admins to select a CSV file from their local machine and submit it for processing.
def upload_page(request):
    return render_jinja(request, "AdminApp/Upload.html")

#Admin Upload Action View : This view handles the POST request from the dataset upload form. It checks if a file was uploaded, saves it to the designated upload folder, reads the dataset into a pandas DataFrame, and then renders a page to display the dataset's columns and a preview of its rows. If no file is uploaded or if the request method is not POST, it redirects back to the upload page with an appropriate error message.
def upload_action(request):
    global dataset, filepath

    if request.method != "POST":
        return redirect("Upload")

    uploaded_file = request.FILES.get("dataset")
    if not uploaded_file:
        return HttpResponse("No file part", status=400)

    filepath = UPLOAD_FOLDER / uploaded_file.name
    _save_uploaded_file(uploaded_file, filepath)

    dataset = pd.read_csv(filepath)
    columns = dataset.columns.tolist()
    rows = dataset.head().values.tolist()
    return render_jinja(
        request,
        "AdminApp/ViewDataset.html",
        {"columns": columns, "rows": rows, "filepath": str(filepath)},
    )

#Data Preprocessing View : This view triggers the data preprocessing function when accessed. It calls the preprocess_data function, which handles all the steps of preparing the dataset for model training, and then renders a page to display the results of the preprocessing, including the total number of samples and the counts for training and testing sets.
def preprocess_view(request):
    total, train_count, test_count = preprocess_data()
    return render_jinja(
        request,
        "AdminApp/SplitStatus.html",
        {"total": total, "train": train_count, "test": test_count},
    )


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _dataset_mtime():
    dataset_path = DATASET_FOLDER / "lifestyle_disorder_wearable_dataset.csv"
    if not dataset_path.exists():
        return None
    return int(dataset_path.stat().st_mtime)


def _normalize_confusion_matrix(matrix):
    if not isinstance(matrix, list):
        return None

    normalized = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for row_index in range(min(3, len(matrix))):
        row_value = matrix[row_index]
        if not isinstance(row_value, list):
            continue
        for col_index in range(min(3, len(row_value))):
            normalized[row_index][col_index] = int(_safe_float(row_value[col_index], 0))
    return normalized


def _normalize_training_metrics_payload(raw_payload):
    if not isinstance(raw_payload, dict):
        return None

    raw_metrics = raw_payload.get("metrics")
    if not isinstance(raw_metrics, dict):
        raw_metrics = raw_payload.get("detailed_metrics", {})
    if not isinstance(raw_metrics, dict):
        raw_metrics = {}

    normalized_metrics = {}
    for algo_name, algo_metrics in raw_metrics.items():
        if isinstance(algo_metrics, dict):
            normalized_metrics[str(algo_name)] = {
                "accuracy": round(_safe_float(algo_metrics.get("accuracy")), 2),
                "precision": round(_safe_float(algo_metrics.get("precision")), 2),
                "recall": round(_safe_float(algo_metrics.get("recall")), 2),
                "f1_score": round(_safe_float(algo_metrics.get("f1_score")), 2),
                "confusion_matrix": _normalize_confusion_matrix(algo_metrics.get("confusion_matrix")),
            }
            continue

        # Backward compatibility with payloads where metric value was plain accuracy.
        normalized_metrics[str(algo_name)] = {
            "accuracy": round(_safe_float(algo_metrics), 2),
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "confusion_matrix": None,
        }

    return {
        "timestamp": raw_payload.get("timestamp"),
        "train_size": raw_payload.get("train_size"),
        "test_size": raw_payload.get("test_size"),
        "metrics": normalized_metrics,
        "dataset_mtime": raw_payload.get("dataset_mtime"),
    }


def _load_saved_training_metrics():
    metrics_file = MODEL_FOLDER / "training_metrics.json"
    if not metrics_file.exists():
        return None

    try:
        with metrics_file.open("r", encoding="utf-8") as file_obj:
            raw_payload = json.load(file_obj)
    except Exception:
        return None

    return _normalize_training_metrics_payload(raw_payload)

#Model Training View : This view handles the training of machine learning models when accessed. It checks if the training data is ready, and if not, it attempts to preprocess the data automatically. If the data is still not ready after preprocessing, it prompts the user to run preprocessing first. If the data is ready, it trains three different models (XGBoost, Random Forest, and Decision Tree), evaluates their performance, generates visualizations for model performance and feature importance, saves the trained models and metrics, and then renders a page to display the results of the training process, including accuracy scores and links to the generated visualizations.
def train_models(request):
    global X_train, X_test, y_train, y_test
    global xgb_acc, rf_acc, dec_acc
    #Ensure Preprocessed Data Exists
    #This checks whether training data is ready.
    if X_train is None or y_train is None or len(X_train) == 0:
        dataset_path = DATASET_FOLDER / "lifestyle_disorder_wearable_dataset.csv"
        if dataset_path.exists():
            try:
                preprocess_data()
            except Exception as exc:
                return render_jinja(
                    request,
                    "AdminApp/AlgorithmStatus.html",
                    {
                        "msg": f"Automatic preprocessing failed: {str(exc)}",
                        "results": {},
                        "detailed_metrics": {},
                        "shap_image": None,
                        "lime_file": None,
                        "confusion_matrix_file": None,
                        "train_size": 0,
                        "test_size": 0,
                    },
                )

            if X_train is None or y_train is None or len(X_train) == 0:
                return render_jinja(
                    request,
                    "AdminApp/AlgorithmStatus.html",
                    {
                        "msg": "Preprocessing ran but training data still not ready. Check dataset.",
                        "results": {},
                        "detailed_metrics": {},
                        "shap_image": None,
                        "lime_file": None,
                        "confusion_matrix_file": None,
                        "train_size": 0,
                        "test_size": 0,
                    },
                )
        else:
            return render_jinja(
                request,
                "AdminApp/AlgorithmStatus.html",
                {
                    "msg": "Training data not ready. Please run preprocessing first (visit /preprocess).",
                    "results": {},
                    "detailed_metrics": {},
                    "shap_image": None,
                    "lime_file": None,
                    "confusion_matrix_file": None,
                    "train_size": 0,
                    "test_size": 0,
                },
            )

    assert X_train is not None and X_test is not None
    assert y_train is not None and y_test is not None

    results = {}
    detailed_metrics = {}

    force_retrain = str(request.GET.get("force", "")).strip().lower() in {"1", "true", "yes"}
    model_files = ["XGModel.joblib", "RFModel.joblib", "DTModel.joblib", "encoders.joblib", "scaler.joblib", "target_encoder.joblib"]
    cached_metrics = _load_saved_training_metrics()
    current_dataset_mtime = _dataset_mtime()
    has_all_model_files = all((MODEL_FOLDER / model_file).exists() for model_file in model_files)

    if (
        not force_retrain
        and cached_metrics is not None
        and has_all_model_files
        and cached_metrics.get("metrics")
        and cached_metrics.get("dataset_mtime") == current_dataset_mtime
    ):
        cached_results = {
            algo_name: metric_values.get("accuracy")
            for algo_name, metric_values in cached_metrics["metrics"].items()
        }
        return render_jinja(
            request,
            "AdminApp/AlgorithmStatus.html",
            {
                "msg": "Using cached training results (dataset unchanged). Add ?force=1 to retrain.",
                "results": cached_results,
                "detailed_metrics": cached_metrics["metrics"],
                "shap_image": "shap_summary.png" if (STATIC_FOLDER / "shap_summary.png").exists() else None,
                "lime_file": "lime_explanation.html" if (STATIC_FOLDER / "lime_explanation.html").exists() else None,
                "confusion_matrix_file": "confusion_matrix_rf.png" if (STATIC_FOLDER / "confusion_matrix_rf.png").exists() else None,
                "train_size": cached_metrics.get("train_size") or len(X_train),
                "test_size": cached_metrics.get("test_size") or len(X_test),
            },
        )
    #Train XGBoost Model : This section initializes an XGBoost classifier, fits it to the training data, saves the trained model, makes predictions on the test set, calculates accuracy and other performance metrics, and stores these metrics in a structured format for later display.
    xgb_model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_estimators=80,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, MODEL_FOLDER / "XGModel.joblib")
    #Evaluate XGBoost Model : This section makes predictions using the trained XGBoost model, calculates accuracy, precision, recall, F1 score, and confusion matrix, and stores these metrics in a structured format for later display.
    xgb_pred = xgb_model.predict(X_test)
    xgb_acc = round(accuracy_score(y_test, xgb_pred) * 100, 2)
    results["XGBoost"] = xgb_acc

    xgb_precision = round(float(precision_score(y_test, xgb_pred, average="weighted")) * 100, 2)
    xgb_recall = round(float(recall_score(y_test, xgb_pred, average="weighted")) * 100, 2)
    xgb_f1 = round(float(f1_score(y_test, xgb_pred, average="weighted")) * 100, 2)
    xgb_cm = confusion_matrix(y_test, xgb_pred).tolist()

    detailed_metrics["XGBoost"] = {
        "accuracy": xgb_acc,
        "precision": xgb_precision,
        "recall": xgb_recall,
        "f1_score": xgb_f1,
        "confusion_matrix": xgb_cm,
    }
    #Train Random Forest Model : This section initializes a Random Forest classifier, fits it to the training data, saves the trained model, makes predictions on the test set, calculates accuracy and other performance metrics, and stores these metrics in a structured format for later display. It also generates a confusion matrix heatmap for the Random Forest model and saves it as an image.
    rf_model = RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, MODEL_FOLDER / "RFModel.joblib")

    rf_pred = rf_model.predict(X_test)
    rf_acc = round(accuracy_score(y_test, rf_pred) * 100, 2)
    results["Random Forest"] = rf_acc

    rf_precision = round(float(precision_score(y_test, rf_pred, average="weighted")) * 100, 2)
    rf_recall = round(float(recall_score(y_test, rf_pred, average="weighted")) * 100, 2)
    rf_f1 = round(float(f1_score(y_test, rf_pred, average="weighted")) * 100, 2)
    rf_cm = confusion_matrix(y_test, rf_pred).tolist()

    detailed_metrics["Random Forest"] = {
        "accuracy": rf_acc,
        "precision": rf_precision,
        "recall": rf_recall,
        "f1_score": rf_f1,
        "confusion_matrix": rf_cm,
    }
    #heatmap for Random Forest Confusion Matrix : This generates a heatmap visualization of the confusion matrix for the Random Forest model using Seaborn and Matplotlib, and saves the resulting image to the static folder for later display in the admin interface.
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        rf_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["HIGH RISK", "LOW RISK", "MODERATE RISK"],
        yticklabels=["HIGH RISK", "LOW RISK", "MODERATE RISK"],
    )
    plt.title("Random Forest Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    plt.tight_layout()
    plt.savefig(STATIC_FOLDER / "confusion_matrix_rf.png", dpi=150, bbox_inches="tight")
    plt.close()
    #decision tree model training and evaluation : This section initializes a Decision Tree classifier, fits it to the training data, saves the trained model, makes predictions on the test set, calculates accuracy and other performance metrics, and stores these metrics in a structured format for later display.
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    joblib.dump(dt_model, MODEL_FOLDER / "DTModel.joblib")

    dt_pred = dt_model.predict(X_test)
    dec_acc = round(accuracy_score(y_test, dt_pred) * 100, 2)
    results["Decision Tree"] = dec_acc

    dt_precision = round(float(precision_score(y_test, dt_pred, average="weighted")) * 100, 2)
    dt_recall = round(float(recall_score(y_test, dt_pred, average="weighted")) * 100, 2)
    dt_f1 = round(float(f1_score(y_test, dt_pred, average="weighted")) * 100, 2)
    dt_cm = confusion_matrix(y_test, dt_pred).tolist()

    detailed_metrics["Decision Tree"] = {
        "accuracy": dec_acc,
        "precision": dt_precision,
        "recall": dt_recall,
        "f1_score": dt_f1,
        "confusion_matrix": dt_cm,
    }
    #Explainable AI 
    #SHAP Summary Plot : This section uses SHAP (SHapley Additive exPlanations) to generate a summary plot that shows the impact of each feature on the predictions made by the Random Forest model. The plot is saved as an image in the static folder for later display in the admin interface.
    shap_image = None
    lime_file = None
    try:
        shap_rows = min(40, len(X_test))
        shap_explainer = shap.TreeExplainer(rf_model)
        shap_values = shap_explainer.shap_values(X_test.iloc[:shap_rows])
        
        # For multi-class, shap_values is a list; average across all classes
        if isinstance(shap_values, list):
            shap_values = np.mean(np.array(shap_values), axis=0)
        
        shap.summary_plot(shap_values, X_test.iloc[:shap_rows], show=False)
        plt.savefig(STATIC_FOLDER / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        shap_image = "shap_summary.png"
    except Exception as e:
        plt.close()
        print(f"SHAP generation failed: {str(e)}")
    #LIME Explanation : This section uses LIME (Local Interpretable Model-agnostic Explanations) to generate an explanation for a single test instance. It creates a LIME explainer using the training data, generates an explanation for the first test instance, and saves the explanation as an HTML file in the static folder for later display in the admin interface.
    try:
        lime_train_sample = X_train.sample(n=min(2000, len(X_train)), random_state=42)
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=lime_train_sample.values,
            feature_names=X_train.columns.tolist(),
            class_names=["HIGH RISK", "LOW RISK", "MODERATE RISK"],
            mode="classification",
        )
        #One Specific Explanation : This generates a LIME explanation for the first instance in the test set using the Random Forest model's predict_proba function. The explanation is saved as an HTML file in the static folder, allowing administrators to view the local feature importance for that specific prediction.
        lime_exp = lime_explainer.explain_instance(
            X_test.iloc[0].values,
            rf_model.predict_proba,
            num_features=10,
        )
        lime_exp.save_to_file(STATIC_FOLDER / "lime_explanation.html")
        lime_file = "lime_explanation.html"
    except Exception as e:
        print(f"LIME generation failed: {str(e)}")
        lime_file = None
    #Save Training Metrics : This section saves the training metrics for all three models (XGBoost, Random Forest, and Decision Tree) to a JSON file in the Models folder. The saved metrics include accuracy, precision, recall, F1 score, and confusion matrix for each model, along with a timestamp and the sizes of the training and testing datasets. This allows for easy retrieval and display of model performance metrics in the admin interface.
    metrics_file = MODEL_FOLDER / "training_metrics.json"
    training_info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "metrics": detailed_metrics,
        "dataset_mtime": current_dataset_mtime,
    }
    with metrics_file.open("w", encoding="utf-8") as file_obj:
        json.dump(training_info, file_obj, indent=2)

    return render_jinja(
        request,
        "AdminApp/AlgorithmStatus.html",
        {
            "msg": "All Algorithms Executed Successfully",
            "results": results,
            "detailed_metrics": detailed_metrics,
            "shap_image": shap_image,
            "lime_file": lime_file,
            "confusion_matrix_file": "confusion_matrix_rf.png",
            "train_size": len(X_train),
            "test_size": len(X_test),
        },
    )


def comparison(request):
    global xgb_acc, rf_acc, dec_acc

    if xgb_acc is None or rf_acc is None or dec_acc is None:
        metrics_file = MODEL_FOLDER / "training_metrics.json"
        if metrics_file.exists():
            try:
                with metrics_file.open("r", encoding="utf-8") as file_obj:
                    metrics_data = json.load(file_obj)
                if "metrics" in metrics_data:
                    xgb_acc = metrics_data["metrics"].get("XGBoost", {}).get("accuracy")
                    rf_acc = metrics_data["metrics"].get("Random Forest", {}).get("accuracy")
                    dec_acc = metrics_data["metrics"].get("Decision Tree", {}).get("accuracy")
            except Exception:
                pass

    if xgb_acc is None or rf_acc is None or dec_acc is None:
        return render_jinja(
            request,
            "AdminApp/Grpah.html",
            {"error": "Models not trained yet. Please train models first.", "has_data": False},
        )

    models = ["XGBoost", "Random Forest", "Decision Tree"]
    accuracies = [xgb_acc, rf_acc, dec_acc]

    valid_data = [(m, a) for m, a in zip(models, accuracies) if a is not None]
    if not valid_data:
        return render_jinja(
            request,
            "AdminApp/Grpah.html",
            {"error": "No valid accuracy data available. Please train models first.", "has_data": False},
        )

    models_clean = [m for m, _ in valid_data]
    accuracies_clean = [float(a) for _, a in valid_data if isinstance(a, (int, float))]

    if not accuracies_clean:
        return render_jinja(
            request,
            "AdminApp/Grpah.html",
            {"error": "No valid accuracy data available. Please train models first.", "has_data": False},
        )

    if len(models_clean) != len(accuracies_clean):
        models_clean = models_clean[: len(accuracies_clean)]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models_clean, accuracies_clean, color=["#1E3A8A", "#B59410", "#198754"][: len(models_clean)])

    for bar, acc in zip(bars, accuracies_clean):
        if isinstance(acc, (int, float)) and not np.isnan(acc):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                float(acc) + 1,
                f"{acc:.2f}%",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

    plt.title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, max(accuracies_clean) + 10 if accuracies_clean else 100)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(STATIC_FOLDER / "model_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()

    return render_jinja(request, "AdminApp/Grpah.html", {"has_data": True})

#User Login Page View : This view renders the user login page when accessed. It serves the HTML template for the user login interface, allowing users to enter their credentials to access their accounts and use the application's features.
def user_login_page(request):
    return render_jinja(request, "UserApp/Login.html")

#User Registration Page View : This view renders the user registration page when accessed. It serves the HTML template for the user registration interface, allowing new users to create an account by providing their name, email, mobile number, username, and password.
def register_page(request):
    return render_jinja(request, "UserApp/Register.html", {"doctors": Doctor.objects.order_by("name")})

#User Registration Action View : This view handles the POST request from the user registration form. It retrieves the submitted user information, checks if the username or email already exists in the database, and if not, it inserts the new user into the database. It then renders the registration page with a success message if registration is successful, or an error message if the username or email already exists. If the request method is not POST, it redirects to the registration page.
def register_action(request):
    if request.method != "POST":
        return redirect("register")

    doctors = Doctor.objects.order_by("name")
    form_data = {
        "name": request.POST.get("name", "").strip(),
        "email": request.POST.get("email", "").strip(),
        "mobile": request.POST.get("mobile", "").strip(),
        "username": request.POST.get("username", "").strip(),
        "password": request.POST.get("password", ""),
        "age": request.POST.get("age", "").strip(),
        "gender": request.POST.get("gender", "").strip(),
        "doctor_id": request.POST.get("doctor_id", "").strip(),
    }

    if not form_data["doctor_id"]:
        return render_jinja(
            request,
            "UserApp/Register.html",
            {"msg": "Please select a doctor.", "doctors": doctors, "form": form_data},
        )

    try:
        doctor = Doctor.objects.get(id=form_data["doctor_id"])
        age_value = int(form_data["age"])
    except (Doctor.DoesNotExist, ValueError):
        return render_jinja(
            request,
            "UserApp/Register.html",
            {"msg": "Invalid registration data. Please review your details.", "doctors": doctors, "form": form_data},
        )

    if Patient.objects.filter(username=form_data["username"]).exists() or Patient.objects.filter(email=form_data["email"]).exists():
        return render_jinja(
            request,
            "UserApp/Register.html",
            {"msg": "Username or email already exists!", "doctors": doctors, "form": form_data},
        )

    Patient.objects.create(
        name=form_data["name"],
        email=form_data["email"],
        mobile=form_data["mobile"],
        username=form_data["username"],
        password=form_data["password"],
        age=age_value,
        gender=form_data["gender"],
        doctor=doctor,
        assigned_doctor=doctor,
    )
    return render_jinja(
        request,
        "UserApp/Register.html",
        {"msg": "Successfully Registered!", "doctors": doctors},
    )

#User Login Action View : This view handles the POST request from the user login form. It retrieves the submitted username and password, checks them against the database, and if a matching user is found, it sets the username in the session and renders the user home page with a welcome message. If the login fails (no matching user), it re-renders the login page with an error message. If the request method is not POST, it redirects to the user login page.
def user_action(request):
    if request.method != "POST":
        return redirect("userlogin")

    username = request.POST.get("username", "")
    password = request.POST.get("password", "")

    patient = Patient.objects.filter(username=username, password=password).first()
    if patient is None:
        return render_jinja(request, "UserApp/Login.html", {"msg": "Login Failed!"})

    # Ensure stale doctor session data cannot affect patient access checks.
    request.session.pop("doctor_id", None)
    request.session.pop("doctor_name", None)
    request.session["username"] = patient.username
    request.session["patient_id"] = patient.pk
    return redirect("UserHome")

#Manage Users View : This view retrieves all user records from the shared NeonDB PostgreSQL database and renders a page to display the list of users.
def manage_users(request):
    users = list(Patient.objects.values_list("name", "email", "mobile", "username"))
    return render_jinja(request, "AdminApp/ManageUsers.html", {"users": users})

#Delete User Action View : This view handles the deletion of a user when accessed. It deletes by username from the shared NeonDB PostgreSQL database and redirects to the manage users page.
def delete_user(request, username):
    Patient.objects.filter(username=username).delete()
    return redirect("ManageUsers")

#Detect Page View : This view renders the detection page for users. It serves the HTML template that contains the form for users to input their health and lifestyle data, which will be used to predict their lifestyle disorder risk using the trained machine learning models.
def detect_page(request):
    patient = _get_session_patient(request)
    if patient is None:
        return redirect("userlogin")
    latest_metric = HealthMetrics.objects.filter(patient=patient).order_by("-date").first()
    latest_prediction = Prediction.objects.filter(patient=patient).order_by("-created_at").first()
    metric_prefill = {
        "heart_rate": latest_metric.heart_rate if latest_metric is not None else None,
        "stress_level": latest_metric.stress_level if latest_metric is not None else None,
        "sleep_hours": latest_metric.sleep_hours if latest_metric is not None else None,
        "sleep_quality": latest_metric.sleep_quality if latest_metric is not None else None,
        "steps": latest_metric.steps if latest_metric is not None else None,
        "calories_burned": latest_metric.calories_burned if latest_metric is not None else None,
        "blood_oxygen": latest_metric.blood_oxygen if latest_metric is not None else None,
        "activity_level": latest_metric.activity_level if latest_metric is not None else "",
    }
    return render_jinja(
        request,
        "UserApp/Detect.html",
        {
            "patient": patient,
            "latest_metric": latest_metric,
            "latest_prediction": latest_prediction,
            "has_saved_metrics": latest_metric is not None,
            "latest_metric_date": latest_metric.date if latest_metric is not None else None,
            "metric_prefill": metric_prefill,
        },
    )

#User Home Page View : This view renders the user home page when accessed. It retrieves the username from the session (defaulting to "Guest" if not found) and passes it to the template for rendering. The resulting page welcomes the user by name and provides access to the application's features, such as the detection page for predicting lifestyle disorder risk.
def user_home(request):
    patient = _get_session_patient(request)
    username = patient.name if patient is not None else request.session.get("username", "Guest")
    latest_prediction = None
    trend_context: Dict[str, Any] = {
        "chart_labels": [],
        "steps_values": [],
        "sleep_values": [],
        "heart_rate_values": [],
        "stress_values": [],
    }
    lifestyle_score = None
    risk_projection = None
    ai_insights: list = []
    risk_history_labels: list = []
    risk_history_values: list = []

    if patient is not None:
        latest_prediction = Prediction.objects.filter(patient=patient).order_by("-created_at").first()
        # Fetch up to 14 days so projections and insight cards have enough history
        metrics = list(HealthMetrics.objects.filter(patient=patient).order_by("-date")[:14])
        metrics.reverse()
        trend_context = _build_trend_payload(metrics[-7:]) if metrics else trend_context

        lifestyle_score = _compute_lifestyle_score(metrics)
        if latest_prediction:
            risk_projection = _generate_risk_projection(metrics, latest_prediction.risk_score)
        ai_insights = _generate_ai_insight_cards(metrics)

        # Last 10 predictions for risk history chart
        prediction_history = list(Prediction.objects.filter(patient=patient).order_by("created_at")[:10])
        risk_history_labels = [p.created_at.strftime("%b %d") for p in prediction_history]
        risk_history_values = [p.risk_score for p in prediction_history]

    return render_jinja(
        request,
        "UserApp/Home.html",
        {
            "username": username,
            "patient": patient,
            "latest_prediction": latest_prediction,
            "risk_factors": _split_text_list(latest_prediction.risk_factors) if latest_prediction else [],
            "lifestyle_recommendations": _split_text_list(latest_prediction.lifestyle_recommendations) if latest_prediction else [],
            "lifestyle_score": lifestyle_score,
            "risk_projection": risk_projection,
            "ai_insights": ai_insights,
            "risk_history_labels": risk_history_labels,
            "risk_history_values": risk_history_values,
            **trend_context,
        },
    )


def _run_prediction(request, raw_input):
    lime_list = []
    lime_filename = None
    shap_bar_filename = None
    feature_importance = None
    patient = _get_session_patient(request)
    recent_metrics = []
    if patient is not None:
        recent_metrics = list(HealthMetrics.objects.filter(patient=patient).order_by("-date")[:7])

    try:
        test = pd.DataFrame(
            [
                {
                    "age": float(raw_input["age"]),
                    "gender": str(raw_input["gender"]),
                    "resting_heart_rate": float(raw_input["resting_heart_rate"]),
                    "stress_level": float(raw_input["stress_level"]),
                    "sleep_duration_hours": float(raw_input["sleep_duration_hours"]),
                    "sleep_quality_score": float(raw_input["sleep_quality_score"]),
                    "steps_per_day": float(raw_input["steps_per_day"]),
                    "calories_burned": float(raw_input["calories_burned"]),
                    "blood_oxygen_level": float(raw_input["blood_oxygen_level"]),
                    "activity_level": str(raw_input["activity_level"]),
                }
            ]
        )

        encoders = joblib.load(MODEL_FOLDER / "encoders.joblib")
        for col in encoders:
            test[col] = [_safe_transform_label(encoders[col], test[col].iloc[0])]

        scaler = joblib.load(MODEL_FOLDER / "scaler.joblib")
        test[NUMERIC_COLUMNS] = scaler.transform(test[NUMERIC_COLUMNS])

        model = joblib.load(MODEL_FOLDER / "RFModel.joblib")
        target_encoder = joblib.load(MODEL_FOLDER / "target_encoder.joblib")

        cache_file = MODEL_FOLDER / "X_train_full_cache.joblib"
        if cache_file.exists():
            X_train_full = joblib.load(cache_file)
        else:
            dataset_file = DATASET_FOLDER / "lifestyle_disorder_wearable_dataset.csv"
            dataset_local = pd.read_csv(dataset_file)
            dataset_local.dropna(inplace=True)

            for col in CATEGORICAL_COLUMNS:
                dataset_local[col] = encoders[col].transform(dataset_local[col])

            dataset_local[NUMERIC_COLUMNS] = scaler.transform(dataset_local[NUMERIC_COLUMNS])
            X_train_full = dataset_local[FEATURE_COLUMNS]
            joblib.dump(X_train_full, cache_file)

        pred_raw = int(model.predict(test)[0])
        proba_all = model.predict_proba(test)[0]
        probability_map = _label_probability_map(model, proba_all, target_encoder)
        predicted_label = str(target_encoder.inverse_transform(np.array([pred_raw]))[0]).upper()
        predicted_probability = float(np.max(proba_all))
        model_risk_score = _calculate_risk_score(probability_map)

        try:
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train_full.values,
                feature_names=X_train_full.columns.tolist(),
                class_names=[str(label).upper() for label in target_encoder.inverse_transform(model.classes_.astype(int))],
                mode="classification",
            )

            lime_exp = lime_explainer.explain_instance(
                test.iloc[0].values,
                model.predict_proba,
                num_features=min(len(FEATURE_COLUMNS), 8),
            )

            lime_list = lime_exp.as_list()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            lime_explanation_file = STATIC_FOLDER / f"lime_user_{timestamp}.html"
            lime_exp.save_to_file(lime_explanation_file)
            lime_filename = lime_explanation_file.name

        except Exception:
            lime_list = []
            lime_filename = None

        try:
            shap_explainer = shap.TreeExplainer(model)
            shap_values = shap_explainer.shap_values(test.iloc[0])
            class_index = int(np.where(model.classes_ == pred_raw)[0][0])

            if isinstance(shap_values, list):
                shap_values_single = shap_values[class_index]
            else:
                if len(np.shape(shap_values)) == 2:
                    shap_values_single = shap_values[class_index]
                else:
                    shap_values_single = shap_values

            if len(np.shape(shap_values_single)) > 1:
                shap_values_single = shap_values_single.flatten()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            shap_bar_file = STATIC_FOLDER / f"shap_bar_user_{timestamp}.png"

            feature_importance = pd.DataFrame(
                {
                    "Feature": FEATURE_COLUMNS,
                    "SHAP_Value": shap_values_single[: len(FEATURE_COLUMNS)],
                }
            ).sort_values("SHAP_Value", key=abs, ascending=False)

            plt.figure(figsize=(10, 6))
            colors = ["#dc3545" if x < 0 else "#198754" for x in feature_importance["SHAP_Value"]]
            plt.barh(feature_importance["Feature"], feature_importance["SHAP_Value"], color=colors)
            plt.xlabel("SHAP Value (Impact on Prediction)", fontsize=12)
            plt.title("Feature Impact on Your Prediction", fontsize=14, fontweight="bold")
            plt.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
            plt.tight_layout()
            plt.savefig(shap_bar_file, dpi=150, bbox_inches="tight")
            plt.close()

            shap_bar_filename = shap_bar_file.name

        except Exception:
            shap_bar_filename = None
            feature_importance = None

        lifestyle_analysis = _analyze_lifestyle_disorder(raw_input, recent_metrics=recent_metrics, model_risk_score=model_risk_score)
        risk_score = lifestyle_analysis["risk_score"]
        priority = lifestyle_analysis["risk_level"]
        risk_factors = lifestyle_analysis["factors"]
        data_source = _derive_data_source(request.session.get("watch_data", {}), raw_input)

        explanation_text = _build_textual_explanation(raw_input, feature_importance, lime_list, risk_score, priority)
        if risk_factors:
            explanation_text = f"{explanation_text} Key risk factors include {', '.join(risk_factors)}."

        badge = _prediction_badge(priority)
        probability = f"{predicted_probability * 100:.2f}%"

        fit_daily = request.session.get("fit_daily_data", {})
        triage_result = suggest_disease(
            {
                "age": raw_input.get("age"),
                "heart_rate": raw_input.get("resting_heart_rate"),
                "steps": raw_input.get("steps_per_day"),
                "sleep": raw_input.get("sleep_duration_hours"),
                "sleep_quality": raw_input.get("sleep_quality_score"),
                "stress_level": raw_input.get("stress_level"),
                "blood_oxygen": raw_input.get("blood_oxygen_level"),
                "calories": raw_input.get("calories_burned"),
            }
        )
        required_specializations = _required_specialization_from_metrics(
            {
                "age": raw_input.get("age"),
                "heart_rate": raw_input.get("resting_heart_rate"),
                "steps": raw_input.get("steps_per_day"),
                "sleep_hours": raw_input.get("sleep_duration_hours"),
                "sleep_quality": raw_input.get("sleep_quality_score"),
                "stress_level": raw_input.get("stress_level"),
                "blood_oxygen": raw_input.get("blood_oxygen_level"),
                "calories": raw_input.get("calories_burned"),
            }
        )
        recovery_result = _nervous_system_recovery_insight(raw_input, recent_metrics=recent_metrics)
        stress_sleep_result = _stress_sleep_interaction_insight(raw_input, recent_metrics=recent_metrics)
        readable_ai_explanation = _build_ai_explanation(risk_factors, feature_importance, lime_list, risk_score)
        recommendation_list = _build_lifestyle_recommendations(risk_factors, recovery_result["score"], priority)
        follow_up_tests = _recommended_follow_up_tests(triage_result, risk_factors, recovery_result["score"])

        if patient is not None:
            with transaction.atomic():
                _persist_patient_metrics(patient, raw_input, fit_daily, data_source)

                auto_doctor = _find_balanced_doctor(required_specializations)
                if auto_doctor is None:
                    auto_doctor = patient.assigned_doctor or patient.doctor

                patient.suggested_disease = triage_result["disease_name"]
                patient.disease_risk_level = triage_result["risk_level"]
                patient.suggested_tests = ", ".join(follow_up_tests)
                patient.assigned_doctor = auto_doctor
                if auto_doctor is not None:
                    patient.doctor = auto_doctor
                patient.save(
                    update_fields=[
                        "suggested_disease",
                        "disease_risk_level",
                        "suggested_tests",
                        "assigned_doctor",
                        "doctor",
                    ]
                )

                Prediction.objects.create(
                    patient=patient,
                    risk_score=risk_score,
                    priority=priority,
                    explanation_text=explanation_text,
                    predicted_label=predicted_label,
                    data_source=data_source,
                    risk_factors="|".join(risk_factors),
                    readable_ai_explanation=readable_ai_explanation,
                    nervous_system_recovery_score=recovery_result["score"],
                    nervous_system_explanation=recovery_result["explanation"],
                    stress_sleep_insight=stress_sleep_result["doctor_text"],
                    recommended_follow_up=", ".join(follow_up_tests),
                    lifestyle_recommendations="|".join(recommendation_list),
                )

        request.session.pop("watch_data", None)

        return render_jinja(
            request,
            "UserApp/Result.html",
            {
                "patient": patient,
                "prediction": predicted_label,
                "probability": probability,
                "badge": badge,
                "recommendations": _prediction_recommendation(priority),
                "input_data": raw_input,
                "show_result": True,
                "risk_score": risk_score,
                "priority": priority,
                "explanation_text": explanation_text,
                "doctor": patient.assigned_doctor if patient is not None and patient.assigned_doctor is not None else (patient.doctor if patient is not None else None),
                "disease_name": triage_result["disease_name"],
                "disease_risk_level": triage_result["risk_level"],
                "suggested_tests": follow_up_tests,
                "assigned_doctor": patient.assigned_doctor if patient is not None else None,
                "data_source": data_source,
                "risk_factors": risk_factors,
                "readable_ai_explanation": readable_ai_explanation,
                "nervous_system_recovery_score": recovery_result["score"],
                "nervous_system_explanation": recovery_result["explanation"],
                "stress_sleep_patient_insight": stress_sleep_result["patient_text"],
                "stress_sleep_doctor_insight": stress_sleep_result["doctor_text"],
                "lifestyle_recommendations_list": recommendation_list,
                "steps": float(raw_input.get("steps_per_day", 0) or 0),
                "calories": float(raw_input.get("calories_burned", 0) or 0),
                "sleep": float(raw_input.get("sleep_duration_hours", 0) or 0),
                "stress_level": float(raw_input.get("stress_level", 0) or 0),
                "day_labels": fit_daily.get("day_labels", []),
                "daily_steps": fit_daily.get("daily_steps", []),
                "daily_calories": fit_daily.get("daily_calories", []),
                "daily_heart_rate": fit_daily.get("daily_heart_rate", []),
                "daily_stress": fit_daily.get("daily_stress", []),
                "lime_explanation": lime_list,
                "lime_file": lime_filename,
                "shap_bar_file": shap_bar_filename,
                "feature_importance": feature_importance.to_dict("records") if feature_importance is not None else [],
                "pred_raw": pred_raw,
            },
        )
    except FileNotFoundError as exc:
        return render_jinja(
            request,
            "UserApp/Result.html",
            {
                "patient": patient,
                "show_result": False,
                "error": f"Model files not found. Please train models first. Error: {str(exc)}",
            },
        )
    except Exception as exc:
        return render_jinja(
            request,
            "UserApp/Result.html",
            {
                "patient": patient,
                "show_result": False,
                "error": f"Error processing prediction: {str(exc)}",
            },
        )
#Detect Action View : This view handles the POST request from the detection form. It retrieves the input data from the form, preprocesses it to match the format expected by the trained machine learning model, makes a prediction using the model, and generates explanations using LIME and SHAP for the prediction. The results, including the predicted risk level, probability, and explanations, are then rendered on a results page for the user to view. If the request method is not POST, it simply renders the results page without showing any results.
def detect_action(request):
    if request.method != "POST":
        return render_jinja(request, "UserApp/Result.html", {"patient": _get_session_patient(request), "show_result": False})

    patient = _get_session_patient(request)
    if patient is None:
        return redirect("userlogin")

    watch_data = request.session.get("watch_data", {})
    raw_input = {}
    for field in REQUIRED_FIELDS:
        watch_value = watch_data.get(field)
        if watch_value not in [None, "", "null"]:
            raw_input[field] = watch_value
        else:
            raw_input[field] = request.POST.get(field)

    if raw_input.get("age") in [None, "", "null"]:
        raw_input["age"] = patient.age
    if raw_input.get("gender") in [None, "", "null"]:
        raw_input["gender"] = patient.gender

    missing_fields = [field for field in REQUIRED_FIELDS if raw_input.get(field) in [None, "", "null"]]
    if missing_fields:
        return render(
            request,
            "UserApp/manual_missing_fields.html",
            {
                "missing_fields": missing_fields,
                "prefilled_data": raw_input,
            },
            using="django",
        )

    return _run_prediction(request, raw_input)

#Retrain View : This view handles the retraining of machine learning models when accessed. It processes a new dataset uploaded by the administrator, combines it with the existing dataset, preprocesses the combined data, and updates the training and testing datasets. The view ensures that the new data is properly integrated into the existing dataset, and it prepares the data for retraining the models. After processing, it redirects to the training page to allow the administrator to train the models with the updated dataset.
def retrain(request):
    global X_train, X_test, y_train, y_test

    if request.method != "POST":
        return redirect("AdminHome")

    try:
        existing_data = pd.read_csv(DATASET_FOLDER / "lifestyle_disorder_wearable_dataset.csv")
        #Receive and Process New Dataset : This section retrieves the new dataset file uploaded by the administrator through the form. It checks if a file was uploaded and has a valid name, saves the uploaded file to the designated upload folder, and reads the new dataset into a pandas DataFrame. The new data is then combined with the existing dataset, any missing values are dropped, and the updated dataset is saved back to disk for use in retraining the models.
        new_file = request.FILES.get("new_data")
        if new_file and new_file.name:
            file_name = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            new_filepath = UPLOAD_FOLDER / file_name
            _save_uploaded_file(new_file, new_filepath)
            new_data = pd.read_csv(new_filepath)

            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            combined_data.dropna(inplace=True)
            combined_data.to_csv(DATASET_FOLDER / "lifestyle_disorder_wearable_dataset.csv", index=False)

            dataset_local = combined_data.copy()

            encoders = joblib.load(MODEL_FOLDER / "encoders.joblib")
            for col in CATEGORICAL_COLUMNS:
                le = LabelEncoder()
                dataset_local[col] = pd.Series(le.fit_transform(dataset_local[col]), index=dataset_local.index)
                encoders[col] = le
            joblib.dump(encoders, MODEL_FOLDER / "encoders.joblib")

            target_encoder = LabelEncoder()
            dataset_local[TARGET_COLUMN] = pd.Series(
                target_encoder.fit_transform(dataset_local[TARGET_COLUMN]), index=dataset_local.index
            )
            joblib.dump(target_encoder, MODEL_FOLDER / "target_encoder.joblib")

            scaler = StandardScaler()
            dataset_local[NUMERIC_COLUMNS] = scaler.fit_transform(dataset_local[NUMERIC_COLUMNS])
            joblib.dump(scaler, MODEL_FOLDER / "scaler.joblib")

            X = dataset_local[FEATURE_COLUMNS]
            y = dataset_local[TARGET_COLUMN]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            return redirect("trainmodels")

        return redirect("AdminHome")

    except Exception as exc:
        return HttpResponse(f"Error during retraining: {str(exc)}", status=500)

#Model Performance View : This view retrieves the training metrics for the machine learning models from a JSON file and renders a page to display these metrics. It checks if the metrics file exists, loads the metrics data if available, and passes it to the template for rendering. If there is an error loading the metrics, it returns an HTTP response with an error message and a 500 status code. This allows administrators to view detailed performance metrics for the trained models in the admin interface.
def model_performance(request):
    try:
        metrics_data = _load_saved_training_metrics()
        model_metrics = []
        has_confusion_matrix = False

        if metrics_data and isinstance(metrics_data.get("metrics"), dict):
            for algo_name, metric_values in metrics_data["metrics"].items():
                confusion_matrix_values = metric_values.get("confusion_matrix")
                if confusion_matrix_values:
                    has_confusion_matrix = True
                model_metrics.append(
                    {
                        "name": algo_name,
                        "accuracy": metric_values.get("accuracy", 0.0),
                        "precision": metric_values.get("precision", 0.0),
                        "recall": metric_values.get("recall", 0.0),
                        "f1_score": metric_values.get("f1_score", 0.0),
                        "confusion_matrix": confusion_matrix_values,
                    }
                )

        return render_jinja(
            request,
            "AdminApp/ModelPerformance.html",
            {
                "metrics_data": metrics_data,
                "model_metrics": model_metrics,
                "has_confusion_matrix": has_confusion_matrix,
            },
        )
    except Exception as exc:
        return HttpResponse(f"Error loading metrics: {str(exc)}", status=500)

def connect_google_fit(request):
    client_secret_file = _resolve_google_client_secret_file()
    if client_secret_file is None:
        return HttpResponse(
            "Google OAuth client secret file not found. Add client_secret.json or client-secret.json in the project root.",
            status=500,
        )

    current_redirect_uri = request.build_absolute_uri("/google-fit-callback/").rstrip("/") + "/"
    registered_redirect_uris = _load_registered_redirect_uris(client_secret_file)
    redirect_uri = current_redirect_uri
    if registered_redirect_uris and current_redirect_uri not in registered_redirect_uris:
        redirect_uri = registered_redirect_uris[0]
        parsed = urlparse(redirect_uri)
        if parsed.scheme and parsed.netloc:
            return redirect(f"{parsed.scheme}://{parsed.netloc}/connect-google-fit/")

    flow = Flow.from_client_secrets_file(
        client_secret_file,
        scopes=GOOGLE_FIT_SCOPES,
        redirect_uri=redirect_uri
    )

    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true"
    )

    # Save PKCE verifier + state
    request.session["state"] = state
    request.session["code_verifier"] = flow.code_verifier
    request.session["google_redirect_uri"] = redirect_uri

    return redirect(authorization_url)

def google_fit_callback(request):
    client_secret_file = _resolve_google_client_secret_file()
    if client_secret_file is None:
        return HttpResponse(
            "Google OAuth client secret file not found. Add client_secret.json or client-secret.json in the project root.",
            status=500,
        )

    redirect_uri = request.session.get("google_redirect_uri")
    if not redirect_uri:
        redirect_uri = request.build_absolute_uri("/google-fit-callback/").rstrip("/") + "/"

    state = request.session.get("state")
    code_verifier = request.session.get("code_verifier")
    if not state or not code_verifier:
        return HttpResponse("Missing OAuth session state. Start again from Connect Google Fit.", status=400)

    flow = Flow.from_client_secrets_file(
        client_secret_file,
        scopes=GOOGLE_FIT_SCOPES,
        state=state,
        redirect_uri=redirect_uri
    )

    # restore PKCE verifier
    flow.code_verifier = code_verifier

    try:
        flow.fetch_token(authorization_response=request.build_absolute_uri())
    except Exception as exc:
        return HttpResponse(f"OAuth callback failed: {str(exc)}", status=400)

    credentials = flow.credentials

    request.session["google_token"] = credentials.token
    return redirect("fetch_google_fit_data")

def fetch_google_fit_data(request):

    token = request.session.get("google_token")
    patient = _get_session_patient(request)
    local_fit_payload = _build_local_fit_payload(patient)

    if not token:
        if not is_online() and local_fit_payload is not None:
            return _run_prediction(request, _prime_fit_session(request, local_fit_payload))
        return redirect("connect_google_fit")

    if not is_online() and local_fit_payload is not None:
        return _run_prediction(request, _prime_fit_session(request, local_fit_payload))

    headers = {
        "Authorization": f"Bearer {token}"
    }

    end_time = int(time.time() * 1000)
    start_time = end_time - (7 * 24 * 60 * 60 * 1000)

    body = {
        "aggregateBy": [
            {"dataTypeName": "com.google.step_count.delta"},
            {"dataTypeName": "com.google.heart_rate.bpm"},
            {"dataTypeName": "com.google.calories.expended"}
        ],
        "bucketByTime": {"durationMillis": 86400000},
        "startTimeMillis": start_time,
        "endTimeMillis": end_time
    }

    try:
        response = requests.post(
            "https://www.googleapis.com/fitness/v1/users/me/dataset:aggregate",
            headers=headers,
            json=body,
            timeout=30,
        )
    except requests.RequestException as exc:
        if local_fit_payload is not None:
            return _run_prediction(request, _prime_fit_session(request, local_fit_payload))
        return HttpResponse(f"Google Fit request failed: {str(exc)}", status=502)

    if response.status_code != 200:
        if local_fit_payload is not None:
            return _run_prediction(request, _prime_fit_session(request, local_fit_payload))
        return HttpResponse(
            f"Google Fit API error ({response.status_code}): {response.text}",
            status=502,
            content_type="text/plain",
        )

    try:
        data = response.json()
    except ValueError:
        if local_fit_payload is not None:
            return _run_prediction(request, _prime_fit_session(request, local_fit_payload))
        return HttpResponse("Google Fit API returned non-JSON response.", status=502)

    # Collect per-day arrays for charts
    daily_steps = []
    daily_calories = []
    daily_heart_rate = []
    day_labels = []
    day_dates = []

    from datetime import timezone as _tz
    for bucket in data.get("bucket", []):
        bucket_start_ms = int(bucket.get("startTimeMillis", 0))
        dt = datetime.fromtimestamp(bucket_start_ms / 1000, tz=_tz.utc)
        day_labels.append(dt.strftime("%b %d"))
        day_dates.append(dt.date().isoformat())

        day_steps = 0
        day_calories = 0.0
        day_hr = None

        for dataset in bucket.get("dataset", []):
            source = dataset.get("dataSourceId", "")
            for point in dataset.get("point", []):
                value = point.get("value", [{}])[0]
                if "step_count.delta" in source:
                    day_steps += value.get("intVal", 0)
                elif "calories.expended" in source:
                    day_calories += value.get("fpVal", 0)
                elif "heart_rate" in source:
                    day_hr = value.get("fpVal")

        daily_steps.append(day_steps)
        daily_calories.append(round(day_calories))
        daily_heart_rate.append(round(day_hr, 1) if day_hr is not None else None)

    # Averages per day for prediction features
    num_days = max(len(daily_steps), 1)
    avg_steps = round(sum(daily_steps) / num_days)
    avg_calories = round(sum(daily_calories) / num_days)
    heart_rate = next((hr for hr in reversed(daily_heart_rate) if hr is not None), None)

    # Store daily arrays in session for chart rendering
    request.session["fit_daily_data"] = {
        "day_labels": day_labels,
        "day_dates": day_dates,
        "daily_steps": daily_steps,
        "daily_calories": daily_calories,
        "daily_heart_rate": daily_heart_rate,
        "daily_stress": [],
    }

    patient = _get_session_patient(request)

    watch_data = {
        "age": patient.age if patient is not None else "",
        "gender": patient.gender if patient is not None else "",
        "resting_heart_rate": heart_rate if heart_rate is not None else "",
        "stress_level": "",
        "sleep_duration_hours": "",
        "sleep_quality_score": "",
        "steps_per_day": avg_steps,
        "calories_burned": avg_calories,
        "blood_oxygen_level": "",
        "activity_level": "",
    }

    request.session["watch_data"] = watch_data
    print("Fetched Google Fit data:", watch_data)
    missing_fields = [field for field in REQUIRED_FIELDS if watch_data.get(field) in [None, "", "null"]]
    if missing_fields:
        return render(
            request,
            "UserApp/manual_missing_fields.html",
            {
                "missing_fields": missing_fields,
                "prefilled_data": watch_data,
            },
            using="django",
        )
    return _run_prediction(request, watch_data)



def doctor_login(request):

    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")

        try:
            doctor = Doctor.objects.get(email=email, password=password)

            # Ensure stale patient session data cannot affect doctor access checks.
            request.session.pop("username", None)
            request.session.pop("patient_id", None)
            request.session['doctor_id'] = doctor.id
            request.session['doctor_name'] = doctor.name

            return redirect("doctor_dashboard")

        except Doctor.DoesNotExist:
            return render(request, "doctor/login.html", {
                "error": "Invalid Email or Password"
            })

    return render(request, "doctor/login.html")


def doctor_dashboard(request):

    doctor_id = request.session.get("doctor_id")

    if not doctor_id:
        return redirect("doctor_login")

    doctor = get_object_or_404(Doctor, id=doctor_id)
    latest_predictions = Prediction.objects.filter(patient=OuterRef("pk")).order_by("-created_at")
    assigned_patient_count = Patient.objects.filter(doctor=doctor).count()
    patient_queue = list(
        Patient.objects.filter(doctor=doctor)
        .annotate(
            latest_risk_score=Subquery(latest_predictions.values("risk_score")[:1]),
            latest_priority=Subquery(latest_predictions.values("priority")[:1]),
            latest_prediction_id=Subquery(latest_predictions.values("id")[:1]),
            latest_explanation=Subquery(latest_predictions.values("explanation_text")[:1]),
            latest_data_source=Subquery(latest_predictions.values("data_source")[:1]),
            latest_follow_up=Subquery(latest_predictions.values("recommended_follow_up")[:1]),
            latest_ai_explanation=Subquery(latest_predictions.values("readable_ai_explanation")[:1]),
            latest_recovery_score=Subquery(latest_predictions.values("nervous_system_recovery_score")[:1]),
        )
        .filter(latest_prediction_id__isnull=False)
        .order_by("-latest_risk_score")
    )

    grouped_patients = {
        Prediction.PRIORITY_HIGH: [patient for patient in patient_queue if getattr(patient, "latest_priority", None) == Prediction.PRIORITY_HIGH],
        Prediction.PRIORITY_MEDIUM: [patient for patient in patient_queue if getattr(patient, "latest_priority", None) == Prediction.PRIORITY_MEDIUM],
        Prediction.PRIORITY_LOW: [patient for patient in patient_queue if getattr(patient, "latest_priority", None) == Prediction.PRIORITY_LOW],
    }
    priority_sections = [
        (Prediction.PRIORITY_HIGH, grouped_patients[Prediction.PRIORITY_HIGH]),
        (Prediction.PRIORITY_MEDIUM, grouped_patients[Prediction.PRIORITY_MEDIUM]),
        (Prediction.PRIORITY_LOW, grouped_patients[Prediction.PRIORITY_LOW]),
    ]

    return render_jinja(
        request,
        "doctor/dashboard.html",
        {
            "doctor": doctor,
            "patient_groups": grouped_patients,
            "chatbot_patients": patient_queue,
            "priority_sections": priority_sections,
            "assigned_patient_count": assigned_patient_count,
            "total_patients": len(patient_queue),
            "high_count": len(grouped_patients[Prediction.PRIORITY_HIGH]),
            "medium_count": len(grouped_patients[Prediction.PRIORITY_MEDIUM]),
            "low_count": len(grouped_patients[Prediction.PRIORITY_LOW]),
        },
    )


def doctor_patient_detail(request, patient_id):
    doctor_id = request.session.get("doctor_id")
    if not doctor_id:
        return redirect("doctor_login")

    doctor = get_object_or_404(Doctor, id=doctor_id)
    patient = get_object_or_404(Patient.objects.select_related("doctor"), id=patient_id, doctor=doctor)

    if request.method == "POST":
        note_text = request.POST.get("note", "").strip()
        if note_text:
            DoctorNote.objects.create(doctor=doctor, patient=patient, note=note_text)
        return redirect("doctor_patient_detail", patient_id=patient.pk)

    latest_prediction = Prediction.objects.filter(patient=patient).order_by("-created_at").first()
    metrics = list(HealthMetrics.objects.filter(patient=patient).order_by("-date")[:14])
    metrics.reverse()
    display_metrics = metrics[-7:] if len(metrics) >= 7 else metrics

    average_steps = round(sum(metric.steps for metric in display_metrics) / len(display_metrics), 1) if display_metrics else None
    available_sleep = [metric.sleep_hours for metric in display_metrics if metric.sleep_hours is not None]
    average_sleep = round(sum(available_sleep) / len(available_sleep), 1) if available_sleep else None
    available_heart_rate = [metric.heart_rate for metric in display_metrics if metric.heart_rate is not None]
    average_heart_rate = round(sum(available_heart_rate) / len(available_heart_rate), 1) if available_heart_rate else None
    available_stress = [metric.stress_level for metric in display_metrics if metric.stress_level is not None]
    average_stress = round(sum(available_stress) / len(available_stress), 1) if available_stress else None
    average_calories = round(sum(metric.calories_burned for metric in display_metrics) / len(display_metrics), 1) if display_metrics else None

    metric_cards = [
        {
            "label": "Steps",
            "average": average_steps,
            "baseline": BASELINES["steps"],
            "interpretation": _metric_interpretation("Steps", average_steps, BASELINES["steps"]),
        },
        {
            "label": "Sleep",
            "average": average_sleep,
            "baseline": BASELINES["sleep_hours"],
            "interpretation": _metric_interpretation("Sleep", average_sleep, BASELINES["sleep_hours"]),
        },
        {
            "label": "Heart Rate",
            "average": average_heart_rate,
            "baseline": BASELINES["heart_rate"],
            "interpretation": _metric_interpretation("Heart Rate", average_heart_rate, BASELINES["heart_rate"]),
        },
        {
            "label": "Stress",
            "average": average_stress,
            "baseline": BASELINES["stress_level"],
            "interpretation": _metric_interpretation("Stress", average_stress, BASELINES["stress_level"]),
        },
        {
            "label": "Calories",
            "average": average_calories,
            "baseline": BASELINES["calories_burned"],
            "interpretation": _metric_interpretation("Calories", average_calories, BASELINES["calories_burned"]),
        },
    ]

    chart_context = _build_trend_payload(display_metrics)

    # New analytics: lifestyle score, risk projection, AI insight cards, risk history
    lifestyle_score = _compute_lifestyle_score(metrics)
    risk_projection = _generate_risk_projection(metrics, latest_prediction.risk_score) if latest_prediction else None
    ai_insights = _generate_ai_insight_cards(metrics)
    prediction_history = list(Prediction.objects.filter(patient=patient).order_by("created_at")[:10])
    risk_history_labels = [p.created_at.strftime("%b %d") for p in prediction_history]
    risk_history_values = [p.risk_score for p in prediction_history]

    return render_jinja(
        request,
        "doctor/patient_detail.html",
        {
            "doctor": doctor,
            "patient": patient,
            "latest_prediction": latest_prediction,
            "metric_cards": metric_cards,
            "notes": DoctorNote.objects.filter(patient=patient).select_related("doctor").order_by("-created_at"),
            "latest_prediction_risk_factors": _split_text_list(latest_prediction.risk_factors) if latest_prediction else [],
            "latest_prediction_recommendations": _split_text_list(latest_prediction.lifestyle_recommendations) if latest_prediction else [],
            "triage_info": {
                "disease_name": patient.suggested_disease or "Not available",
                "risk_level": patient.disease_risk_level or "Not available",
                "suggested_tests": [item.strip() for item in patient.suggested_tests.split(",") if item.strip()] if patient.suggested_tests else [],
                "assigned_doctor": patient.assigned_doctor if patient.assigned_doctor is not None else patient.doctor,
            },
            "lifestyle_score": lifestyle_score,
            "risk_projection": risk_projection,
            "ai_insights": ai_insights,
            "risk_history_labels": risk_history_labels,
            "risk_history_values": risk_history_values,
            **chart_context,
        },
    )


def doctor_logout(request):

    request.session.flush()

    return redirect("doctor_login")


def _chatbot_detect_intent(question_text):
    """Map a free-text question to the most relevant response category.

    Patterns are checked in priority order so more specific matches win over
    generic ones (e.g. 'what is stress level' hits explain_stress_term before
    the generic 'stress' intent).
    """
    question = (question_text or "").strip().lower()

    ordered_patterns = [
        # -- Terminology explanations
        ("explain_shap", [
            r"\bwhat\s+is\s+shap\b",
            r"\bhow\s+(does\s+)?shap\b",
            r"\bshap\s+explanation\b",
            r"\bshap\b.*\bwork\b",
            r"\bexplain\s+shap\b",
            r"\bshap\s+method\b",
        ]),
        ("explain_lifestyle_risk", [
            r"\bwhat\s+is\s+lifestyle\s+disorder\b",
            r"\bwhat\s+is\s+(the\s+)?risk\s+score\b",
            r"\bwhat\s+does\s+(the\s+)?risk\b.*\bmean\b",
            r"\bwhat\s+is\s+lifestyle\s+risk\b",
            r"\bexplain\s+(the\s+)?risk\s+score\b",
            r"\bwhat\s+(is|are)\s+lifestyle\s+disorder(s)?\b",
        ]),
        ("explain_nervous", [
            r"\bwhat\s+is\s+nervous\s+system\b",
            r"\bnervous\s+system\s+balance\b",
            r"\brecovery\s+score\b",
            r"\bnervous\s+system\s+recovery\b",
            r"\bwhat\s+is\s+(the\s+)?recovery\s+score\b",
        ]),
        ("explain_stress_term", [
            r"\bwhat\s+is\s+stress\s+level\b",
            r"\bwhat\s+does\s+stress\s+level\b",
            r"\bhow\s+is\s+stress\s+(level\s+)?measured\b",
        ]),
        ("deficiency", [
            r"\bfeel\s+(tired|exhausted|fatigued|weak|drained)\b",
            r"\bi\s+(am|feel|have\s+been)\s+(tired|exhausted|drained|weak)\b",
            r"\bno\s+energy\b",
            r"\blow\s+energy\b",
            r"\balways\s+(tired|exhausted)\b",
            r"\bconstantly\s+tired\b",
            r"\bnot\s+sleeping\s+well\b",
            r"\bcan.t\s+sleep\b",
            r"\bcannot\s+sleep\b",
            r"\bvitamin\s+d\b",
            r"\bvitamin\s+b\d*\b",
            r"\bb12\b",
            r"\bfatigue\b",
            r"\bwhy\s+(am\s+i\s+)?(tired|exhausted|fatigued|weak)\b",
        ]),
        # stress before risk to prevent 'why is my stress high' matching risk
        ("stress", [
            r"\bstress\s+(is|level|high|score|too)\b",
            r"\bwhy\s+(is\s+)?(my\s+)?stress\b",
            r"\breduce\s+stress\b",
            r"\bmy\s+stress\b",
            r"\bhow\s+to\s+(lower|reduce|manage)\s+stress\b",
            r"\banxious\b",
            r"\banxiety\b",
            r"\boverwhelmed\b",
        ]),
        ("risk", [
            r"\b(why|what|how)\s+(is|was|am|are)\s+.*\brisk\s+score\b",
            r"\bhigh(er)?\s+priority\b",
            r"\bmarked\s+(as\s+)?high\s+priority\b",
            r"\brisk\s+score\b",
            r"\blifestyle\s+disorder\s+risk\b",
            r"\brisk\s+factor(s)?\b",
            r"\bwhy\s+.*\bpriority\b",
            r"\bpriority.*\bwhy\b",
            r"\bwhy\s+am\s+i\s+(in\s+)?high\s+priority\b",
            r"\b(why|what|how)\s+(is|was|am|are)\s+.*\brisk\b",
            r"\brisk\b",
        ]),
        ("heart_rate", [
            r"\bheart\s*rate\b",
            r"\bpulse\b",
            r"\bbpm\b",
            r"\bnormal\s+heart\b",
            r"\bheart\s+beat\b",
            r"\bcardiac\b",
        ]),
        ("sleep", [
            r"\bsleep\s+quality\b",
            r"\bsleep\s+(hours|duration|pattern)\b",
            r"\bsleeping\b",
            r"\binsomnia\b",
            r"\bnot\s+getting\s+enough\s+sleep\b",
            r"\bsleep\b",
        ]),
        ("steps", [
            r"\bsteps?\b",
            r"\bwalking\b",
            r"\bnot\s+(active|moving)\b",
            r"\bsedentary\b",
            r"\bexercise\s+enough\b",
            r"\bphysical\s+activity\b",
            r"\bdaily\s+activity\b",
        ]),
        ("blood_oxygen", [
            r"\bblood\s+oxygen\b",
            r"\bspo2\b",
            r"\boxygen\s+(level|saturation|reading)\b",
        ]),
        ("calories", [
            r"\bcalorie(s)?\b",
            r"\bcaloric\b",
            r"\benergy\s+burn(ed)?\b",
            r"\bhow\s+much\s+(am\s+i\s+)?burning\b",
        ]),
        ("habit_insights", [
            r"\bunhealthy\s+habit(s)?\b",
            r"\blifestyle\s+habit(s)?\b",
            r"\bmy\s+habit(s)?\b",
            r"\bhealth\s+pattern(s)?\b",
        ]),
        ("advice", [
            r"\bhow\s+(can|do|should)\s+(i|you)\s+improve\b",
            r"\badvice\b",
            r"\brecommend\b",
            r"\bwhat\s+should\s+i\s+do\b",
            r"\bimprove\s+my\s+(health|lifestyle)\b",
            r"\bhow\s+to\s+(get\s+better|be\s+healthier)\b",
            r"\bwhat\s+(to|can)\s+(do|change|start)\b",
        ]),
        ("summary", [
            r"\bsummary\b",
            r"\boverview\b",
            r"\bstatus\b",
            r"\bhow\s+am\s+i\s+doing\b",
            r"\bmy\s+health\b",
            r"\bgeneral\b",
        ]),
    ]
    for intent_name, regex_list in ordered_patterns:
        for regex in regex_list:
            if re.search(regex, question):
                return intent_name
    return "summary"


def _chatbot_patient_snapshot(patient):
    latest_metric = HealthMetrics.objects.filter(patient=patient).order_by("-date").first()
    latest_prediction = Prediction.objects.filter(patient=patient).order_by("-created_at").first()
    return latest_metric, latest_prediction


def _chatbot_permission_context(request, patient_id, requested_role=""):
    role = None
    patient = None

    requested_role = (requested_role or "").strip().lower()

    # If role is explicitly provided, make it authoritative.
    if requested_role == "patient":
        session_patient = _get_session_patient(request)
        if session_patient is not None and getattr(session_patient, "pk", None) == patient_id:
            return "patient", session_patient
        return "patient", None

    if requested_role == "doctor":
        doctor_id = request.session.get("doctor_id")
        if doctor_id:
            patient = Patient.objects.filter(id=patient_id, doctor_id=doctor_id).first()
            if patient is None:
                patient = Patient.objects.filter(id=patient_id, assigned_doctor_id=doctor_id).first()
            if patient is not None:
                return "doctor", patient
        return "doctor", None

    doctor_id = request.session.get("doctor_id")
    if doctor_id:
        role = "doctor"
        patient = Patient.objects.filter(id=patient_id, doctor_id=doctor_id).first()
        if patient is None:
            patient = Patient.objects.filter(id=patient_id, assigned_doctor_id=doctor_id).first()
    else:
        role = "patient"
        session_patient = _get_session_patient(request)
        if session_patient is not None and getattr(session_patient, "pk", None) == patient_id:
            patient = session_patient

    return role, patient


def _chatbot_metric_summary(metric, prediction):
    return {
        "heart_rate": metric.heart_rate if metric is not None else None,
        "stress_level": metric.stress_level if metric is not None else None,
        "sleep_hours": metric.sleep_hours if metric is not None else None,
        "sleep_quality": metric.sleep_quality if metric is not None else None,
        "steps": metric.steps if metric is not None else None,
        "calories": metric.calories_burned if metric is not None else None,
        "blood_oxygen": metric.blood_oxygen if metric is not None else None,
        "activity_level": metric.activity_level if metric is not None else "",
        "lifestyle_disorder_risk_score": prediction.risk_score if prediction is not None else None,
    }


# ── Term explanation responses ────────────────────────────────────────────────

def _chatbot_term_explanation(intent):
    """Return a plain-language explanation for a clinical or AI system term."""
    explanations = {
        "explain_shap": (
            "SHAP (SHapley Additive exPlanations) is the AI method used by this system to explain "
            "which of your health metrics contributed most to your risk prediction — and in which direction. "
            "A high positive SHAP value for a metric means it pushed your risk score upward; a negative value "
            "means it helped lower your score. This makes the AI's decision transparent and understandable "
            "rather than a 'black box'."
        ),
        "explain_lifestyle_risk": (
            "The Lifestyle Disorder Risk Score is a number between 0 and 100 calculated by the AI model "
            "using your wearable health data — heart rate, sleep, steps, stress, blood oxygen, and calorie burn. "
            "A higher score indicates a greater likelihood of developing lifestyle-related conditions such as "
            "stress disorders, sleep imbalance, metabolic issues, or cardiovascular strain. "
            "Scores above 70 are flagged as high risk and are prioritised for medical review."
        ),
        "explain_nervous": (
            "The Nervous System Recovery Score (0–100) measures how well your body is recovering from daily "
            "physiological stress. It is calculated from your resting heart rate, sleep duration, sleep quality, "
            "and stress level. A score above 70 means your nervous system is recovering well. "
            "Scores between 40–70 suggest moderate strain, and scores below 40 indicate the body is under "
            "sustained stress and needs rest, sleep improvement, or active stress reduction."
        ),
        "explain_stress_term": (
            "The Stress Level metric (scale 1–10) is derived from wearable pulse signals and heart rate "
            "variability data, indicating how much strain your nervous system is under. "
            "A score of 1–3 is healthy, 4–6 is moderate stress, and 7–10 signals high chronic stress — "
            "which over time negatively impacts sleep quality, heart rate, immune function, and metabolic health."
        ),
    }
    return explanations.get(
        intent,
        "I am sorry, I do not have an explanation for that term right now. Please ask your healthcare provider."
    )


# ── Metric-specific response functions ───────────────────────────────────────

def _chatbot_summary_response(role, patient, metric, prediction):
    """Rich health overview — used when the question is general ('how am I doing?')."""
    m = _chatbot_metric_summary(metric, prediction)
    risk_score = m["lifestyle_disorder_risk_score"]
    risk_text = f"{risk_score:.1f}" if isinstance(risk_score, (int, float)) else "not yet available"
    priority_text = f" ({prediction.priority} priority)" if prediction is not None else ""

    def _fmt(val, unit="", decimals=1):
        return f"{val:.{decimals}f}{unit}" if isinstance(val, (int, float)) else "N/A"

    lines = [
        f"Health summary for {patient.name}:",
        f"• Lifestyle Disorder Risk Score: {risk_text}{priority_text}",
        f"• Heart Rate: {_fmt(m['heart_rate'], ' bpm')} (healthy: ≤{BASELINES['heart_rate']} bpm)",
        f"• Stress Level: {_fmt(m['stress_level'], '/10')} (healthy: ≤{BASELINES['stress_level']}/10)",
        f"• Sleep: {_fmt(m['sleep_hours'], ' hrs')} | Quality {_fmt(m['sleep_quality'], '/10')} "
        f"(targets: ≥{BASELINES['sleep_hours']} hrs, ≥{BASELINES['sleep_quality']}/10)",
        f"• Daily Steps: {int(m['steps']):,}" if isinstance(m["steps"], (int, float)) else "• Daily Steps: N/A",
        f"• Blood Oxygen: {_fmt(m['blood_oxygen'], '%')} (healthy: ≥{BASELINES['blood_oxygen']}%)",
        f"• Calories Burned: {int(m['calories']):,} kcal"
        if isinstance(m["calories"], (int, float))
        else "• Calories Burned: N/A",
    ]

    # Flag the most urgent concern
    concerns = []
    if isinstance(risk_score, (int, float)) and risk_score >= 70:
        concerns.append("risk score is critically high")
    if isinstance(m["stress_level"], (int, float)) and m["stress_level"] >= 7:
        concerns.append("stress level is elevated")
    if isinstance(m["sleep_hours"], (int, float)) and m["sleep_hours"] < 6:
        concerns.append("sleep duration is below the safe threshold")
    if isinstance(m["steps"], (int, float)) and m["steps"] < 4000:
        concerns.append("daily activity is very low")

    if concerns:
        lines.append("\nKey concern(s): " + "; ".join(concerns) + ".")
        lines.append("Ask me about any specific metric for a deeper explanation.")
    else:
        lines.append("\nAll monitored metrics appear within an acceptable range. Keep up the good work!")

    if role == "doctor" and prediction is not None:
        follow_up = getattr(prediction, "recommended_follow_up", None)
        if follow_up:
            lines.append(f"\nRecommended follow-up: {follow_up}")

    return "\n".join(lines)


def _chatbot_risk_response(role, patient, prediction):
    """Explain the lifestyle disorder risk score and its contributing factors."""
    if prediction is None:
        return (
            f"No prediction is available yet for {patient.name}. "
            "Please submit health metrics to generate a risk assessment."
        )

    risk_factors = _split_text_list(prediction.risk_factors)
    factors_text = (
        ", ".join(risk_factors[:4]) if risk_factors else "combined wearable and lifestyle signals"
    )
    explain_text = (prediction.readable_ai_explanation or prediction.explanation_text or "").strip()
    stress_insight = getattr(prediction, "stress_sleep_insight", None) or ""
    recovery = getattr(prediction, "nervous_system_recovery_score", None)
    recovery_text = f" Nervous system recovery score: {recovery:.0f}/100." if isinstance(recovery, (int, float)) else ""

    score = prediction.risk_score
    if score >= 80:
        risk_desc = "critically high — immediate lifestyle intervention is advised"
    elif score >= 70:
        risk_desc = "high — significant lifestyle modification is recommended"
    elif score >= 50:
        risk_desc = "moderate — some areas of concern that should be addressed"
    else:
        risk_desc = "within a manageable range"

    if role == "doctor":
        response = (
            f"{patient.name} has a lifestyle disorder risk score of {score:.1f} ({prediction.priority} priority) "
            f"— {risk_desc}. "
            f"Top contributing factors: {factors_text}.{recovery_text}"
        )
        if explain_text:
            response += f" AI interpretation: {explain_text}"
        if stress_insight:
            response += f" Stress-sleep pattern: {stress_insight}"
        follow_up = getattr(prediction, "recommended_follow_up", None)
        if follow_up:
            response += f" Suggested follow-up: {follow_up}"
        return response

    response = (
        f"Your lifestyle disorder risk score is {score:.1f} ({prediction.priority}) — {risk_desc}. "
        f"The main factors driving this score are: {factors_text}.{recovery_text}"
    )
    if explain_text:
        response += f" AI explanation: {explain_text}"
    if stress_insight:
        response += f" Stress-sleep insight: {stress_insight}"
    return response


def _chatbot_heart_rate_response(role, patient, metric):
    """Analyse resting heart rate and explain what it means for the patient."""
    if metric is None or metric.heart_rate is None:
        return "Heart rate data is not available yet. Please sync your wearable device or enter data manually."

    heart_rate = float(metric.heart_rate)
    baseline = BASELINES["heart_rate"]
    person = f"{patient.name}'s" if role == "doctor" else "Your"

    if heart_rate >= 100:
        status = "high (tachycardia range)"
        impact = (
            "A resting heart rate above 100 bpm can indicate dehydration, elevated stress, poor cardiovascular "
            "fitness, or an underlying condition. This should be monitored closely and evaluated by a physician."
        )
    elif heart_rate >= 90:
        status = "above optimal resting range"
        impact = (
            "While not immediately dangerous, a consistently elevated resting heart rate above 90 bpm "
            "is a risk marker for cardiovascular strain and lifestyle disorders."
        )
    elif heart_rate < 50:
        status = "low (bradycardia range)"
        impact = (
            "A resting heart rate below 50 bpm may be normal for highly trained athletes, but for others "
            "it can signal an issue with the heart's electrical system and warrants medical evaluation."
        )
    elif heart_rate < 60:
        status = "below average — possibly indicating good fitness"
        impact = "Lower resting heart rates are often associated with good cardiovascular fitness."
    else:
        status = "within a healthy resting range"
        impact = "A resting heart rate in the 60–80 bpm zone indicates good cardiovascular health."

    return (
        f"{person} latest resting heart rate is {heart_rate:.1f} bpm — {status} "
        f"(healthy baseline: ~{baseline} bpm). {impact}"
    )


def _chatbot_sleep_response(role, patient, metric):
    """Analyse sleep hours and quality and explain their health impact."""
    if metric is None:
        return "Sleep data is not available yet. Please enter your sleep information or sync your wearable."

    sleep_hours = metric.sleep_hours
    sleep_quality = metric.sleep_quality
    if sleep_hours is None and sleep_quality is None:
        return "Both sleep hours and sleep quality are currently unavailable for this patient."

    baseline_hours = BASELINES["sleep_hours"]
    baseline_quality = BASELINES["sleep_quality"]
    person = f"{patient.name}'s" if role == "doctor" else "Your"
    pronoun = "their" if role == "doctor" else "your"

    parts = []

    if isinstance(sleep_hours, (int, float)):
        if sleep_hours < 5:
            parts.append(
                f"{person} sleep duration is critically low at {sleep_hours:.1f} hours "
                f"(minimum recommended: {baseline_hours} hours). "
                "Sleeping fewer than 5 hours consistently is strongly associated with increased "
                "stress hormones, impaired immune function, and elevated lifestyle disorder risk."
            )
        elif sleep_hours < 6:
            parts.append(
                f"{person} sleep duration is {sleep_hours:.1f} hours, below the recommended {baseline_hours}-hour minimum. "
                "This level of sleep deprivation increases cortisol, raises resting heart rate, and reduces "
                f"{pronoun} body's ability to recover from daily physiological stress."
            )
        elif sleep_hours < baseline_hours:
            parts.append(
                f"{person} sleep duration is {sleep_hours:.1f} hours — slightly below the {baseline_hours}-hour target. "
                "Aiming for at least 30 more minutes per night would improve recovery metrics."
            )
        else:
            parts.append(
                f"{person} sleep duration is {sleep_hours:.1f} hours — meeting the recommended {baseline_hours}-hour target. Good."
            )

    if isinstance(sleep_quality, (int, float)):
        if sleep_quality <= 4:
            parts.append(
                f"Sleep quality score is {sleep_quality:.1f}/10 — poor. "
                "Even with adequate hours, poor sleep quality prevents the deep sleep cycles needed for "
                "hormonal recovery, memory consolidation, and nervous system repair. "
                "Consider reviewing sleep environment, screen exposure, and caffeine intake."
            )
        elif sleep_quality <= 6:
            parts.append(
                f"Sleep quality score is {sleep_quality:.1f}/10 — below the target of {baseline_quality}/10. "
                "Inconsistent sleep quality may be contributing to elevated stress and fatigue patterns."
            )
        else:
            parts.append(
                f"Sleep quality score is {sleep_quality:.1f}/10 — strong. "
                "Good sleep quality supports nervous system recovery and hormonal balance."
            )

    return " ".join(parts) if parts else "Sleep data is present but values could not be interpreted at this time."


def _chatbot_stress_response(role, patient, metric):
    """Analyse the stress level metric with contextual lifestyle impact."""
    if metric is None or metric.stress_level is None:
        return "Stress level data is not currently available for this patient."

    stress = float(metric.stress_level)
    baseline = BASELINES["stress_level"]
    person = f"{patient.name}'s" if role == "doctor" else "Your"
    pronoun = "their" if role == "doctor" else "your"

    if stress >= 8:
        status = "critically high"
        impact = (
            f"This level of sustained stress significantly impairs sleep quality, raises {pronoun} resting "
            "heart rate, and accelerates lifestyle disorder risk. The nervous system is showing signs of "
            "chronic overload which increases the risk of stress-related cardiovascular and metabolic disorders."
        )
        action = (
            " Immediate recommended actions: implement structured breathing exercises (4-7-8 technique), "
            "reduce workload or screen time before bed, and consider a mindfulness or cognitive behavioural "
            "stress management programme."
        )
    elif stress >= 6:
        status = "elevated"
        impact = (
            f"Elevated stress disrupts {pronoun} sleep cycles, raises cortisol levels, and can increase "
            "blood pressure over time. This often co-occurs with poor sleep quality and reduced physical activity."
        )
        action = (
            " Recommended actions: schedule regular rest breaks, engage in 20–30 minutes of moderate "
            "physical activity daily, and aim for a consistent sleep schedule."
        )
    elif stress >= 4:
        status = "moderate"
        impact = (
            f"Moderate stress is manageable but should be monitored. If it persists over several weeks, "
            f"it may begin to affect {pronoun} sleep quality and energy levels."
        )
        action = " Consider incorporating daily brief mindfulness or relaxation practices."
    else:
        status = "within a healthy range"
        impact = (
            f"{person.capitalize()} stress level is well-managed — this is a protective factor against "
            "lifestyle disorders. Maintain current lifestyle balance."
        )
        action = ""

    return (
        f"{person} stress level is {stress:.1f}/10 — {status} "
        f"(healthy target: ≤{baseline}/10). {impact}{action}"
    )


def _chatbot_steps_response(role, patient, metric):
    """Analyse daily step count and interpret its effect on metabolic health."""
    if metric is None or metric.steps is None:
        return "Step count data is not currently available. Please sync your wearable or enter data manually."

    steps = float(metric.steps)
    baseline = BASELINES["steps"]
    person = f"{patient.name}'s" if role == "doctor" else "Your"

    if steps < 3000:
        status = "very low — severely sedentary"
        impact = (
            "Fewer than 3,000 steps per day is associated with significantly increased metabolic risk, "
            "weight gain, poor circulation, and elevated cardiovascular risk. This pattern directly "
            "contributes to lifestyle disorder development."
        )
        advice = "Start with a 15-minute walk twice daily and increase by 500 steps per week toward the 8,000-step goal."
    elif steps < 5000:
        status = "low — sedentary range"
        impact = (
            "Being in the sedentary range raises the risk of lifestyle-related conditions including "
            "metabolic imbalance and cardiovascular strain. Physical inactivity is one of the top modifiable "
            "risk factors for lifestyle disorders."
        )
        advice = "Aim for at least one 30-minute walk each day to move towards the active range."
    elif steps < baseline:
        status = "below the healthy target"
        impact = (
            f"Getting closer to the {baseline:,}-step target will noticeably improve metabolism, "
            "reduce stress response, and lower lifestyle disorder risk scores."
        )
        advice = "Try adding a 15-minute walk after meals — this alone can add 1,500–2,000 steps daily."
    elif steps < 12000:
        status = "meeting the healthy target"
        impact = (
            "Consistent achievement of 8,000+ steps is one of the strongest protective factors against "
            "lifestyle disorders, metabolic disease, and cardiovascular conditions."
        )
        advice = "Maintain this consistency. Consider adding light strength training for additional longevity benefits."
    else:
        status = "highly active"
        impact = "Excellent daily movement. High step counts are associated with significantly reduced all-cause mortality risk."
        advice = "Ensure adequate rest and caloric intake to support this activity level."

    return (
        f"{person} recorded step count is {int(steps):,} steps — {status} "
        f"(target: {baseline:,} steps/day). {impact} {advice}"
    )


def _chatbot_blood_oxygen_response(role, patient, metric):
    """Analyse blood oxygen saturation (SpO2) and explain its significance."""
    if metric is None or metric.blood_oxygen is None:
        return "Blood oxygen data is not currently available for this patient."

    spo2 = float(metric.blood_oxygen)
    baseline = BASELINES["blood_oxygen"]
    person = f"{patient.name}'s" if role == "doctor" else "Your"

    if spo2 < 90:
        status = "critically low"
        impact = (
            "An SpO2 level below 90% is a medical emergency indicator. This level of hypoxaemia means "
            "the blood is not carrying sufficient oxygen to vital organs. Immediate medical assessment "
            "is strongly recommended."
        )
    elif spo2 < 94:
        status = "below normal"
        impact = (
            "SpO2 below 94% indicates the blood is carrying less oxygen than required for normal physiological "
            "function. This can cause fatigue, shortness of breath, reduced physical performance, "
            "and brain fog. Respiratory assessment may be warranted."
        )
    elif spo2 < baseline:
        status = "slightly below ideal"
        impact = (
            f"While not immediately alarming, SpO2 below {baseline}% should be monitored over time "
            "to ensure it does not continue to decline. Good sleep position and aerobic fitness help maintain optimal levels."
        )
    else:
        status = "normal and healthy"
        impact = (
            "Blood oxygen is well within the healthy range, indicating good respiratory and cardiovascular "
            "function. No intervention required."
        )

    return (
        f"{person} blood oxygen saturation (SpO2) is {spo2:.1f}% — {status} "
        f"(healthy target: ≥{baseline}%). {impact}"
    )


def _chatbot_calories_response(role, patient, metric):
    """Analyse calorie burn and relate it to activity and metabolic health."""
    if metric is None or metric.calories_burned is None:
        return "Calorie expenditure data is not currently available for this patient."

    calories = float(metric.calories_burned)
    baseline = BASELINES["calories_burned"]
    person = f"{patient.name}'s" if role == "doctor" else "Your"

    if calories < 1200:
        status = "very low"
        impact = (
            "Very low calorie expenditure suggests minimal physical activity throughout the day. "
            "This significantly increases risk of metabolic disorders, weight gain, poor cardiovascular "
            "health, and contributes to lifestyle disorder development."
        )
    elif calories < 1600:
        status = "low"
        impact = (
            "Below-average calorie burn suggests insufficient daily movement. "
            "Increasing light activity such as walking, stretching, or household activity can raise this "
            "to a healthier range."
        )
    elif calories < baseline:
        status = "moderate — slightly below target"
        impact = (
            "Closer to the target than sedentary range, but increasing daily movement or exercise "
            "intensity can bring calorie burn to a more protective level."
        )
    else:
        status = "at or above a healthy level"
        impact = (
            "Good energy expenditure supports metabolic health, helps maintain a healthy body weight, "
            "and reduces lifestyle disorder risk."
        )

    return (
        f"{person} daily calorie expenditure is approximately {int(calories):,} kcal — {status} "
        f"(reference target: ~{baseline:,} kcal/day). {impact}"
    )


def _chatbot_deficiency_response(role, question, metric):
    """Suggest possible causes of fatigue or low-energy symptoms using patient metrics."""
    person = "The patient" if role == "doctor" else "You"
    pronoun = "their" if role == "doctor" else "your"
    possible_causes = []

    if metric is not None:
        if isinstance(metric.sleep_hours, (int, float)) and metric.sleep_hours < 6:
            possible_causes.append(
                f"Sleep deprivation — {pronoun} recorded sleep is {metric.sleep_hours:.1f} hours, "
                f"below the 7-hour minimum. This is the most common cause of persistent fatigue and low energy."
            )
        if isinstance(metric.sleep_quality, (int, float)) and metric.sleep_quality <= 5:
            possible_causes.append(
                f"Poor sleep quality — even with sufficient hours, {pronoun} sleep quality score of "
                f"{metric.sleep_quality:.1f}/10 means the body is not entering restorative deep-sleep cycles."
            )
        if isinstance(metric.stress_level, (int, float)) and metric.stress_level >= 6:
            possible_causes.append(
                f"Stress-related fatigue — {pronoun} stress level is {metric.stress_level:.1f}/10. "
                "Chronic high stress depletes energy reserves, disrupts the sleep-wake cycle, and causes "
                "persistent mental and physical exhaustion."
            )
        if isinstance(metric.steps, (int, float)) and metric.steps < 3000:
            possible_causes.append(
                f"Inactivity fatigue — only {int(metric.steps):,} steps recorded. "
                "Paradoxically, very low physical activity reduces circadian energy signals and can "
                "increase feelings of tiredness by reducing circulation and oxygen delivery to muscles."
            )
        if isinstance(metric.blood_oxygen, (int, float)) and metric.blood_oxygen < 95:
            possible_causes.append(
                f"Reduced blood oxygen ({metric.blood_oxygen:.1f}% SpO2) — low oxygen saturation directly "
                "causes fatigue, brain fog, and reduced physical and cognitive capacity."
            )
        if isinstance(metric.heart_rate, (int, float)) and metric.heart_rate >= 95:
            possible_causes.append(
                f"Elevated resting heart rate ({metric.heart_rate:.1f} bpm) — a high resting pulse can "
                "indicate the body is working harder than normal to maintain baseline function, "
                "contributing to a constant sense of tiredness."
            )

    # Nutritional deficiencies not detectable from wearable data
    possible_causes.extend([
        "Vitamin D deficiency — especially when sun exposure is limited. "
        "Symptoms include persistent fatigue, low mood, bone discomfort, and immune weakness.",
        "Vitamin B12 deficiency — causes neurological fatigue, weakness, numbness, and disrupted sleep. "
        "Common in individuals with low animal-product intake or digestive absorption issues.",
        "Iron deficiency (anaemia) — reduces the blood's oxygen-carrying capacity, "
        "causing tiredness, pale skin, shortness of breath, and poor concentration.",
        "Thyroid dysfunction (hypothyroidism) — an underactive thyroid slows metabolism "
        "causing fatigue, weight gain, cold sensitivity, and brain fog.",
    ])

    intro = (
        f"{person} reported symptoms may have one or more of the following causes "
        f"(based on {pronoun} health metrics and common clinical patterns):\n"
    )
    causes_text = "\n".join(f"  • {c}" for c in possible_causes)
    closing = (
        "\n\nRecommended next steps: consult a healthcare provider for a blood panel including "
        "CBC (complete blood count), ferritin, vitamin D (25-OH), vitamin B12, TSH (thyroid), "
        "and fasting glucose. These tests can confirm or rule out deficiency-related fatigue."
    )
    return intro + causes_text + closing


def _chatbot_habit_insights(role, patient, metric, prediction):
    """Provide a comprehensive analysis of the patient's lifestyle habit patterns."""
    if metric is None:
        return (
            f"No health data is available yet for {patient.name}. "
            "Please submit health metrics to receive a habit analysis."
        )

    person = patient.name if role == "doctor" else "you"
    pronoun_poss = f"{patient.name}'s" if role == "doctor" else "Your"
    concerns = []
    positives = []

    # Steps / physical activity
    if isinstance(metric.steps, (int, float)):
        if metric.steps < 3000:
            concerns.append(
                f"Very low daily activity — {int(metric.steps):,} steps (target: 8,000). "
                "This severely sedentary pattern is a primary driver of metabolic and cardiovascular risk."
            )
        elif metric.steps < 5000:
            concerns.append(
                f"Low daily activity — {int(metric.steps):,} steps (target: 8,000). "
                "Being in the sedentary range raises lifestyle disorder risk."
            )
        elif metric.steps >= 8000:
            positives.append(f"Good physical activity — {int(metric.steps):,} daily steps meeting the healthy target.")

    # Sleep duration
    if isinstance(metric.sleep_hours, (int, float)):
        if metric.sleep_hours < 6:
            concerns.append(
                f"Insufficient sleep — {metric.sleep_hours:.1f} hours per night (target: ≥7 hours). "
                "Chronic short sleep is one of the strongest modifiable risk factors for lifestyle disorders."
            )
        elif metric.sleep_hours >= 7.5:
            positives.append(f"Adequate sleep duration — {metric.sleep_hours:.1f} hours per night.")

    # Sleep quality
    if isinstance(metric.sleep_quality, (int, float)) and metric.sleep_quality <= 5:
        concerns.append(
            f"Poor sleep quality — score {metric.sleep_quality:.1f}/10. "
            "Even with sufficient sleep hours, poor quality prevents restorative deep sleep cycles."
        )

    # Stress
    if isinstance(metric.stress_level, (int, float)):
        if metric.stress_level >= 7:
            concerns.append(
                f"High chronic stress — {metric.stress_level:.1f}/10. "
                "Sustained high stress accelerates lifestyle disorder development and disrupts all other metrics."
            )
        elif metric.stress_level <= 3:
            positives.append(f"Well-managed stress — {metric.stress_level:.1f}/10.")

    # Calorie burn
    if isinstance(metric.calories_burned, (int, float)) and metric.calories_burned < 1500:
        concerns.append(
            f"Low energy expenditure — ~{int(metric.calories_burned):,} kcal/day. "
            "This level of calorie burn is consistent with a sedentary lifestyle pattern."
        )

    # Blood oxygen
    if isinstance(metric.blood_oxygen, (int, float)) and metric.blood_oxygen < 95:
        concerns.append(
            f"Below-optimal blood oxygen — {metric.blood_oxygen:.1f}% SpO2. "
            f"This may limit {pronoun_poss.lower()} physical and cognitive performance."
        )

    lines = [f"Lifestyle habit analysis for {person}:\n"]

    if concerns:
        lines.append("Areas of concern:")
        lines.extend(f"  ⚠  {c}" for c in concerns)

    if positives:
        lines.append("\nPositive habits to maintain:")
        lines.extend(f"  ✓  {p}" for p in positives)

    if not concerns and not positives:
        lines.append("Insufficient metric data to perform a full habit analysis. Please add more health records.")

    if prediction is not None and prediction.risk_score is not None:
        lines.append(f"\nOverall lifestyle disorder risk score: {prediction.risk_score:.1f} ({prediction.priority} priority).")
        recs = _split_text_list(prediction.lifestyle_recommendations) if prediction is not None else []
        if recs:
            lines.append("Top recommendation: " + recs[0])

    return "\n".join(lines)


def _chatbot_advice_response(role, patient, metric, prediction):
    """Generate personalised lifestyle improvement recommendations from stored or derived data."""
    recommendation_items = (
        _split_text_list(prediction.lifestyle_recommendations) if prediction is not None else []
    )

    # Derive metric-aware recommendations if stored ones are missing
    if not recommendation_items and metric is not None:
        if isinstance(metric.sleep_hours, (int, float)) and metric.sleep_hours < 7:
            recommendation_items.append(
                f"Improve sleep duration: aim for at least 7 hours. "
                f"Current sleep is {metric.sleep_hours:.1f} hours — a {7 - metric.sleep_hours:.1f}-hour deficit."
            )
        if isinstance(metric.stress_level, (int, float)) and metric.stress_level >= 6:
            recommendation_items.append(
                f"Reduce stress (current: {metric.stress_level:.1f}/10) through structured breathing, "
                "mindfulness, and limiting stressors in the 2 hours before sleep."
            )
        if isinstance(metric.steps, (int, float)) and metric.steps < BASELINES["steps"]:
            recommendation_items.append(
                f"Increase daily steps to at least {BASELINES['steps']:,} "
                f"(currently {int(metric.steps):,}). Add a 20-minute walk after your main meal."
            )
        if isinstance(metric.heart_rate, (int, float)) and metric.heart_rate >= 90:
            recommendation_items.append(
                "Lower resting heart rate through regular aerobic exercise such as brisk walking, "
                "cycling, or swimming 3–5 times per week."
            )

    if not recommendation_items:
        recommendation_items = [
            "Increase daily steps toward 8,000 with consistent activity timing.",
            "Maintain a stable sleep schedule targeting at least 7 hours per night.",
            "Use short breathing or mindfulness breaks during high-stress periods.",
        ]

    intro = (
        f"Personalised recommendations for {patient.name}:"
        if role == "doctor"
        else "Here are personalised steps to improve your health:"
    )
    points = "\n".join(f"  {i + 1}. {item}" for i, item in enumerate(recommendation_items[:5]))
    return f"{intro}\n{points}"


@csrf_exempt
def chatbot_query(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST is allowed."}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8")) if request.body else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        payload = {}

    question = (payload.get("question") or request.POST.get("question") or "").strip()
    patient_id_raw = payload.get("patient_id") or request.POST.get("patient_id")
    # Allow overriding role from the frontend (useful when session-based detection is ambiguous)
    override_role = (payload.get("user_role") or request.POST.get("user_role") or "").strip().lower()

    if not question:
        return JsonResponse({"error": "Question is required."}, status=400)

    try:
        patient_id = int(patient_id_raw)
    except (TypeError, ValueError):
        return JsonResponse({"error": "A valid patient_id is required."}, status=400)

    role, patient = _chatbot_permission_context(request, patient_id, override_role)
    if patient is None:
        return JsonResponse({"error": "Patient not found or access denied."}, status=403)

    # Honour client-supplied role only if it matches what the session already established
    if override_role in ("patient", "doctor") and override_role == role:
        role = override_role

    latest_metric, latest_prediction = _chatbot_patient_snapshot(patient)
    intent = _chatbot_detect_intent(question)

    dispatch = {
        "explain_shap":         lambda: _chatbot_term_explanation("explain_shap"),
        "explain_lifestyle_risk": lambda: _chatbot_term_explanation("explain_lifestyle_risk"),
        "explain_nervous":      lambda: _chatbot_term_explanation("explain_nervous"),
        "explain_stress_term":  lambda: _chatbot_term_explanation("explain_stress_term"),
        "deficiency":           lambda: _chatbot_deficiency_response(role, question, latest_metric),
        "risk":                 lambda: _chatbot_risk_response(role, patient, latest_prediction),
        "heart_rate":           lambda: _chatbot_heart_rate_response(role, patient, latest_metric),
        "sleep":                lambda: _chatbot_sleep_response(role, patient, latest_metric),
        "stress":               lambda: _chatbot_stress_response(role, patient, latest_metric),
        "steps":                lambda: _chatbot_steps_response(role, patient, latest_metric),
        "blood_oxygen":         lambda: _chatbot_blood_oxygen_response(role, patient, latest_metric),
        "calories":             lambda: _chatbot_calories_response(role, patient, latest_metric),
        "habit_insights":       lambda: _chatbot_habit_insights(role, patient, latest_metric, latest_prediction),
        "advice":               lambda: _chatbot_advice_response(role, patient, latest_metric, latest_prediction),
        "summary":              lambda: _chatbot_summary_response(role, patient, latest_metric, latest_prediction),
    }

    handler = dispatch.get(intent, dispatch["summary"])
    response_text = handler()

    ChatbotInteraction.objects.create(
        patient=patient,
        role=role if role in {ChatbotInteraction.ROLE_PATIENT, ChatbotInteraction.ROLE_DOCTOR} else ChatbotInteraction.ROLE_SYSTEM,
        intent=intent,
        question=question,
        response=response_text,
    )

    return JsonResponse(
        {
            "role": role,
            "intent": intent,
            "patient_id": getattr(patient, "pk", None),
            "response": response_text,
            "metrics": _chatbot_metric_summary(latest_metric, latest_prediction),
        }
    )
