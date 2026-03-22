from datetime import date, timedelta
import json

from django.test import TestCase

from core.models import ChatbotInteraction, Doctor, HealthMetrics, Patient, Prediction
from core.views import (
	_analyze_lifestyle_disorder,
	_derive_data_source,
	_find_balanced_doctor,
	_nervous_system_recovery_insight,
	_stress_sleep_interaction_insight,
)


class LifestyleInsightTests(TestCase):
	def setUp(self):
		self.primary_doctor = Doctor.objects.create(
			name="Dr. Primary",
			email="primary@example.com",
			specialization="General Physician",
			experience_years=8,
			hospital="City Hospital",
			password="pass123",
		)
		self.sleep_doctor = Doctor.objects.create(
			name="Dr. Sleep",
			email="sleep@example.com",
			specialization="Sleep Specialist",
			experience_years=10,
			hospital="City Hospital",
			password="pass123",
		)
		self.secondary_sleep_doctor = Doctor.objects.create(
			name="Dr. Lighter Load",
			email="sleep2@example.com",
			specialization="Sleep Specialist",
			experience_years=6,
			hospital="City Hospital",
			password="pass123",
		)
		self.patient = Patient.objects.create(
			name="Alex Patient",
			age=42,
			gender="Male",
			doctor=self.primary_doctor,
			assigned_doctor=self.primary_doctor,
			email="alex@example.com",
			mobile="9999999999",
			username="alex",
			password="secret",
		)

		for offset, steps, stress, sleep_quality, sleep_hours, heart_rate in [
			(3, 9200, 3, 7, 7.4, 72),
			(2, 2500, 8, 4, 5.2, 92),
			(1, 8600, 4, 6, 6.5, 76),
		]:
			HealthMetrics.objects.create(
				patient=self.patient,
				date=date.today() - timedelta(days=offset),
				steps=steps,
				stress_level=stress,
				sleep_quality=sleep_quality,
				sleep_hours=sleep_hours,
				heart_rate=heart_rate,
				calories_burned=1900,
				data_source="Wearable",
			)

	def test_lifestyle_disorder_analysis_flags_expected_factors(self):
		result = _analyze_lifestyle_disorder(
			{
				"stress_level": 9,
				"steps_per_day": 1800,
				"sleep_duration_hours": 4.8,
				"sleep_quality_score": 3,
				"resting_heart_rate": 101,
				"activity_level": "Low",
			},
			recent_metrics=list(HealthMetrics.objects.filter(patient=self.patient).order_by("-date")[:7]),
			model_risk_score=78,
		)

		self.assertGreaterEqual(result["risk_score"], 70)
		self.assertEqual(result["risk_level"], "High")
		self.assertIn("high stress", result["factors"])
		self.assertIn("low physical activity", result["factors"])
		self.assertIn("poor sleep quality", result["factors"])
		self.assertIn("elevated heart rate", result["factors"])

	def test_recovery_insight_drops_for_high_stress_and_poor_sleep(self):
		recovery = _nervous_system_recovery_insight(
			{
				"resting_heart_rate": 96,
				"stress_level": 8,
				"sleep_duration_hours": 5.0,
				"sleep_quality_score": 4,
			},
			recent_metrics=list(HealthMetrics.objects.filter(patient=self.patient).order_by("-date")[:7]),
		)

		self.assertLess(recovery["score"], 60)
		self.assertIn("lifestyle imbalance", recovery["explanation"].lower())

	def test_stress_sleep_interaction_detects_sequence(self):
		insight = _stress_sleep_interaction_insight(
			{
				"stress_level": 7,
				"sleep_duration_hours": 5.5,
				"sleep_quality_score": 4,
			},
			recent_metrics=list(HealthMetrics.objects.filter(patient=self.patient).order_by("date")),
		)

		self.assertIn("poor sleep", insight["patient_text"].lower())
		self.assertIn("stress-sleep", insight["doctor_text"].lower())

	def test_data_source_derivation_distinguishes_mixed_inputs(self):
		source = _derive_data_source(
			{"steps_per_day": 7000, "resting_heart_rate": 78},
			{
				"steps_per_day": 7000,
				"resting_heart_rate": 78,
				"stress_level": 6,
				"sleep_duration_hours": 6.2,
			},
		)
		self.assertEqual(source, "Wearable + manual")

	def test_balanced_doctor_prefers_lowest_patient_load(self):
		overloaded_patient = Patient.objects.create(
			name="Busy Queue",
			age=38,
			gender="Female",
			doctor=self.sleep_doctor,
			assigned_doctor=self.sleep_doctor,
			email="busy@example.com",
			mobile="8888888888",
			username="busy",
			password="secret",
		)
		self.assertIsNotNone(overloaded_patient)

		doctor = _find_balanced_doctor(["Sleep Specialist"])
		self.assertEqual(doctor, self.secondary_sleep_doctor)


class ChatbotEndpointTests(TestCase):
	def setUp(self):
		self.doctor = Doctor.objects.create(
			name="Dr. Insight",
			email="insight@example.com",
			specialization="General Physician",
			experience_years=7,
			hospital="Insight Hospital",
			password="pass123",
		)
		self.patient = Patient.objects.create(
			name="Jamie",
			age=36,
			gender="Female",
			doctor=self.doctor,
			assigned_doctor=self.doctor,
			email="jamie@example.com",
			mobile="7777777777",
			username="jamie",
			password="secret",
		)
		HealthMetrics.objects.create(
			patient=self.patient,
			date=date.today(),
			steps=3200,
			heart_rate=94,
			stress_level=8,
			sleep_hours=5.4,
			sleep_quality=4,
			calories_burned=1700,
			blood_oxygen=95,
			activity_level="Low",
		)
		Prediction.objects.create(
			patient=self.patient,
			risk_score=82.5,
			priority="High",
			explanation_text="Risk is driven by elevated stress and low activity.",
			predicted_label="HIGH",
			readable_ai_explanation="Low activity and high stress contributed most to the elevated risk score.",
			risk_factors="high stress|low physical activity|poor sleep quality",
			lifestyle_recommendations="Increase steps daily|Stabilize sleep schedule|Reduce stress load",
		)

	def _post_query(self, question, patient_id):
		return self.client.post(
			"/chatbot/query/",
			data=json.dumps({"question": question, "patient_id": patient_id}),
			content_type="application/json",
		)

	def _post_query_with_role(self, question, patient_id, user_role):
		return self.client.post(
			"/chatbot/query/",
			data=json.dumps({"question": question, "patient_id": patient_id, "user_role": user_role}),
			content_type="application/json",
		)

	def test_patient_can_query_own_risk(self):
		session = self.client.session
		session["username"] = self.patient.username
		session.save()

		response = self._post_query("Why is my risk high?", self.patient.id)
		self.assertEqual(response.status_code, 200)
		payload = response.json()
		self.assertEqual(payload["intent"], "risk")
		self.assertIn("risk score", payload["response"].lower())

	def test_doctor_gets_patient_priority_explanation(self):
		session = self.client.session
		session["doctor_id"] = self.doctor.id
		session.save()

		response = self._post_query("Why was this patient marked high priority?", self.patient.id)
		self.assertEqual(response.status_code, 200)
		payload = response.json()
		self.assertEqual(payload["role"], "doctor")
		# New response includes "<name> has a lifestyle disorder risk score of X.X (High priority)"
		self.assertIn("risk score", payload["response"].lower())
		self.assertIn("high", payload["response"].lower())

	def test_patient_cannot_query_other_patient(self):
		other_patient = Patient.objects.create(
			name="Other",
			age=30,
			gender="Male",
			doctor=self.doctor,
			assigned_doctor=self.doctor,
			email="other@example.com",
			mobile="6666666666",
			username="other",
			password="secret",
		)

		session = self.client.session
		session["username"] = self.patient.username
		session.save()

		response = self._post_query("Is heart rate normal?", other_patient.id)
		self.assertEqual(response.status_code, 403)

	def test_explain_shap_intent(self):
		session = self.client.session
		session["username"] = self.patient.username
		session.save()

		response = self._post_query("What is SHAP?", self.patient.id)
		self.assertEqual(response.status_code, 200)
		payload = response.json()
		self.assertEqual(payload["intent"], "explain_shap")
		self.assertIn("shap", payload["response"].lower())
		self.assertIn("shapley", payload["response"].lower())

	def test_stress_intent_returns_dynamic_data(self):
		session = self.client.session
		session["username"] = self.patient.username
		session.save()

		response = self._post_query("Why is my stress high?", self.patient.id)
		self.assertEqual(response.status_code, 200)
		payload = response.json()
		self.assertEqual(payload["intent"], "stress")
		# Should include the actual stress value (8.0 set in setUp)
		self.assertIn("8.0", payload["response"])

	def test_deficiency_intent_mentions_sleep(self):
		session = self.client.session
		session["username"] = self.patient.username
		session.save()

		response = self._post_query("I feel tired and have no energy", self.patient.id)
		self.assertEqual(response.status_code, 200)
		payload = response.json()
		self.assertEqual(payload["intent"], "deficiency")
		# Patient has sleep_hours=5.4 so sleep deprivation should be mentioned
		self.assertIn("sleep", payload["response"].lower())

	def test_steps_intent_returns_dynamic_count(self):
		session = self.client.session
		session["username"] = self.patient.username
		session.save()

		response = self._post_query("How many steps have I taken?", self.patient.id)
		self.assertEqual(response.status_code, 200)
		payload = response.json()
		self.assertEqual(payload["intent"], "steps")
		# Patient has 3200 steps — should appear in response
		self.assertIn("3,200", payload["response"])

	def test_summary_intent_covers_all_metrics(self):
		session = self.client.session
		session["username"] = self.patient.username
		session.save()

		response = self._post_query("How am I doing overall?", self.patient.id)
		self.assertEqual(response.status_code, 200)
		payload = response.json()
		self.assertEqual(payload["intent"], "summary")
		body = payload["response"].lower()
		self.assertIn("heart rate", body)
		self.assertIn("sleep", body)
		self.assertIn("stress", body)

	def test_explain_lifestyle_risk_term(self):
		session = self.client.session
		session["username"] = self.patient.username
		session.save()

		response = self._post_query("What is lifestyle disorder risk?", self.patient.id)
		self.assertEqual(response.status_code, 200)
		payload = response.json()
		self.assertEqual(payload["intent"], "explain_lifestyle_risk")
		self.assertIn("lifestyle disorder risk score", payload["response"].lower())

	def test_patient_query_succeeds_even_with_stale_doctor_session_key(self):
		session = self.client.session
		session["username"] = self.patient.username
		session["patient_id"] = self.patient.pk
		session["doctor_id"] = 9999  # stale key from a previous role
		session.save()

		response = self._post_query_with_role("Why is my stress high?", self.patient.id, "patient")
		self.assertEqual(response.status_code, 200)
		payload = response.json()
		self.assertEqual(payload["role"], "patient")
		self.assertEqual(payload["intent"], "stress")

	def test_chatbot_query_is_logged_for_usage_analytics(self):
		session = self.client.session
		session["username"] = self.patient.username
		session.save()

		response = self._post_query("What is my risk today?", self.patient.id)
		self.assertEqual(response.status_code, 200)

		interaction = ChatbotInteraction.objects.order_by("-created_at").first()
		self.assertIsNotNone(interaction)
		self.assertEqual(interaction.patient, self.patient)
		self.assertEqual(interaction.role, "patient")
		self.assertIn("risk", interaction.intent)


class ManagementAnalyticsDashboardTests(TestCase):
	def setUp(self):
		self.doctor = Doctor.objects.create(
			name="Dr. Admin Metrics",
			email="adminmetrics@example.com",
			specialization="General Physician",
			experience_years=9,
			hospital="Metro General",
			password="pass123",
		)
		self.patient = Patient.objects.create(
			name="Morgan",
			age=40,
			gender="Female",
			doctor=self.doctor,
			assigned_doctor=self.doctor,
			email="morgan@example.com",
			mobile="9991112222",
			username="morgan",
			password="secret",
		)
		HealthMetrics.objects.create(
			patient=self.patient,
			date=date.today(),
			steps=2800,
			heart_rate=96,
			stress_level=8,
			sleep_hours=5.3,
			sleep_quality=4,
			calories_burned=1700,
			activity_level="Low",
			data_source="Manual entry",
		)
		Prediction.objects.create(
			patient=self.patient,
			risk_score=84.2,
			priority="High",
			explanation_text="Elevated stress and low sleep quality.",
			predicted_label="HIGH",
			data_source="Manual entry",
			risk_factors="high stress|poor sleep quality",
		)
		ChatbotInteraction.objects.create(
			patient=self.patient,
			role="patient",
			intent="summary",
			question="How am I doing?",
			response="Summary response",
		)

	def test_management_dashboard_renders_with_expected_metrics(self):
		response = self.client.get("/management-analytics")
		self.assertEqual(response.status_code, 200)
		content = response.content.decode("utf-8")
		self.assertIn("Hospital-Level Lifestyle Disorder Monitoring", content)
		self.assertIn("Chatbot Interactions", content)
		self.assertIn("84.2", content)

	def test_management_dashboard_includes_chart_payload(self):
		response = self.client.get("/management-analytics")
		self.assertEqual(response.status_code, 200)
		content = response.content.decode("utf-8")
		self.assertIn("const chartData =", content)
		self.assertIn("risk_labels", content)
		self.assertIn("hospital_labels", content)
		self.assertIn("xai_features", content)
