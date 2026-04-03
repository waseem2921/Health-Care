import re
import random
import warnings
from typing import Dict, List, TypedDict, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
}

TREND_SOURCES = [
    "https://news.google.com/rss/search?q=india+health+disease+outbreak&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=india+dengue+chikungunya+malaria+heatwave+flu&hl=en-IN&gl=IN&ceid=IN:en",
    "https://www.who.int/rss-feeds/news-english.xml",
]

DISEASE_KEYWORD_MAP = {
    "Dengue": ["dengue", "aedes", "platelet", "mosquito-borne"],
    "Tuberculosis": ["tuberculosis", "tb", "drug-resistant tb", "pulmonary tb"],
    "Nipah": ["nipah", "bat-borne", "encephalitis cluster"],
    "Viral Flu": ["flu", "influenza", "h1n1", "seasonal flu"],
    "Heatstroke": ["heatstroke", "heat wave", "heatwave", "extreme heat", "heat stress"],
    "Malaria": ["malaria", "plasmodium", "anopheles"],
    "Chikungunya": ["chikungunya", "chikv"],
    "Typhoid": ["typhoid", "enteric fever", "salmonella typhi"],
    "Cholera": ["cholera", "water-borne outbreak"],
    "Measles": ["measles", "rubella"],
    "COVID-19": ["covid", "coronavirus", "sars-cov-2"],
}

DISEASE_ALIAS_MAP = {
    "dengue": "Dengue",
    "tuberculosis": "Tuberculosis",
    "tb": "Tuberculosis",
    "nipah": "Nipah",
    "viral flu": "Viral Flu",
    "flu": "Viral Flu",
    "influenza": "Viral Flu",
    "heatstroke": "Heatstroke",
    "heat wave": "Heatstroke",
    "heatwave": "Heatstroke",
    "malaria": "Malaria",
    "chikungunya": "Chikungunya",
    "typhoid": "Typhoid",
    "cholera": "Cholera",
    "measles": "Measles",
    "covid": "COVID-19",
    "coronavirus": "COVID-19",
}

FALLBACK_GENERAL_VIRAL = "General Viral Alert"
FALLBACK_RESPIRATORY = "Respiratory Risk"
FALLBACK_SEASONAL = "Seasonal Health Risk"

DISEASE_PREVENTIVE_TIPS = {
    "Dengue": [
        "Drain stagnant water around homes and use mosquito repellents, especially at dawn and dusk.",
        "Wear full-sleeve clothing and install window screens to reduce mosquito exposure.",
        "Seek immediate medical care for persistent fever, body pain, or unusual bleeding signs.",
    ],
    "Tuberculosis": [
        "Cover coughs, improve room ventilation, and avoid prolonged exposure in crowded indoor spaces.",
        "If cough lasts more than two weeks, get sputum testing and start treatment early if advised.",
        "Complete full TB treatment without interruption to prevent drug resistance.",
    ],
    "Nipah": [
        "Avoid consuming raw date palm sap and wash fruits thoroughly before eating.",
        "Use gloves and masks when caring for symptomatic patients and report clusters quickly.",
        "Avoid contact with bats and sick animals in outbreak-prone districts.",
    ],
    "Viral Flu": [
        "Wash hands frequently and use a mask in crowded places during active flu spread.",
        "Isolate early when symptomatic and maintain hydration with adequate rest.",
        "Protect high-risk family members by limiting close contact during fever or cough.",
    ],
    "Heatstroke": [
        "Drink oral fluids regularly and avoid outdoor activity during peak afternoon heat.",
        "Use light cotton clothing and keep vulnerable people in cool, shaded spaces.",
        "Watch for dizziness, confusion, or fainting and seek urgent care if these appear.",
    ],
    "Malaria": [
        "Sleep under insecticide-treated mosquito nets and use repellents at night.",
        "Clear standing water weekly and test early for fever with chills.",
        "Complete antimalarial treatment as prescribed to prevent complications.",
    ],
    "Chikungunya": [
        "Prevent mosquito bites with repellents and covered clothing throughout the day.",
        "Remove water-filled containers around the home to reduce breeding sites.",
        "Consult a clinician for severe joint pain or prolonged fever.",
    ],
    "Typhoid": [
        "Drink boiled or filtered water and avoid uncovered street food during outbreaks.",
        "Practice strict hand hygiene before meals and after using restrooms.",
        "Get evaluated early for prolonged fever and abdominal discomfort.",
    ],
    "Cholera": [
        "Use safe drinking water and oral rehydration promptly for watery diarrhea.",
        "Maintain food and hand hygiene, especially in flood-affected areas.",
        "Seek urgent treatment for dehydration signs such as reduced urination or lethargy.",
    ],
    "Measles": [
        "Ensure age-appropriate vaccination and isolate symptomatic children early.",
        "Monitor fever and rash progression, and seek care for breathing difficulty.",
        "Improve nutrition and hydration during recovery to reduce complications.",
    ],
    "COVID-19": [
        "Improve indoor ventilation and wear masks in crowded enclosed settings.",
        "Test early if symptomatic and isolate to protect older adults and comorbid patients.",
        "Keep vaccinations and booster guidance up to date for high-risk groups.",
    ],
    "General Viral Alert": [
        "Limit close contact while symptomatic and follow respiratory hygiene consistently.",
        "Hydrate adequately and seek medical evaluation for persistent fever beyond 48 hours.",
        "Use masks in crowded places during viral surges and monitor elderly family members.",
    ],
    "Respiratory Risk": [
        "Reduce exposure to poor air quality, wear masks outdoors, and improve indoor ventilation.",
        "Seek prompt care for breathlessness, persistent cough, or chest discomfort.",
        "Avoid smoke exposure and monitor oxygen levels in high-risk individuals.",
    ],
    "Seasonal Health Risk": [
        "Follow seasonal health advisories and maintain hydration, sleep, and nutrition.",
        "Practice hand hygiene and avoid sharing utensils during active local illness waves.",
        "Monitor fever clusters in family members and seek timely medical review.",
    ],
}

RISK_HIGH_KEYWORDS = ["outbreak", "outbreaks", "spreading", "spread"]
RISK_MODERATE_KEYWORDS = ["alert", "alerts", "rising", "increase"]

SOURCE_RELIABILITY_WEIGHTS = {
    "who.int": 18,
    "cdc.gov": 17,
    "nih.gov": 16,
    "gov.in": 15,
    "nic.in": 14,
    "news.google.com": 10,
    "reuters.com": 11,
    "apnews.com": 10,
    "bbc.com": 10,
}


class HeadlineAnalysis(TypedDict):
    disease_name: str
    risk_level: str
    description: str
    preventive_advice: str
    confidence_score: float


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title or "").strip().lower()


def fetch_health_headlines(limit: int = 10) -> List[Dict[str, str]]:
    """Fetch top health-related headlines from trusted public sources."""
    collected: List[Dict[str, str]] = []
    seen_titles = set()

    for source_url in TREND_SOURCES:
        if len(collected) >= limit:
            break
        try:
            response = requests.get(source_url, headers=HEADERS, timeout=10)
            response.raise_for_status()
        except requests.RequestException:
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        items = soup.find_all("item")

        if not items:
            soup = BeautifulSoup(response.text, "html.parser")
            items = soup.select("article h3 a")
            for anchor in items:
                title = (anchor.get_text() or "").strip()
                href_value = anchor.get("href")
                link = href_value.strip() if isinstance(href_value, str) else ""
                if not title:
                    continue
                normalized = _normalize_title(title)
                if normalized in seen_titles:
                    continue
                seen_titles.add(normalized)
                collected.append({"title": title, "source_url": link or source_url})
                if len(collected) >= limit:
                    break
            continue

        for item in items:
            title_tag = item.find("title")
            link_tag = item.find("link")
            title = (title_tag.get_text() if title_tag else "").strip()
            link = (link_tag.get_text() if link_tag else "").strip()
            if not title:
                continue
            normalized = _normalize_title(title)
            if normalized in seen_titles:
                continue
            seen_titles.add(normalized)
            collected.append({"title": title, "source_url": link or source_url})
            if len(collected) >= limit:
                break

    return collected[:limit]


def analyze_headline(title: str, source_url: str = "") -> HeadlineAnalysis:
    """Map a headline to disease category, risk level, summary, and preventive recommendations."""
    normalized = (title or "").strip().lower()
    disease_name, match_count, keyword_score = _infer_india_disease_name(normalized)

    if not _is_label_supported_by_headline(normalized, disease_name):
        disease_name = _fallback_disease_label(normalized)
        match_count = 0
        keyword_score = 1

    risk_level = _classify_risk(normalized)
    source_weight = _source_reliability_score(source_url)
    confidence_score = _compute_confidence(match_count, keyword_score, risk_level, source_weight)
    preventive_advice = _pick_preventive_tip(disease_name)
    description = _build_contextual_description(title, disease_name, risk_level)

    return {
        "disease_name": disease_name,
        "risk_level": risk_level,
        "description": description,
        "preventive_advice": preventive_advice,
        "confidence_score": confidence_score,
    }


def _infer_india_disease_name(normalized_title: str) -> Tuple[str, int, int]:
    best_disease = ""
    best_match_count = 0
    best_keyword_score = -1

    for disease, keywords in DISEASE_KEYWORD_MAP.items():
        matched_keywords = [keyword for keyword in keywords if _keyword_in_text(keyword, normalized_title)]
        if not matched_keywords:
            continue

        keyword_score = sum(len(keyword.replace(" ", "")) for keyword in matched_keywords)
        match_count = len(matched_keywords)
        if keyword_score > best_keyword_score or (
            keyword_score == best_keyword_score and match_count > best_match_count
        ):
            best_disease = disease
            best_match_count = match_count
            best_keyword_score = keyword_score

    if best_disease:
        return best_disease, best_match_count, best_keyword_score

    return _fallback_disease_label(normalized_title), 0, 1


def _pick_preventive_tip(disease_name: str) -> str:
    tips = DISEASE_PREVENTIVE_TIPS.get(disease_name) or DISEASE_PREVENTIVE_TIPS[FALLBACK_GENERAL_VIRAL]
    return random.choice(tips)


def _classify_risk(normalized_title: str) -> str:
    if any(_keyword_in_text(keyword, normalized_title) for keyword in RISK_HIGH_KEYWORDS):
        return "High"
    if any(_keyword_in_text(keyword, normalized_title) for keyword in RISK_MODERATE_KEYWORDS):
        return "Moderate"
    return "Low"


def _build_contextual_description(title: str, disease_name: str, risk_level: str) -> str:
    cleaned_title = re.sub(r"\s+", " ", title or "").strip().rstrip(".")
    if not cleaned_title:
        return f"{disease_name} activity is being monitored under a {risk_level.lower()} public health risk signal."

    if risk_level == "High":
        return f"{cleaned_title}. This headline suggests active {disease_name.lower()} transmission pressure and needs rapid community-level preventive action."
    if risk_level == "Moderate":
        return f"{cleaned_title}. Signals point to a rising {disease_name.lower()} concern, requiring targeted awareness and early symptom screening."
    return f"{cleaned_title}. Current reporting indicates a low-level {disease_name.lower()} signal that should continue to be tracked with routine precautions."


def _is_label_supported_by_headline(normalized_title: str, disease_name: str) -> bool:
    disease_keywords = DISEASE_KEYWORD_MAP.get(disease_name, [])
    if any(_keyword_in_text(keyword, normalized_title) for keyword in disease_keywords):
        return True

    # Allow lightweight alias matching for common abbreviations in headlines.
    aliases = [alias for alias, mapped in DISEASE_ALIAS_MAP.items() if mapped == disease_name]
    return any(_keyword_in_text(alias, normalized_title) for alias in aliases)


def _fallback_disease_label(normalized_title: str) -> str:
    if any(_keyword_in_text(token, normalized_title) for token in ["viral", "fever", "infection"]):
        return FALLBACK_GENERAL_VIRAL
    if any(
        _keyword_in_text(token, normalized_title)
        for token in ["respiratory", "lung", "breath", "cough", "air quality", "pneumonia", "legionnaire"]
    ):
        return FALLBACK_RESPIRATORY
    return FALLBACK_SEASONAL


def _compute_confidence(match_count: int, keyword_score: int, risk_level: str, source_weight: int) -> float:
    # Confidence combines keyword evidence, urgency signal, and source reliability.
    if match_count > 0:
        evidence_points = 50 + (match_count * 8) + min(20, keyword_score)
    else:
        evidence_points = 28 + min(10, keyword_score)
    risk_bonus = {"High": 20, "Moderate": 12, "Low": 6}.get(risk_level, 6)
    score = evidence_points + risk_bonus + int(source_weight)
    return round(max(35.0, min(99.0, float(score))), 1)


def _source_reliability_score(source_url: str) -> int:
    domain = (urlparse(source_url or "").netloc or "").lower()
    if not domain:
        return 6
    for known_domain, weight in SOURCE_RELIABILITY_WEIGHTS.items():
        if domain == known_domain or domain.endswith(f".{known_domain}"):
            return weight
    if any(domain.endswith(suffix) for suffix in [".gov", ".gov.in", ".edu", ".int"]):
        return 12
    return 7


def _keyword_in_text(keyword: str, text: str) -> bool:
    keyword = str(keyword or "").strip().lower()
    if not keyword:
        return False
    escaped = re.escape(keyword).replace(r"\ ", r"\s+")
    pattern = rf"\b{escaped}\b"
    return re.search(pattern, text or "") is not None
