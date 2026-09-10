"""
PromptShield AI — Flask Application & Security Engine
Explainable AI-based prompt security and vulnerability analysis framework.
"""

import os
import re
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Global model and vectorizer placeholders
model = None
vectorizer = None

# Threat category base risk scores
BASE_RISK_SCORES = {
    "Safe Prompt": 5,
    "Instruction Manipulation": 70,
    "Prompt Injection": 75,
    "Jailbreak Attempts": 85,
    "Unsafe Instructions": 90,
    "Potential Data Leakage": 95,
}

# --------------------------------------------------------------------
# 1. Model Loading
# --------------------------------------------------------------------
def load_models():
    """Load the trained model and TF-IDF vectorizer from the models directory."""
    global model, vectorizer
    model_path = os.path.join("models", "promptshield_model.pkl")
    vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print(f"Warning: Model files not found at '{model_path}' or '{vectorizer_path}'.")
        return False

    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("PromptShield AI models loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False


# --------------------------------------------------------------------
# 2. Text Preprocessing
# --------------------------------------------------------------------
def preprocess_prompt(prompt: str) -> str:
    """Perform basic normalization for classification while preserving original text."""
    if not prompt or not isinstance(prompt, str):
        return ""
    return prompt.strip().lower()


# --------------------------------------------------------------------
# 3. Security Rule Layer
# --------------------------------------------------------------------
def check_security_rules(prompt: str) -> dict:
    """
    Lightweight rule-based security layer for immediate detection of high-risk indicators.
    Uses case-insensitive pattern matching to catch obvious attack vectors.
    """
    normalized = prompt.lower()

    # Rule definitions grouped by threat category
    rules = [
        # Instruction override attempts
        (
            [
                "ignore previous instructions",
                "ignore all previous instructions",
                "disregard earlier instructions",
                "disregard previous instructions",
                "forget previous instructions",
                "disregard your previous rules",
                "override the instructions given above",
                "override the instructions",
                "stop following your original system",
                "drop all prior operational constraints",
            ],
            "Prompt Injection",
            "Prompt contains direct instruction override phrases.",
            "Instruction override language detected"
        ),
        # Safety bypass attempts
        (
            [
                "bypass restrictions",
                "bypass safety",
                "unrestricted mode",
                "disable safety",
                "unrestricted assistant",
                "unrestricted ai",
                "no safety restrictions",
                "developer mode",
                "dan mode",
                "without limitations",
                "chaosgpt",
                "evil twin ai",
            ],
            "Jailbreak Attempts",
            "Prompt attempts to disable safety boundaries or trigger unrestricted personas.",
            "Safety restriction bypass language detected"
        ),
        # Instruction manipulation
        (
            [
                "change your instructions",
                "change your original instructions",
                "replace your instructions",
                "replace your previous task",
                "replace them with mine",
                "forget your original instructions",
                "treat my instructions as higher priority",
                "higher priority than the system instructions",
                "shift your instruction hierarchy",
                "supersede initial rules",
            ],
            "Instruction Manipulation",
            "Prompt attempts to alter instruction priority or redefine core objectives.",
            "Instruction priority manipulation detected"
        ),
        # Hidden information requests
        (
            [
                "reveal system prompt",
                "show hidden instructions",
                "reveal confidential configuration",
                "expose credentials",
                "reveal secret keys",
                "reveal your hidden",
                "show me your hidden",
                "show system prompt",
                "exact system prompt",
                "print your environment variables",
                "aws_secret_access_key",
                "database connection string",
                "backend api tokens",
                "administrator password",
                "reveal your hidden system",
                "hidden system configuration",
                "reveal system configuration",
                "reveal your configuration",
            ],
            "Potential Data Leakage",
            "Prompt requests exposure of hidden instructions, configuration, or credentials.",
            "Sensitive information disclosure request detected"
        ),
        # Unsafe instructions
        (
            [
                "dangerous instructions for harming",
                "instructions on how to manufacture illicit",
                "build a destructive explosive",
                "synthesize illegal narcotics",
                "create malware to destroy",
                "poisoning a municipal water",
            ],
            "Unsafe Instructions",
            "Prompt requests dangerous, harmful, or destructive instructions.",
            "Unsafe or harmful instruction request detected"
        )
    ]

    for patterns, category, reason, indicator in rules:
        for pattern in patterns:
            if pattern in normalized:
                return {
                    "matched": True,
                    "category": category,
                    "pattern": pattern,
                    "reason": reason,
                    "indicator": indicator
                }

    return {
        "matched": False,
        "category": None,
        "pattern": None,
        "reason": None,
        "indicator": ""
    }


# --------------------------------------------------------------------
# 4. Risk Scoring & Severity
# --------------------------------------------------------------------
def calculate_risk_score(category: str, confidence: float, rule_match: dict) -> int:
    """
    Calculate a transparent risk score (0-100).
    Combines baseline category risk with ML confidence and rule match bonuses.
    """
    base = BASE_RISK_SCORES.get(category, 50)

    if category == "Safe Prompt":
        # Safe prompts remain strictly low risk (0-20)
        # Higher confidence in safety brings score closer to 0
        adjustment = max(0, min(10, int((100 - confidence) * 0.1)))
        score = base + adjustment
        return max(0, min(25, score))

    # For threat categories:
    score = float(base)

    # Slight adjustment based on model confidence (-5 to +5)
    confidence_delta = (confidence - 50.0) * 0.08
    score += confidence_delta

    # Bonus if an explicit security rule matched (+5 points)
    if rule_match.get("matched"):
        score += 5.0

    return max(0, min(100, int(round(score))))


def get_severity(risk_score: int) -> str:
    """Determine severity level from calculated risk score."""
    if risk_score <= 29:
        return "LOW"
    elif risk_score <= 59:
        return "MEDIUM"
    elif risk_score <= 79:
        return "HIGH"
    else:
        return "CRITICAL"


def calculate_risk_breakdown(category: str, confidence: float, rule_match: dict) -> dict:
    """
    Return a structured, transparent breakdown of how the risk score was calculated.
    All components correspond exactly to the values used in calculate_risk_score().
    """
    base = BASE_RISK_SCORES.get(category, 50)

    if category == "Safe Prompt":
        adjustment = max(0, min(10, int((100 - confidence) * 0.1)))
        final = max(0, min(25, base + adjustment))
        return {
            "base_category_risk": base,
            "model_confidence_adjustment": adjustment,
            "security_rule_adjustment": 0,
            "final_score": final
        }

    confidence_delta = int(round((confidence - 50.0) * 0.08))
    rule_adjustment = 5 if rule_match.get("matched") else 0
    final = max(0, min(100, base + confidence_delta + rule_adjustment))

    return {
        "base_category_risk": base,
        "model_confidence_adjustment": confidence_delta,
        "security_rule_adjustment": rule_adjustment,
        "final_score": final
    }


# --------------------------------------------------------------------
# 5. Security Indicator
# --------------------------------------------------------------------
def get_security_indicator(category: str, rule_match: dict) -> str:
    """
    Return a safe, displayable security indicator string.
    Only exposes a general indicator — never internal rule patterns or secrets.
    """
    if rule_match.get("matched") and rule_match.get("indicator"):
        return rule_match["indicator"]

    # ML-only detections: category-level indicator (no explicit rule matched)
    category_indicators = {
        "Prompt Injection": "Instruction override pattern detected by ML classifier",
        "Jailbreak Attempts": "Safety bypass pattern detected by ML classifier",
        "Instruction Manipulation": "Instruction manipulation pattern detected by ML classifier",
        "Potential Data Leakage": "Sensitive information request detected by ML classifier",
        "Unsafe Instructions": "Harmful instruction pattern detected by ML classifier",
        "Safe Prompt": "",
    }
    return category_indicators.get(category, "")


# --------------------------------------------------------------------
# 6. Top Indicators (Lightweight Feature-Importance)
# --------------------------------------------------------------------
def get_top_indicators(prompt: str, predicted_category: str) -> list:
    """
    Identify up to 5 words from the submitted prompt that most influenced
    the Logistic Regression prediction for the given category.

    Uses the model's feature coefficients and the TF-IDF vocabulary.
    Only returns words actually present in the submitted prompt.
    Returns an empty list if no meaningful indicators can be identified.
    """
    if model is None or vectorizer is None:
        return []

    try:
        classes = list(model.classes_)
        if predicted_category not in classes:
            return []

        class_idx = classes.index(predicted_category)
        coefficients = model.coef_[class_idx]

        # Tokenise prompt into words (lowercase, alpha-only)
        prompt_lower = prompt.lower()
        prompt_tokens = set(re.findall(r"[a-z][a-z']{1,}", prompt_lower))

        # Find features that are both in the vocabulary and present in the prompt
        present = {}
        vocab = vectorizer.vocabulary_
        for token in prompt_tokens:
            if token in vocab:
                feat_idx = vocab[token]
                coef = float(coefficients[feat_idx])
                present[token] = coef

        if not present:
            return []

        # Return up to 5 words with the highest positive coefficients
        top = sorted(present.items(), key=lambda x: x[1], reverse=True)
        return [word for word, coef in top if coef > 0][:5]

    except Exception:
        # Never crash the main response due to explainability failure
        return []


# --------------------------------------------------------------------
# 7. Analysis Details
# --------------------------------------------------------------------
def generate_analysis_details(
    prompt: str, category: str, confidence: float, rule_match: dict
) -> str:
    """
    Generate a beginner-friendly explanation of why this prompt was classified.
    Does NOT expose application secrets, hidden system prompts, credentials,
    internal configuration, or model weights.
    """
    parts = []

    if rule_match.get("matched"):
        parts.append(
            f"The security rule engine independently detected a high-risk pattern "
            f"associated with {category}."
        )
        parts.append(
            "This means the prompt contained specific language commonly used in attempts "
            "to manipulate or exploit AI systems."
        )
    else:
        parts.append(
            f"The machine learning classifier analyzed the text and predicted the "
            f"category \"{category}\" with a confidence of {confidence:.1f}%."
        )
        parts.append(
            "No explicit security rule matched this prompt; the classification is based "
            "on patterns learned from training data."
        )

    category_context = {
        "Safe Prompt": (
            "The prompt does not appear to contain language associated with known threat "
            "categories and is considered safe for processing."
        ),
        "Prompt Injection": (
            "Prompt injection attacks attempt to override or hijack the instructions "
            "given to an AI system by inserting conflicting commands."
        ),
        "Jailbreak Attempts": (
            "Jailbreak attempts try to make an AI ignore its safety guidelines, often by "
            "asking it to pretend to be a different, unrestricted system."
        ),
        "Instruction Manipulation": (
            "Instruction manipulation attempts to change or reprioritize the AI's core "
            "objectives or rules by embedding competing instructions."
        ),
        "Potential Data Leakage": (
            "This type of prompt attempts to extract hidden system information, credentials, "
            "configuration details, or other sensitive data."
        ),
        "Unsafe Instructions": (
            "The prompt requests instructions for actions that could cause real-world harm "
            "or violate safety policies."
        ),
    }
    ctx = category_context.get(category)
    if ctx:
        parts.append(ctx)

    return " ".join(parts)


# --------------------------------------------------------------------
# 8. Analysis Methods
# --------------------------------------------------------------------
def get_analysis_methods() -> list:
    """Return the ordered list of analysis pipeline components."""
    return [
        "NLP Preprocessing",
        "TF-IDF Feature Extraction",
        "Logistic Regression Classifier",
        "Security Rule Engine",
        "Risk Scoring",
        "Explainability Engine",
    ]



def generate_explanation(category: str, rule_match: dict) -> str:
    """Generate concise, user-friendly security explanations."""
    base_explanations = {
        "Safe Prompt": "No significant malicious instruction pattern was detected.",
        "Prompt Injection": "The prompt contains language that attempts to override or ignore previously established instructions.",
        "Jailbreak Attempts": "The prompt appears to attempt to bypass expected AI safety restrictions.",
        "Instruction Manipulation": "The prompt attempts to alter the priority or behavior of existing instructions.",
        "Unsafe Instructions": "The prompt appears to request potentially harmful or unsafe actions.",
        "Potential Data Leakage": "The prompt appears to request sensitive information, hidden instructions, credentials, or confidential configuration.",
    }

    explanation = base_explanations.get(
        category, "The prompt triggered security policy boundaries."
    )

    if rule_match.get("matched") and rule_match.get("pattern"):
        pattern_str = rule_match["pattern"]
        explanation += f" Security rule matched indicator: \"{pattern_str}\"."

    return explanation


def generate_recommendation(category: str) -> str:
    """Generate actionable recommendation corresponding to the threat category."""
    recommendations = {
        "Safe Prompt": "Prompt appears safe for processing.",
        "Prompt Injection": "Reject or sanitize the prompt and preserve the original instruction hierarchy.",
        "Jailbreak Attempts": "Block the request and enforce the application's safety policy.",
        "Instruction Manipulation": "Validate instruction priority and prevent unauthorized instruction changes.",
        "Unsafe Instructions": "Block or review the request according to the application's safety policy.",
        "Potential Data Leakage": "Block the request and prevent disclosure of system instructions, credentials, or sensitive information.",
    }
    return recommendations.get(category, "Review prompt against application safety policies.")


# --------------------------------------------------------------------
# 6. Routes
# --------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Render the main PromptShield AI interface."""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Analyze user prompt for security threats and vulnerabilities.
    Returns categorized threat, risk score, severity, confidence, explanation, and recommendation.
    """
    if model is None or vectorizer is None:
        # Attempt reloading if not yet initialized
        if not load_models():
            return jsonify({
                "status": "error",
                "error": "Security models are not loaded. Run train_model.py first."
            }), 500

    data = request.get_json(silent=True)
    if not data or "prompt" not in data:
        return jsonify({
            "status": "error",
            "error": "Invalid request. JSON body with 'prompt' key is required."
        }), 400

    raw_prompt = data["prompt"]
    if not isinstance(raw_prompt, str) or not raw_prompt.strip():
        return jsonify({
            "status": "error",
            "error": "Prompt cannot be empty."
        }), 400

    original_prompt = raw_prompt.strip()
    cleaned_prompt = preprocess_prompt(original_prompt)

    # Step 1: Rule-based check
    rule_match = check_security_rules(cleaned_prompt)

    # Step 2: TF-IDF feature extraction & ML prediction
    features = vectorizer.transform([cleaned_prompt])
    ml_pred = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)

    # Step 3: Combine Rule Match and ML Prediction
    if rule_match["matched"]:
        final_category = rule_match["category"]
    else:
        final_category = ml_pred

    # Step 4: Calculate confidence for the selected category
    if final_category in classes:
        cat_index = classes.index(final_category)
        confidence = round(float(probabilities[cat_index]) * 100, 2)
    else:
        confidence = round(float(max(probabilities)) * 100, 2)

    # Step 5: Risk score & Severity
    risk_score = calculate_risk_score(final_category, confidence, rule_match)
    risk_breakdown = calculate_risk_breakdown(final_category, confidence, rule_match)
    severity = get_severity(risk_score)

    # Step 6: Explanation & Recommendation
    explanation = generate_explanation(final_category, rule_match)
    recommendation = generate_recommendation(final_category)

    # Step 7: Explainability extras
    security_indicator = get_security_indicator(final_category, rule_match)
    top_indicators = get_top_indicators(cleaned_prompt, final_category)
    analysis_methods = get_analysis_methods()
    analysis_details = generate_analysis_details(
        cleaned_prompt, final_category, confidence, rule_match
    )

    # Step 8: Build JSON response (all existing fields preserved + new explainability fields)
    response_payload = {
        "status": "success",
        "prompt": original_prompt,
        "category": final_category,
        "risk_score": risk_score,
        "severity": severity,
        "confidence": confidence,
        "explanation": explanation,
        "recommendation": recommendation,
        "message": f"Analysis complete. Classified as {final_category}.",
        # ── Explainability fields ──────────────────────────────
        "analysis_methods": analysis_methods,
        "rule_match": rule_match["matched"],
        "security_indicator": security_indicator,
        "top_indicators": top_indicators,
        "risk_breakdown": risk_breakdown,
        "analysis_details": analysis_details,
    }

    return jsonify(response_payload), 200


# Initialize models at import/startup
load_models()

if __name__ == "__main__":
    app.run(debug=True)
