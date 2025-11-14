import os
import requests
import json
import time
import re
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# --- Configuration ---
API_KEY = os.environ.get("GEMINI_KEY", "")
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# --- Core Prompt ---
SYSTEM_INSTRUCTION = (
    "You are an expert Financial Compliance Officer specializing in anti-money-mule protocols and "
    "Nigerian financial regulations. Your task is to generate a Suspicious Activity Report (SAR) "
    "that complies with Nigerian court standards and regulatory requirements. "
    "You must analyze transaction patterns, identify fraud indicators, and provide a legally "
    "defensible narrative suitable for regulatory filing and court proceedings. "
    "Focus on patterns such as structuring, smurfing, unusual activity patterns, and transaction anomalies."
)

def generate_compliance_narrative(gnn_score: float, regulatory_penalty: float,
                                  reputational_damage: float, transaction_data: dict) -> dict:

    # 1. Structured evidence
    structured_evidence = f"""
    --- EVIDENCE DATA ---
    Transaction Amount: ₦{transaction_data['amount']:,.2f}
    Sender ID: {transaction_data['sender_id']}
    Receiver ID: {transaction_data.get('receiver_id', 'N/A')}
    Device ID: {transaction_data['device_id']}
    Merchant ID: {transaction_data.get('merchant_id', 'N/A')}
    GNN Fraud Score: {gnn_score:.4f} (Risk Score: {int(gnn_score * 100)})
    Predicted Regulatory Penalty (NGN): ₦{regulatory_penalty:,.2f}
    Reputational Damage Score (0-10): {reputational_damage:.2f}
    """

    # 2. Structured Response
    user_query = (
        f"Based on the following evidence, generate a Suspicious Activity Report (SAR) in JSON format. "
        f"The report must comply with Nigerian financial regulations and court standards. "
        f"Analyze the transaction for fraud indicators such as structuring, smurfing, unusual patterns, "
        f"transaction amount anomalies, behavioral deviations, or other suspicious activities.\n\n"
        f"Return a JSON object with the following structure:\n"
        f'{{"reportType": "SAR", "indicators": ["indicator1", "indicator2"], "narrative": "detailed narrative text"}}\n\n'
        f"Report Type should be 'SAR' (Suspicious Activity Report) for transactions with GNN score > 0.5 or regulatory penalty > ₦50,000, "
        f"otherwise use 'MONITOR' for routine monitoring reports.\n"
        f"Indicators should be an array of specific fraud patterns detected (e.g., 'Structuring', 'Unusual Activity Pattern', "
        f"'Transaction Amount Anomaly', 'Behavioral Deviation', 'Smurfing', 'Rapid Succession Transactions', etc.).\n"
        f"Narrative should be a concise, legally defensible description of the suspicious activity, referencing the GNN score, "
        f"regulatory penalty, and specific indicators. Use professional language suitable for Nigerian regulatory filing.\n\n"
        f"{structured_evidence}\n\n"
        f"IMPORTANT: Return ONLY valid JSON, no additional text or markdown formatting."
    )

    # 3. PAYLOAD
    payload = {
        "systemInstruction": {
            "parts": [
                {"text": SYSTEM_INSTRUCTION}
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": user_query}
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 800,
            "temperature": 0.3
        }
    }

    # 4. Model Call
    for attempt in range(4):
        try:
            response = requests.post(
                f"{API_URL}?key={API_KEY}",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=20
            )

            if response.status_code == 200:
                result = response.json()
                raw_text = result["candidates"][0]["content"]["parts"][0]["text"]
                
                try:
                    cleaned_text = re.sub(r'```json\s*', '', raw_text)
                    cleaned_text = re.sub(r'```\s*', '', cleaned_text)
                    cleaned_text = cleaned_text.strip()
                    
                    ccn_data = json.loads(cleaned_text)
                    
                    return {
                        "reportType": ccn_data.get("reportType", "MONITOR"),
                        "indicators": ccn_data.get("indicators", []),
                        "narrative": ccn_data.get("narrative", raw_text)
                    }
                except json.JSONDecodeError:
                    indicators = []
                    narrative = raw_text
                    
                    indicator_keywords = {
                        "Structuring": ["structuring", "structured", "structure"],
                        "Smurfing": ["smurfing", "smurf"],
                        "Unusual Activity Pattern": ["unusual", "pattern", "anomaly"],
                        "Transaction Amount Anomaly": ["amount", "anomaly", "threshold"],
                        "Behavioral Deviation": ["behavioral", "deviation", "behavior"],
                        "Rapid Succession Transactions": ["rapid", "succession", "multiple"]
                    }
                    
                    text_lower = raw_text.lower()
                    for indicator, keywords in indicator_keywords.items():
                        if any(keyword in text_lower for keyword in keywords):
                            indicators.append(indicator)
                    
                    report_type = "SAR" if (gnn_score > 0.5 or regulatory_penalty > 50000) else "MONITOR"
                    
                    return {
                        "reportType": report_type,
                        "indicators": indicators if indicators else ["Transaction Review"],
                        "narrative": narrative
                    }

            elif response.status_code in [429, 500, 503]:
                wait = 2 ** attempt
                print(f"Temporary API error {response.status_code}. Retrying in {wait}s...")
                time.sleep(wait)

            else:
                return {
                    "reportType": "MONITOR",
                    "indicators": ["API Error"],
                    "narrative": f"ERROR: Gemini API returned {response.status_code}. Details: {response.text}"
                }

        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt
            print(f"Network error. Retrying in {wait}s: {e}")
            time.sleep(wait)

    return {
        "reportType": "MONITOR",
        "indicators": ["System Error"],
        "narrative": "FATAL ERROR: Gemini API failed after multiple attempts. Manual review required."
    }