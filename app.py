from flask import Flask, request, jsonify
import torch
import joblib
import pickle
import os
import datetime
import numpy as np
import requests
from datetime import timezone

from model.gnn_model import FraudDetectionGNN
from prepare_data import (
    load_transaction_graph,
    ENCODERS_PATH,
    SCALER_PATH,
    FRAUD_FREQ_PATH
)
from ccn_system import generate_compliance_narrative


# -----------------------------------------------------------------------------
#  FLASK APP INITIALIZATION
# -----------------------------------------------------------------------------
app = Flask(__name__)

CDT_MODEL_PATH = os.path.join("models", "cdt_model.pkl")


# -----------------------------------------------------------------------------
#  LOAD PREPROCESSING OBJECTS
# -----------------------------------------------------------------------------
try:
    with open(ENCODERS_PATH, 'rb') as f:
        LABEL_ENCODERS = pickle.load(f)

    with open(SCALER_PATH, 'rb') as f:
        SCALER = pickle.load(f)

    with open(FRAUD_FREQ_PATH, 'rb') as f:
        FRAUD_FREQ_MAP = pickle.load(f)

    print("✅ Preprocessing encoders/scaler/frequency map loaded.")

except FileNotFoundError:
    print("FATAL ERROR: Preprocessing files not found. Run train_model.py first.")
    LABEL_ENCODERS = {}
    SCALER = None
    FRAUD_FREQ_MAP = {'sender_id': {}, 'device_id': {}}


# -----------------------------------------------------------------------------
#  LOAD CDT CONSEQUENCE MODEL
# -----------------------------------------------------------------------------
try:
    CDT_MODEL = joblib.load(CDT_MODEL_PATH)
    print("✅ CDT Consequence Model loaded.")
except FileNotFoundError:
    print(f"FATAL ERROR: CDT model not found at {CDT_MODEL_PATH}. Run train_cdt.py first.")
    CDT_MODEL = None


# -----------------------------------------------------------------------------
#  LOAD GNN MODEL + BASE GRAPH
# -----------------------------------------------------------------------------
data, _, _, _ = load_transaction_graph()

model = FraudDetectionGNN(
    in_channels=data.num_features,
    hidden_channels=128,
    conv_type="sage",
    num_layers=3,
    dropout=0.5,
    use_residual=True,
    edge_attr_dim=data.edge_attr.size(1) if hasattr(data, "edge_attr") else None,
)

model.load_state_dict(torch.load('models/gnn_model.pt', map_location=torch.device('cpu')))
model.eval()

print("✅ GNN Model loaded and ready for API requests.")


# -----------------------------------------------------------------------------
#  HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def get_fraud_frequency(entity_id, entity_type):
    """Return precomputed fraud frequency or fallback global rate."""
    global_rate = 0.08
    return FRAUD_FREQ_MAP.get(entity_type, {}).get(entity_id, global_rate)


def encode_transaction(amount, device_id, sender_id, receiver_id, merchant_id=None, timestamp=None):
    """Convert raw transaction into normalized + encoded feature vector for inference."""
    if timestamp is None:
        timestamp = datetime.datetime.now()

    hour_norm = timestamp.hour / 24.0

    if SCALER:
        amount_norm = SCALER.transform(np.array([[amount]]))[0][0]
    else:
        amount_norm = float(amount) / 1e6

    sender_fraud_freq = get_fraud_frequency(sender_id, 'sender_id')
    device_fraud_freq = get_fraud_frequency(device_id, 'device_id')

    feature_values = {
        'amount_norm': amount_norm,
        'hour_norm': hour_norm,
        'device_id': device_id,
        'merchant_id': merchant_id,
        'sender_id': sender_id,
        'receiver_id': receiver_id,
        'sender_id_fraud_freq': sender_fraud_freq,
        'device_id_fraud_freq': device_fraud_freq,
    }

    feature_order = [
        'amount_norm', 'hour_norm', 'device_id', 'merchant_id',
        'sender_id', 'receiver_id', 'sender_id_fraud_freq', 'device_id_fraud_freq'
    ]

    final_features = []

    for col in feature_order:
        value = feature_values[col]

        # Encode categorical IDs
        if col in ['device_id', 'merchant_id', 'sender_id', 'receiver_id']:
            encoder = LABEL_ENCODERS.get(col)
            if encoder:
                try:
                    encoded_val = encoder.transform([str(value)])[0]
                except ValueError:
                    encoded_val = len(encoder.classes_)
                final_features.append(float(encoded_val))
            else:
                final_features.append(0.0)
        else:
            final_features.append(value)

    return torch.tensor([final_features], dtype=torch.float)


def connect_transaction(x_new, data, k=5):
    """Connect new node to graph via similarity edges."""
    new_idx = data.num_nodes
    src_nodes, dst_nodes = [], []

    device_feat, sender_feat = 2, 4

    same_device = (torch.isclose(data.x[:, device_feat], x_new[0, device_feat], atol=1e-3)).nonzero().flatten()
    same_sender = (torch.isclose(data.x[:, sender_feat], x_new[0, sender_feat], atol=1e-3)).nonzero().flatten()

    neighbors = torch.unique(torch.cat([same_device, same_sender]))[:k]

    for n in neighbors:
        src_nodes.append(new_idx)
        dst_nodes.append(n.item())

    if not src_nodes:
        src_nodes.append(new_idx)
        dst_nodes.append(0)

    new_edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    edge_index = torch.cat([data.edge_index, new_edge_index], dim=1)

    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        time_diff_np = torch.abs(data.x[dst_nodes, 1] - x_new[0, 1]).cpu().numpy().astype(np.float32)
        amount_diff_np = torch.abs(data.x[dst_nodes, 0] - x_new[0, 0]).cpu().numpy().astype(np.float32)

        new_edge_attr = torch.stack([
            torch.tensor(time_diff_np, dtype=torch.float),
            torch.tensor(amount_diff_np, dtype=torch.float)
        ], dim=1)

        edge_attr = torch.cat([data.edge_attr, new_edge_attr], dim=0)
    else:
        edge_attr = None

    return edge_index, edge_attr


# -----------------------------------------------------------------------------
#  HELPER FUNCTIONS FOR RISK ASSESSMENT
# -----------------------------------------------------------------------------
def calculate_risk_level(gnn_score: float, regulatory_penalty: float) -> str:
    """Calculate risk level based on GNN score and regulatory penalty."""
    if gnn_score >= 0.7 or regulatory_penalty >= 200000:
        return "HIGH"
    elif gnn_score >= 0.4 or regulatory_penalty >= 50000:
        return "MEDIUM"
    else:
        return "LOW"


def calculate_impact_level(regulatory_penalty: float, reputational_damage: float) -> str:
    """Calculate impact level for CDT section."""
    if regulatory_penalty >= 200000 or reputational_damage >= 7:
        return "HIGH"
    elif regulatory_penalty >= 50000 or reputational_damage >= 4:
        return "MEDIUM"
    else:
        return "LOW"


# -----------------------------------------------------------------------------
#  FULL PREDICTION PIPELINE (GNN + CDT + CCN)
# -----------------------------------------------------------------------------
def predict_fraud(amount, device_id, sender_id, receiver_id, merchant_id=None, transaction_timestamp=None):
    """Run inference across GNN, CDT, and compliance narrative."""
    # Generate timestamps
    if transaction_timestamp is None:
        transaction_timestamp = datetime.datetime.now(timezone.utc)
    elif isinstance(transaction_timestamp, str):
        transaction_timestamp = datetime.datetime.fromisoformat(transaction_timestamp.replace('Z', '+00:00'))
    
    analysis_timestamp = datetime.datetime.now(timezone.utc)
    
    x_new = encode_transaction(amount, device_id, sender_id, receiver_id, merchant_id, transaction_timestamp)
    edge_index, edge_attr = connect_transaction(x_new, data)

    x_all = torch.cat([data.x, x_new], dim=0)

    # 1. GNN Classification
    with torch.no_grad():
        out = model(x_all, edge_index, edge_attr=edge_attr)
        fraud_score = out[-1].item()

    # 2. CDT Consequence Prediction
    regulatory_penalty = 0.0
    if CDT_MODEL:
        cdt_features = np.array([[fraud_score, amount]])
        predicted_penalty = CDT_MODEL.predict(cdt_features)[0]
        regulatory_penalty = np.clip(predicted_penalty, 0, 5_000_000)

    # 3. Reputational Score
    reputational_damage = np.clip(regulatory_penalty / 100000, 0, 10)
    
    # 4. Calculate risk and impact levels
    risk_level = calculate_risk_level(fraud_score, regulatory_penalty)
    confidence = int(round(fraud_score * 100))
    
    # 5. Cognitive Compliance Narrative (LLM) - now returns structured dict
    ccn_data = generate_compliance_narrative(
        gnn_score=fraud_score,
        regulatory_penalty=regulatory_penalty,
        reputational_damage=reputational_damage,
        transaction_data={
            "amount": amount,
            "device_id": device_id,
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "merchant_id": merchant_id
        }
    )

    # 6. Calculate CDT metrics
    impact_level = calculate_impact_level(regulatory_penalty, reputational_damage)
    estimated_loss = round(regulatory_penalty * 0.25, 2)  # 25% of regulatory penalty
    
    # 7. Build CDT section - always include all fields except regulatoryPenalty
    # regulatoryPenalty only included for fraudulent/high-risk transactions
    is_fraudulent = risk_level == "HIGH" or fraud_score >= 0.6
    cdt_section = {
        "reputationalDamageScore": round(reputational_damage, 2),
        "impactLevel": impact_level,
        "estimatedLoss": estimated_loss
    }
    
    # Only include regulatoryPenalty for fraudulent transactions
    if is_fraudulent:
        cdt_section["regulatoryPenalty"] = round(regulatory_penalty, 2)
    
    # 8. Build structured response
    response = {
        "timestamp": analysis_timestamp.isoformat().replace('+00:00', 'Z'),
        "transaction": {
            "amount": float(amount),
            "sender": sender_id,
            "receiver": receiver_id,
            "device_id": device_id,
            "timestamp": transaction_timestamp.isoformat().replace('+00:00', 'Z')
        },
        "analysis": {
            "gnnScore": round(fraud_score, 4),
            "riskLevel": risk_level,
            "confidence": confidence
        },
        "cdt": cdt_section,
        "ccn": {
            "reportType": ccn_data.get("reportType", "MONITOR"),
            "riskScore": confidence,
            "indicators": ccn_data.get("indicators", []),
            "narrative": ccn_data.get("narrative", ""),
            "dateGenerated": analysis_timestamp.isoformat().replace('+00:00', 'Z')
        }
    }

    return response


# -----------------------------------------------------------------------------
#  FLASK API ROUTE
# -----------------------------------------------------------------------------
@app.route('/api/check_fraud', methods=['POST'])
def check_fraud():
    payload = request.json

    result = predict_fraud(
        amount=float(payload.get("amount", 0)),
        device_id=payload.get("device_id", "D0"),
        sender_id=payload.get("sender_id", "S0"),
        receiver_id=payload.get("receiver_id", "R0"),
        merchant_id=payload.get("merchant_id"),
        transaction_timestamp=payload.get("timestamp")  # Optional: ISO format timestamp
    )

    return jsonify(result)


# -----------------------------------------------------------------------------
#  RUN SERVER
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
