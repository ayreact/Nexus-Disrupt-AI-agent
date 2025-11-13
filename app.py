from flask import Flask, request, jsonify
import torch
import joblib 
import pickle
import os
from model.gnn_model import FraudDetectionGNN
from prepare_data import load_transaction_graph, ENCODERS_PATH, SCALER_PATH, FRAUD_FREQ_PATH 
import datetime
import numpy as np 

app = Flask(__name__)

CDT_MODEL_PATH = os.path.join("models", "cdt_model.pkl") 

# --- LOAD PREPROCESSING OBJECTS ---
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
    FRAUD_FREQ_MAP = {'sender_id': {}, 'device_id': {}} # Fallback map

# --- LOAD CONSEQUECE (CDT) MODEL ---
try:
    CDT_MODEL = joblib.load(CDT_MODEL_PATH)
    print("✅ CDT Consequence Model loaded.")
except FileNotFoundError:
    print(f"FATAL ERROR: CDT model not found at {CDT_MODEL_PATH}. Run train_cdt.py first.")
    CDT_MODEL = None


# --- 1. Model and Data Initialization (FIXED UNPACKING) ---
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


def get_fraud_frequency(entity_id, entity_type):
    """Retrieves the pre-calculated fraud frequency for a given ID."""
    global_rate = 0.08 
    
    if entity_type in FRAUD_FREQ_MAP:
        return FRAUD_FREQ_MAP[entity_type].get(entity_id, global_rate) 
    return global_rate

def encode_transaction(amount, device_id, sender_id, receiver_id, merchant_id=None, timestamp=None):
    """
    Converts a transaction into 8 features, including the new Fraud Frequency features.
    """
    if timestamp is None:
        timestamp = datetime.datetime.now()
    
    # 1. Standard Features
    hour_norm = timestamp.hour / 24.0
    if SCALER:
        amount_norm = SCALER.transform(np.array([[amount]]))[0][0]
    else:
        amount_norm = float(amount) / 1e6

    # 2. Augmentation Features (CRITICAL FOR GNN PERFORMANCE)
    sender_fraud_freq = get_fraud_frequency(sender_id, 'sender_id')
    device_fraud_freq = get_fraud_frequency(device_id, 'device_id')
    
    feature_values = {
        'amount_norm': amount_norm,
        'hour_norm': hour_norm,
        # Categorical IDs
        'device_id': device_id, 
        'merchant_id': merchant_id,
        'sender_id': sender_id,
        'receiver_id': receiver_id,
        # Augmentation Values
        'sender_id_fraud_freq': sender_fraud_freq, 
        'device_id_fraud_freq': device_fraud_freq, 
    }
    
    final_features = []
    
    feature_order = [
        'amount_norm', 'hour_norm', 'device_id', 'merchant_id', 
        'sender_id', 'receiver_id', 'sender_id_fraud_freq', 'device_id_fraud_freq'
    ]
    
    for col in feature_order:
        value = feature_values[col]
        
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

    x_new = torch.tensor([final_features], dtype=torch.float)
    return x_new

def connect_transaction(x_new, data, k=5):
    """Connect new node to existing graph based on sender/device similarity. (No functional change here)."""
    new_idx = data.num_nodes
    src_nodes, dst_nodes = [], []

    device_feat, sender_feat = 2, 4

    same_device = (torch.isclose(data.x[:, device_feat], x_new[0, device_feat], atol=1e-3)).nonzero().flatten()
    same_sender = (torch.isclose(data.x[:, sender_feat], x_new[0, sender_feat], atol=1e-3)).nonzero().flatten()

    neighbors = torch.unique(torch.cat([same_device, same_sender]))[:k]
    for n in neighbors:
        src_nodes.append(new_idx)
        dst_nodes.append(n.item())

    if len(src_nodes) == 0:
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

def predict_fraud(amount, device_id, sender_id, receiver_id, merchant_id=None):
    """
    Run GNN inference and then use the CDT model to predict consequences.
    """
    x_new = encode_transaction(amount, device_id, sender_id, receiver_id, merchant_id)
    edge_index, edge_attr = connect_transaction(x_new, data)

    x_all = torch.cat([data.x, x_new], dim=0)
    
    # 1. GNN Prediction (Classification)
    with torch.no_grad():
        out = model(x_all, edge_index, edge_attr=edge_attr)
        fraud_score = out[-1].item()
    
    # --- 2. CDT Prediction (Consequence Modeling) ---
    regulatory_penalty = 0.0
    
    if CDT_MODEL:
        cdt_features = np.array([[fraud_score, amount]])
        
        predicted_penalty = CDT_MODEL.predict(cdt_features)[0]
        
        regulatory_penalty = np.clip(predicted_penalty, a_min=0, a_max=5000000)
    
    # --- 3. Decision & Reputational Score (Simple Rules) ---
    
    reputational_damage_score = np.clip(regulatory_penalty / 100000, a_min=0, a_max=10)
    freeze_now = "yes" if (fraud_score > 0.6 or regulatory_penalty > 100000) else "no"

    return {
        "freeze_now": freeze_now,
        "gnn_score": round(fraud_score, 4),
        "regulatory_penalty_ngn": round(regulatory_penalty, 2),
        "reputational_damage_score": round(reputational_damage_score, 2)
    }

# --- 3. Flask API ---
@app.route('/api/check_fraud', methods=['POST'])
def check_fraud():
    payload = request.json
    amount = float(payload.get("amount", 0))
    device_id = payload.get("device_id", "D0")
    sender_id = payload.get("sender_id", "S0")
    receiver_id = payload.get("receiver_id", "R0")
    merchant_id = payload.get("merchant_id", None)

    result = predict_fraud(amount, device_id, sender_id, receiver_id, merchant_id)
    result.update({
        "amount": amount,
        "device_id": device_id,
        "sender_id": sender_id,
        "receiver_id": receiver_id,
        "merchant_id": merchant_id
    })

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)