from flask import Flask, request, jsonify
import torch
from model.gnn_model import FraudDetectionGNN
from prepare_data import load_transaction_graph, ENCODERS_PATH, SCALER_PATH 
import datetime
import numpy as np 
import pickle
import os

app = Flask(__name__)

try:
    with open(ENCODERS_PATH, 'rb') as f:
        LABEL_ENCODERS = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        SCALER = pickle.load(f)
    print("✅ Preprocessing encoders/scaler loaded.")
except FileNotFoundError:
    print("FATAL ERROR: Preprocessing files not found. Run train_model.py first.")
    LABEL_ENCODERS = {}
    SCALER = None

data, _, _ = load_transaction_graph() 
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


def encode_transaction(amount, device_id, sender_id, receiver_id, merchant_id=None, timestamp=None):
    """
    Convert a single transaction into the correct feature tensor using 
    saved LabelEncoders and StandardScaler.
    """
    if timestamp is None:
        timestamp = datetime.datetime.now()
    
    # 1. TIME FEATURE
    hour_norm = timestamp.hour / 24.0
    
    # 2. AMOUNT FEATURE 
    if SCALER:
        amount_norm = SCALER.transform(np.array([[amount]]))[0][0]
    else:
        amount_norm = float(amount) / 1e6 

    # 3. CATEGORICAL FEATURES
    feature_values = {
        'amount_norm': amount_norm,
        'hour_norm': hour_norm,
        'device_id': device_id, 
        'merchant_id': merchant_id,
        'sender_id': sender_id,
        'receiver_id': receiver_id,
    }
    
    final_features = []
    
    for col in ['amount_norm', 'hour_norm', 'device_id', 'merchant_id', 'sender_id', 'receiver_id']:
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
    """Connect new node to existing graph based on sender/device similarity."""
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
    """Run GNN inference on a single transaction."""
    x_new = encode_transaction(amount, device_id, sender_id, receiver_id, merchant_id)
    edge_index, edge_attr = connect_transaction(x_new, data)

    x_all = torch.cat([data.x, x_new], dim=0)
    with torch.no_grad():
        out = model(x_all, edge_index, edge_attr=edge_attr)
        
        fraud_score = out[-1].item()
        
        freeze_now = "yes" if fraud_score > 0.6 else "no"

    return {"freeze_now": freeze_now, "gnn_score": round(fraud_score, 4)}

# --- 3. API Entry Point ---
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