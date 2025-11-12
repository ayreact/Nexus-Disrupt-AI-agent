from flask import Flask, request, jsonify
import torch
from model.gnn_model import FraudDetectionGNN
from prepare_data import load_transaction_graph

app = Flask(__name__)

# Load model and example data (for feature scaling reference)
data = load_transaction_graph()
model = FraudDetectionGNN(in_channels=data.num_features, hidden_channels=32, out_channels=2)
model.load_state_dict(torch.load('models/gnn_model.pt'))
model.eval()

@app.route('/api/check_fraud', methods=['POST'])
def check_fraud():
    payload = request.json
    amount = float(payload.get('amount', 0))
    device_id = payload.get('device_id', 'D0')

    # Prepare single-node inputs similar to predict.py
    device_val = abs(hash(device_id)) % 10
    amount_scaled = amount / (data.x[:,0].max().item() + 1e-9)

    x = torch.tensor([[amount_scaled, float(device_val)]], dtype=torch.float)
    edge_index = torch.tensor([[0],[0]], dtype=torch.long)

    with torch.no_grad():
        out = model(x, edge_index)
        prob = torch.exp(out)
        fraud_score = prob[0][1].item()
        freeze_now = "yes" if fraud_score > 0.6 else "no"

    return jsonify({
        "freeze_now": freeze_now,
        "gnn_score": round(fraud_score, 2),
        "amount": amount,
        "device_id": device_id
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)



# Invoke-WebRequest -Uri http://127.0.0.1:5000/api/check_fraud `-Method POST ` -Headers @{ "Content-Type" = "application/json" } ` -Body '{"amount":50000,"device_id":"D7"}'
