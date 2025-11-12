import torch
from model.gnn_model import FraudDetectionGNN
from prepare_data import load_transaction_graph

def load_model(path='models/gnn_model.pt'):
    data = load_transaction_graph()
    model = FraudDetectionGNN(in_channels=data.num_features, hidden_channels=32, out_channels=2)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model, data

def analyze_transaction(amount, device_id, model=None, data=None):
    # Simple single-node input: create a feature vector consistent with training features
    # device encoding: hash to small integer
    device_val = abs(hash(device_id)) % 10
    amount_scaled = amount / (data.x[:,0].max().item() + 1e-9)

    x = torch.tensor([[amount_scaled, float(device_val)]], dtype=torch.float)
    edge_index = torch.tensor([[0],[0]], dtype=torch.long)  # self-loop for single node

    with torch.no_grad():
        out = model(x, edge_index)
        prob = torch.exp(out)
        fraud_score = prob[0][1].item()
        freeze_now = "yes" if fraud_score > 0.6 else "no"

        return {"freeze_now": freeze_now, "gnn_score": round(fraud_score, 2)}

if __name__ == '__main__':
    model, data = load_model()
    print(analyze_transaction(50000, 'D7', model=model, data=data))
