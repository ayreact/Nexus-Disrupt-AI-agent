import torch
from torch_geometric.data import Data
from model.gnn_model import FraudDetectionGNN
from prepare_data import load_transaction_graph
import pandas as pd
import datetime

def load_model(model_path="models/gnn_model.pt", conv_type="sage"):
    data = load_transaction_graph()
    model = FraudDetectionGNN(
        in_channels=data.num_features,
        hidden_channels=64,
        conv_type=conv_type,
        num_layers=3,
        dropout=0.5,
        use_residual=True,
        edge_attr_dim=data.edge_attr.size(1) if hasattr(data, 'edge_attr') else None,
    )

    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    print(f"✅ Model loaded successfully from {model_path}")
    return model, data


def prepare_new_transaction(amount, device_id, sender_id, receiver_id, merchant_id=None, timestamp=None):
    """Convert a single transaction into a tensor for inference."""
    if timestamp is None:
        timestamp = datetime.datetime.now()

    hour_norm = timestamp.hour / 24.0
    amount_norm = float(amount) / 1e6

    def hash_to_float(val):
        return float(abs(hash(str(val))) % 1000) / 1000.0

    sender_f = hash_to_float(sender_id)
    receiver_f = hash_to_float(receiver_id)
    device_f = hash_to_float(device_id)
    merchant_f = hash_to_float(merchant_id) if merchant_id else 0.0

    x_new = torch.tensor([[amount_norm, hour_norm, device_f, merchant_f, sender_f, receiver_f]], dtype=torch.float)
    return x_new


def connect_to_graph(x_new, data, k=5):
    new_idx = data.num_nodes
    src_nodes = []
    dst_nodes = []

    if data.x.size(1) >= 6:
        device_feat = 2
        sender_feat = 4 

        same_device = (torch.isclose(data.x[:, device_feat], x_new[0, device_feat], atol=1e-3)).nonzero().flatten()
        same_sender = (torch.isclose(data.x[:, sender_feat], x_new[0, sender_feat], atol=1e-3)).nonzero().flatten()

        neighbors = torch.unique(torch.cat([same_device, same_sender]))[:k]
        for n in neighbors:
            src_nodes.append(new_idx)
            dst_nodes.append(n.item())

    if len(src_nodes) == 0:
        src_nodes.append(new_idx)
        dst_nodes.append(0)

    new_edges = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    new_edge_index = torch.cat([data.edge_index, new_edges], dim=1)

    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        time_diff_np = torch.abs(data.x[dst_nodes, 1] - x_new[0, 1]).cpu().numpy().astype(np.float32)
        amount_diff_np = torch.abs(data.x[dst_nodes, 0] - x_new[0, 0]).cpu().numpy().astype(np.float32)
        
        new_edge_attr = torch.stack([
            torch.tensor(time_diff_np, dtype=torch.float),
            torch.tensor(amount_diff_np, dtype=torch.float)
        ], dim=1)
        edge_attr = torch.cat([data.edge_attr, new_edge_attr], dim=0)
    else:
        edge_attr = None

    return new_edge_index, edge_attr


def analyze_transaction(amount, device_id, sender_id, receiver_id, merchant_id=None, model=None, data=None):
    """Run inference on a new transaction and return fraud likelihood."""
    if model is None or data is None:
        model, data = load_model()

    x_new = prepare_new_transaction(amount, device_id, sender_id, receiver_id, merchant_id)
    edge_index, edge_attr = connect_to_graph(x_new, data)

    x_all = torch.cat([data.x, x_new], dim=0)

    with torch.no_grad():
        out = model(x_all, edge_index, edge_attr=edge_attr)
        
        fraud_score = out[-1].item() 
        
        freeze_now = "yes" if fraud_score > 0.6 else "no"

    return {
        "freeze_now": freeze_now,
        "gnn_score": round(fraud_score, 4),
        "details": {
            "amount": amount,
            "device_id": device_id,
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "merchant_id": merchant_id,
        },
    }


if __name__ == "__main__":
    import numpy as np 
    
    model, data = load_model()
    result = analyze_transaction(
        amount=95000,
        device_id="ATM_12B",
        sender_id="ACCT_001",
        receiver_id="ACCT_999",
        merchant_id="MRC_45",
        model=model,
        data=data
    )
    print(result)