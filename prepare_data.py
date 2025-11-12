import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

def load_transaction_graph(csv_path="data/transactions.csv"):
    df = pd.read_csv(csv_path)

    # For this simple prototype we create one node per transaction.
    # Node features are [amount_scaled, device_encoded].
    # Edges connect consecutive transactions (simple chain) to give the GNN some structure.

    # Encode device IDs
    device_encoder = LabelEncoder()
    df['device_encoded'] = device_encoder.fit_transform(df['device_id'])

    # Extract amounts and devices
    amounts = df['amount'].values.astype(float)
    devices = df['device_encoded'].values.astype(float)

    # Scale amounts
    amounts = amounts / (amounts.max() + 1e-9)

    # Combine into a 2D tensor: [num_nodes, num_features]
    features = torch.tensor(list(zip(amounts, devices)), dtype=torch.float)

    # Build a simple chain edge_index: 0->1, 1->2, ... and back edges to make it undirected
    n = len(df)
    if n >= 2:
        src = list(range(0, n-1)) + list(range(1, n))
        dst = list(range(1, n)) + list(range(0, n-1))
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.tensor([[0],[0]], dtype=torch.long)

    labels = torch.tensor(df['is_fraud'].values, dtype=torch.long)

    data = Data(x=features, edge_index=edge_index, y=labels)
    return data

if __name__ == "__main__":
    data = load_transaction_graph()
    print(data)
