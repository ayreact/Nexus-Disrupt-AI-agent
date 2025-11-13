import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np 
import pickle
import os

MODEL_DIR = "models"
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

def load_transaction_graph(csv_path="data/transactions.csv"):
    """
    Build a transaction graph for fraud detection AND save the preprocessing objects.
    """
    df = pd.read_csv(csv_path)

    # --- 1. Encode categorical data ---
    label_encoders = {}
    for col in ['sender_id', 'receiver_id', 'device_id', 'merchant_id', 'location']:
        if col in df.columns:
            le = LabelEncoder()
            # Fit and transform
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            df[col] = 0

    # --- 2. Normalize numerical columns ---
    scaler = StandardScaler()
    if 'amount' in df.columns:
        # Fit and transform
        df['amount_norm'] = scaler.fit_transform(df[['amount']])
    else:
        df['amount_norm'] = 0.0
    
    # Time normalization
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['hour_norm'] = df['hour'] / 24.0
    else:
        df['hour_norm'] = 0.0

    # --- SAVE ENCODERS AND SCALER (NEW LOGIC) ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(ENCODERS_PATH, 'wb') as f:
        pickle.dump(label_encoders, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Encoders/Scalers saved to {MODEL_DIR}/")

    # --- 3. Create node features (DTYPE FIX RETAINED) ---
    feature_cols = ['amount_norm', 'hour_norm', 'device_id', 'merchant_id', 'sender_id', 'receiver_id']
    features_np = df[feature_cols].values.astype(np.float32)
    features = torch.tensor(features_np, dtype=torch.float)

    # --- 4. Create labels ---
    labels = torch.tensor(df['is_fraud'].astype(int).values, dtype=torch.long)

    # --- 5. Create edges ---
    src_nodes = []
    dst_nodes = []

    def add_edges(group_col):
        groups = df.groupby(group_col).groups
        for _, node_indices in groups.items():
            node_indices = list(node_indices)
            for i in range(len(node_indices)):
                for j in range(i + 1, len(node_indices)):
                    src_nodes.append(node_indices[i])
                    dst_nodes.append(node_indices[j])

    for col in ['sender_id', 'receiver_id', 'device_id', 'merchant_id']:
        add_edges(col)

    if len(src_nodes) == 0:
        src_nodes = list(range(len(df) - 1))
        dst_nodes = list(range(1, len(df)))

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

    # --- 6. Build edge features (DTYPE FIX RETAINED) ---
    time_diff_np = np.abs(df['hour_norm'].values[src_nodes] - df['hour_norm'].values[dst_nodes]).astype(np.float32)
    amount_diff_np = np.abs(df['amount_norm'].values[src_nodes] - df['amount_norm'].values[dst_nodes]).astype(np.float32)

    time_feat = torch.tensor(time_diff_np, dtype=torch.float)
    amount_feat = torch.tensor(amount_diff_np, dtype=torch.float)

    edge_attr = torch.stack([time_feat, amount_feat], dim=1)

    # --- 7. Build torch_geometric Data object ---
    data = Data(
        x=features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=labels
    )

    print(f"Graph built: {data.num_nodes} nodes, {data.num_edges} edges")
    print(f"Feature dim: {data.num_features}, Edge dim: {data.edge_attr.size(1)}")

    return data, label_encoders, scaler


if __name__ == "__main__":
    data, _, _ = load_transaction_graph()
    print(data)