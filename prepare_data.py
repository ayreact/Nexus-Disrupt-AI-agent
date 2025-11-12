import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

def load_transaction_graph(csv_path="data/transactions.csv"):
    df = pd.read_csv(csv_path)

    device_encoder = LabelEncoder()
    df['device_encoded'] = device_encoder.fit_transform(df['device_id'])

    amounts = df['amount'].values.astype(float)
    devices = df['device_encoded'].values.astype(float)

    amounts = amounts / (amounts.max() + 1e-9)

    features = torch.tensor(list(zip(amounts, devices)), dtype=torch.float)
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
