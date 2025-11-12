import torch
import torch.nn.functional as F
from model.gnn_model import FraudDetectionGNN
from prepare_data import load_transaction_graph
from sklearn.model_selection import train_test_split
import os

def train_and_save_model(save_path="gnn_model.pt"):
    data = load_transaction_graph()

    # Train/test split on node indices
    nodes = list(range(data.num_nodes))
    train_idx, test_idx = train_test_split(nodes, test_size=0.2, random_state=42)

    model = FraudDetectionGNN(in_channels=data.num_features, hidden_channels=32, out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("Training started...")
    for epoch in range(80):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/80, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    train_and_save_model(save_path=os.path.join('models', 'gnn_model.pt'))
