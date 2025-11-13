import os
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from model.gnn_model import FraudDetectionGNN
from prepare_data import load_transaction_graph


def train_and_save_model(save_path="models/gnn_model.pt"):
    # -------------------------------
    # 1. Load dataset
    # -------------------------------
    data = load_transaction_graph() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------
    # 2. Split data (Stratified Split for better metrics)
    # -------------------------------
    nodes = list(range(data.num_nodes))
    train_idx, test_idx = train_test_split(
        nodes,
        test_size=0.2,
        random_state=42,
        stratify=data.y.cpu().numpy() 
    )

    # -------------------------------
    # 3. Model initialization
    # -------------------------------
    in_channels = data.num_features
    edge_attr_dim = getattr(data, "edge_attr", None)
    edge_attr_dim = edge_attr_dim.shape[1] if edge_attr_dim is not None else None

    model = FraudDetectionGNN(
        in_channels=in_channels,
        hidden_channels=128,
        conv_type="sage",
        num_layers=3,
        dropout=0.4,
        edge_attr_dim=edge_attr_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCELoss()

    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")
    print("Training started...\n")

    # -------------------------------
    # 4. Training Loop
    # -------------------------------
    best_val_loss = float("inf")

    for epoch in range(1, 81):
        model.train()
        optimizer.zero_grad()

        out = model(
            data.x.to(device),
            data.edge_index.to(device),
            edge_attr=(data.edge_attr.to(device) if hasattr(data, "edge_attr") else None),
        ).squeeze()

        y_true = data.y.to(device).float()
        y_pred_train = out[train_idx]
        y_true_train = y_true[train_idx]

        loss = criterion(y_pred_train, y_true_train)
        loss.backward()
        optimizer.step()

        # -------------------------------
        # 5. Validation
        # -------------------------------
        model.eval()
        with torch.no_grad():
            y_pred_test = out[test_idx]
            y_true_test = y_true[test_idx]

            val_loss = criterion(y_pred_test, y_true_test).item()

            y_pred_bin = (y_pred_test.cpu().numpy() > 0.5).astype(int)
            y_true_np = y_true_test.cpu().numpy()

            acc = accuracy_score(y_true_np, y_pred_bin)
            prec = precision_score(y_true_np, y_pred_bin, zero_division=0)
            rec = recall_score(y_true_np, y_pred_bin, zero_division=0)
            try:
                auc = roc_auc_score(y_true_np, y_pred_test.cpu().numpy())
            except ValueError as e:
                auc = 0.0

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {loss.item():.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | AUC: {auc:.3f}"
        )

        # -------------------------------
        # 6. Save best model
        # -------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model updated and saved to {save_path}\n")

    print("\nTraining complete.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_and_save_model(save_path=os.path.join("models", "gnn_model.pt"))