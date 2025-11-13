import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATConv,
    GINEConv,
)
from torch_geometric.nn.norm import BatchNorm

CONV_MAP = {
    "gcn": GCNConv,
    "sage": SAGEConv,
    "gat": GATConv,
    "gine": GINEConv,
}


class MLP(nn.Module):
    """Simple multilayer perceptron with normalization and dropout."""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x):
        return self.net(x)


class FraudDetectionGNN(nn.Module):
    """
    Real-world GNN for dynamic fraud detection in financial transactions.
    
    This model is designed for NODE-LEVEL PREDICTION, where each node
    (transaction) gets its own fraud score.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=128,
        conv_type="sage",
        num_layers=3,
        dropout=0.5,
        use_residual=True,
        edge_attr_dim=None,
    ):
        super().__init__()
        assert conv_type in CONV_MAP, f"conv_type must be one of {list(CONV_MAP.keys())}"

        self.conv_type = conv_type
        self.num_layers = max(1, num_layers)
        self.dropout = dropout
        self.use_residual = use_residual
        self.edge_attr_dim = edge_attr_dim

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_c = in_channels if i == 0 else hidden_channels
            out_c = hidden_channels

            if conv_type == "gine":
                gine_in_c = in_c + (self.edge_attr_dim if self.edge_attr_dim is not None else 0)
                
                msg_mlp = nn.Sequential(
                    nn.Linear(gine_in_c, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, out_c),
                )
                conv = GINEConv(msg_mlp, edge_dim=self.edge_attr_dim) 
            elif conv_type == "gat":
                conv = GATConv(in_c, out_c, heads=1, concat=False)
            else:
                ConvClass = CONV_MAP[conv_type]
                conv = ConvClass(in_c, out_c)

            self.convs.append(conv)
            self.norms.append(BatchNorm(out_c))

        self.fraud_head = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(), 
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None, context=None):
        """
        Processes the graph and returns a fraud score FOR EACH NODE.

        Returns:
            torch.Tensor of shape [num_nodes, 1]
        """

        h = x
        for i, conv in enumerate(self.convs):
            residual = h
            if self.conv_type == "gine" and edge_attr is not None:
                h = conv(h, edge_index, edge_attr)
            else:
                h = conv(h, edge_index)

            h = self.norms[i](h)
            if i != (self.num_layers - 1):
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

            if self.use_residual and residual.shape == h.shape:
                h = h + residual

        fraud_scores = self.fraud_head(h)

        return fraud_scores

    @torch.no_grad()
    def predict_fraud_score(self, x, edge_index, edge_attr=None, batch=None, context=None):
        """Predict fraud probability (float between 0 and 1) for all nodes."""
        self.eval()
        scores = self.forward(x, edge_index, edge_attr, batch, context)
        return scores.squeeze().cpu().numpy()