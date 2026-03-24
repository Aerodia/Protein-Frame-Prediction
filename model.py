import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter


class EGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')

        self.edge_mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + 1, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.SiLU()
        )

        # Residual projection
        if in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x, pos, edge_index):
        row, col = edge_index

        diff = pos[row] - pos[col]
        dist = (diff ** 2).sum(dim=-1, keepdim=True)

        edge_feat = torch.cat([x[row], x[col], dist], dim=-1)

        messages = self.edge_mlp(edge_feat)

        agg = scatter(messages, col, dim=0, dim_size=x.size(0))

        x_res = self.residual_proj(x)
        x = x_res + self.node_mlp(agg)

        return x


class ProteinEGNN(nn.Module):
    def __init__(self, in_channels=45, hidden=64, out_channels=3):
        super().__init__()

        self.layer1 = EGNNLayer(in_channels, hidden)
        self.layer2 = EGNNLayer(hidden, hidden)

        self.out_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_channels)
        )

    def forward(self, data):
        x, pos, edge_index = data.x, data.pos, data.edge_index

        x = self.layer1(x, pos, edge_index)
        x = self.layer2(x, pos, edge_index)

        return self.out_mlp(x)