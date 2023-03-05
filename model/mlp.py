import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, out_size: int) -> None:
        super().__init__()

        self.lin1 = nn.Linear(in_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, out_size)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.lin1(x)
        out = self.lin2(out)
        out = self.relu(out)

        return self.dropout(out)
