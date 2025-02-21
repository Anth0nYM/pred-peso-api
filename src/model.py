import torch
import torch.nn as nn


class DummyModel(torch.nn.Module):

    def __init__(self, output_dim: int = 1):
        super(DummyModel, self).__init__()

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(500 * 500, output_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        x = torch.relu(x)
        return x
