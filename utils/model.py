import torch
from torch_geometric.nn import GCNConv

class GNNOptimizer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class DynamicRouteOptimizer:
    def __init__(self):
        self.model = GNNOptimizer(in_channels=3, hidden_channels=16, out_channels=1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.MSELoss()

    def train(self, data, epochs=50):
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            loss = self.criterion(out.view(-1), data.y)
            loss.backward()
            self.optimizer.step()
        print(f'Training complete. Final loss: {loss.item()}')

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            return self.model(data.x, data.edge_index)
