from torch import nn
from torch import Tensor
import torch
import torch.nn.functional as F

class Model(nn.Module):
    """Just a dummy model to show how to structure your code"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=128 * 7 * 7, out_features=256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1) # flatten the tensor before the fully connected layers, alternatively use torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = Model()
    print("model architecture:", model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    x = torch.randn(1, 1, 28, 28)  # dummy input (batch=1, channel=1, 28x28)
    with torch.no_grad():
        out = model(x)
        
    print(f"Output shape of model: {model(x).shape}")


    # dummy_input = torch.randn(1, 1, 28, 28)
    # output = model(dummy_input)
    # print("output shape:", output.shape)

