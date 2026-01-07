import torch
import torch.nn as nn

class PilotNet(nn.Module):
    """
    NVIDIA PilotNet architecture for end-to-end self-driving.
    Input: 66x200x3 Image (YUV or RGB)
    Output: Steering, Throttle, Brake
    """
    def __init__(self, input_shape=(3, 66, 200), output_dim=3):
        super(PilotNet, self).__init__()
        self.input_shape = input_shape
        
        # Convolutional layers
        # Strided convolutions (5x5, stride 2)
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        
        # Non-strided convolutions (3x3)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        # Calculate flat size:
        # 66x200 -> (5x5 s2) -> 31x98 -> (5x5 s2) -> 14x47 -> (5x5 s2) -> 5x22
        # -> (3x3 s1) -> 3x20 -> (3x3 s1) -> 1x18.
        # Wait, let's verify calculation.
        # H_out = floor((H_in - K)/S + 1)
        # 66 -> (66-5)/2 + 1 = 30.5 -> 30 
        # 200 -> (200-5)/2 + 1 = 97.5 -> 97
        # 30x97 -> (30-5)/2 + 1 = 12.5 -> 12
        # 97 -> (97-5)/2 + 1 = 46. 
        # 12x46 -> (12-5)/2 + 1 = 3.5 -> 3
        # 46 -> (46-5)/2 + 1 = 20.
        # 3x20 -> (3-3)+1 = 1
        # 20 -> (20-3)+1 = 18.
        # So 64 * 1 * 18 = 1152.
        
        self.fc1 = nn.Linear(1152, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_dim)
        
        self.relu = nn.ReLU()
        # No activation on output for regression (or Sigmoid/Tanh if normalized)
        # Usually steering is [-1, 1], throttle/brake [0, 1].
        # We will output raw values and handle range in training or via activation.
        # For simplicity, raw linear output.
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        
        x = self.flatten(x)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x
        
if __name__ == "__main__":
    model = PilotNet()
    print(model)
    dummy_input = torch.randn(1, 3, 66, 200)
    output = model(dummy_input)
    print("Output shape:", output.shape)
