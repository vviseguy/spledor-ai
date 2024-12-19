import torch
import torch.nn as nn

class SplendorAI(nn.Module):
    def __init__(self, output_size=5000):
        super(SplendorAI, self).__init__()
        
        # 3x4x7 input (cards)
        self.conv1_cards = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(1, 7))  # Process features
        self.conv2_cards = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 1))  # Process tiers
        self.fc_cards = nn.Linear(32, 128)

        # 5x6 input (nobles)
        self.fc_nobles = nn.Linear(5 * 6, 128)

        # 6 input (tokens)
        self.fc_tokens = nn.Linear(6, 64)

        # 3x7 input (other data)
        self.conv_other = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=7)
        self.fc_other = nn.Linear(16, 64)

        # Final layers
        self.fc_combined = nn.Linear(128 + 128 + 64 + 64, 1024)
        self.fc_output = nn.Linear(1024, output_size)
        
    def forward(self, x1, x2, x3, x4):
        # Process 3x4x7 cards
        x1 = self.conv1_cards(x1)  # Conv over features
        x1 = self.conv2_cards(x1)  # Conv over tiers
        x1 = x1.view(x1.size(0), -1)  # Flatten
        x1 = torch.relu(self.fc_cards(x1))

        # Process 5x6 nobles
        x2 = x2.view(x2.size(0), -1)  # Flatten
        x2 = torch.relu(self.fc_nobles(x2))

        # Process 6 tokens
        x3 = torch.relu(self.fc_tokens(x3))

        # Process 3x7 other data
        x4 = self.conv_other(x4)  # Conv over features
        x4 = x4.view(x4.size(0), -1)  # Flatten
        x4 = torch.relu(self.fc_other(x4))

        # Combine and output
        combined = torch.cat([x1, x2, x3, x4], dim=1)
        combined = torch.relu(self.fc_combined(combined))
        output = self.fc_output(combined)

        return output

# Example usage:
# Instantiate the model
model = SplendorAI(output_size=5000)

# Example input tensors
x1 = torch.rand((32, 3, 4, 7))  # Batch size 32
x2 = torch.rand((32, 5, 6))
x3 = torch.rand((32, 6))
x4 = torch.rand((32, 3, 7))

# Forward pass
output = model(x1, x2, x3, x4)  # Output: (32, 5000)
