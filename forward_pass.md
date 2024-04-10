# Let's redefine the neural network with print statements to log the forward pass

class NeuralNetworkWithLogs(nn.Module):
    def __init__(self):
        super(NeuralNetworkWithLogs, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(2, 3) # Input layer to hidden layer 1
        self.fc2 = nn.Linear(3, 2) # Hidden layer 1 to hidden layer 2
        self.fc3 = nn.Linear(2, 2) # Hidden layer 2 to output layer

    def forward(self, x):
        print("Input:", x)
        x = self.fc1(x)
        print("After first linear layer:", x)
        x = torch.sigmoid(x)
        print("After first sigmoid activation:", x)
        x = self.fc2(x)
        print("After second linear layer:", x)
        x = torch.sigmoid(x)
        print("After second sigmoid activation:", x)
        x = self.fc3(x)
        print("After third linear layer (logits):", x)
        return x

# Instantiate the network with logs
model_with_logs = NeuralNetworkWithLogs()

# Dummy input for demonstration
dummy_input = torch.tensor([[0.3, 0.7]])

# Forward pass with logs
print("Forward Pass Logs:")
model_with_logs(dummy_input)
