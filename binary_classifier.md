import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple feedforward neural network architecture as per the image
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(2, 3) # Input layer to hidden layer 1 (2 input features, 3 nodes)
        self.fc2 = nn.Linear(3, 2) # Hidden layer 1 to hidden layer 2 (3 nodes in the first hidden layer, 2 nodes in the second)
        self.fc3 = nn.Linear(2, 2) # Hidden layer 2 to output layer (2 nodes, 2 outputs)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x)) # Activation function between input layer and hidden layer 1
        x = torch.sigmoid(self.fc2(x)) # Activation function between hidden layer 1 and hidden layer 2
        x = self.fc3(x) # No activation function before the output layer
        return x

# Create the neural network
model = NeuralNetwork()

# Create dummy dataset
inputs = torch.rand(16, 2)  # 16 samples, 2 features (e.g., revenue growth and moving average ratio)
targets = torch.randint(0, 2, (16, 1)).float()  # 16 samples, binary target

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Display the dummy data
print("Input Data:")
print(inputs)
print("Targets:")
print(targets)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets.view(-1).long())  # Reshape targets to match output dimensions

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Dummy inference
with torch.no_grad():
    dummy_input = torch.tensor([[0.3, 0.7]])  # Dummy data point for inference
    predicted = model(dummy_input)
    predicted = torch.softmax(predicted, dim=1)  # Apply softmax to get probabilities
    print(f"Predicted probabilities: {predicted}")
    _, predicted_class = torch.max(predicted, 1)
    print(f"Predicted class: {predicted_class.item()}")

