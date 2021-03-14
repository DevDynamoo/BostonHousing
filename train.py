import numpy as np
import torch
import torch.nn as nn
from model import NeuralNetwork
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# Loading data
csv = np.genfromtxt('Data/housing.csv', delimiter="", dtype='float32')

# Preparing data
inputs = csv[:, 1:13]
targets = csv[:, 13:]

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

# Dataset and DataLoader
train_ds = TensorDataset(inputs, targets)
train_dl = DataLoader(dataset=train_ds, batch_size=4, shuffle=True)

# Model
model = NeuralNetwork(inputs.shape[1], 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

# Training
for epoch in range(1000):

    # Train with batches of data
    for xb, yb in train_dl:
        # Generating predictions
        pred = model(xb)

        # Calculating loss
        loss = criterion(pred, yb)

        # Computing gradients
        loss.backward()

        # Updating parameters using gradients
        optimizer.step()

        # Resetting the gradients to zero
        optimizer.zero_grad()

    # Print the progress
    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, 1000, loss.item()))

# Save training data
data = {
    "model_state": model.state_dict(),
    "input_size": inputs.shape[1],
    "output_size": 1,
    "inputs": inputs,
    "targets": targets
}

FILE = "Data/data.pth"
torch.save(data, FILE)
