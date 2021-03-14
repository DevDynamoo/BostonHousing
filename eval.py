import numpy as np
import torch
from model import NeuralNetwork


# Loading training data file
FILE = "Data/data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
output_size = data["output_size"]
model_state = data["model_state"]
inputs = data["inputs"]
targets = data["targets"]

# Model
model = NeuralNetwork(input_size, output_size)
model.load_state_dict(model_state)
model.eval()

# Predicted values
output = model(inputs)
prediction = output.detach().numpy()

# Target values
targets = targets.numpy()

# Concatinating and saving the predicted values from model and target values
predictions_target = np.concatenate((prediction, targets), axis=1)

np.savetxt("Data/predictions_and_targets.csv", predictions_target, delimiter=',', header="prediction,target",
           comments="")
