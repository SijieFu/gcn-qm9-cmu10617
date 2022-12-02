import torch
import sys
sys.path.append("torchdrug/")
from torchdrug import data, datasets, core, models, tasks, utils
import pickle as pkl
import numpy as np
import os

print(f"Loading QM9 dataset...")
with open("QM9.pkl", "rb") as f:
     qm9 = pkl.load(f)
print("Loaded.")
dataset = qm9

lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
lengths += [len(dataset) - sum(lengths)]
train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths)

# Arguments
hidden_dim = 256
lr = 1e-3
batch_size = 32
epochs = 100
if torch.cuda.is_available():
     gpus = [0]
else:
     gpus = None

# Define model
model = models.MPNN(input_dim = dataset.node_feature_dim,
                    hidden_dim = hidden_dim,
                    edge_input_dim = dataset.edge_feature_dim,
                    num_layer = 1,
                    num_gru_layer = 1,
                    num_mlp_layer = 2,
                    num_s2s_step = 3)

# Define task
task = tasks.PropertyPrediction(model, task=dataset.tasks)

# Optimizer
optimizer = torch.optim.Adam(task.parameters(), lr=lr)

# Solver
solver = core.Engine(task,
                     train_set,
                     valid_set,
                     test_set,
                     optimizer,
                     gpus = gpus,
                     batch_size = batch_size)

# Train model
solver.train(num_epoch=epochs)

# Save model
os.system("mkdir -p trained_models/")
with open("trained_models/mpnn_qm9.json", "w") as out_file:
     json.dump(solver.config_dict(), out_file)
solver.save("trained_models/mpnn_qm9.pth")

print("---Final evaluation---")
# Evaluate model on training set
print("Train:")
solver.evaluate("train")
# Evaluate model on validation set
print("Validation:")
solver.evaluate("valid")
# Evaluate model on test set
print("Test:")
solver.evaluate("test")
