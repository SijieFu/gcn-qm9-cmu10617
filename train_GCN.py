import torch, sys, json, os, pickle
sys.path.append("torchdrug/")
from torchdrug import data, datasets, core, models, tasks, utils
import numpy as np

print(f"Loading QM9 dataset...")
with open("QM9.pkl", "rb") as f:
     qm9 = pickle.load(f)
print("Loaded.")
dataset = qm9

lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
lengths += [len(dataset) - sum(lengths)]
train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths)

# Arguments
hidden_dim = 256
lr = 1e-3
batch_size = 128
epochs = 100
gpus = [0]
if torch.cuda.is_available():
     gpus = [0]
else:
     gpus = None
     
# Define model
model = models.GCN(input_dim = dataset.node_feature_dim,
                   hidden_dims = [256, 128, 64],
                   edge_input_dim = dataset.edge_feature_dim)

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
                     batch_size = batch_size,
                     logger = "wandb")

# Train model
solver.train(num_epoch=epochs)

# Save model
os.system("mkdir -p trained_models/")
with open("trained_models/gcn_qm9.json") as out_file:
     json.dump(solver.config_dict(), out_file)
solver.save("trained_models/gcn_qm9.pth")

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
