import torch, sys, json, os, pickle, wandb, yaml
sys.path.append("torchdrug/")
from torchdrug import data, datasets, core, models, tasks, utils
import numpy as np



with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    

# Get hyperparameters from config file 
run = wandb.init(config=config)
lr = wandb.config.learning_rate
bs = wandb.config.batch_size
hidden_dim = wandb.config.hidden_dims
num_layer= wandb.config.num_layer
num_gru_layer = wandb.config.num_gru_layer
num_mlp_layer = wandb.config.num_mlp_layer
num_s2s_step = wandb.config.num_s2s_step










print(f"Loading QM9 dataset...")
with open("base_line/QM9.pkl", "rb") as f:
     qm9 = pickle.load(f)
print("Loaded.")
dataset = qm9

lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
lengths += [len(dataset) - sum(lengths)]
train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths)

epochs = 100 # Default for now 
if torch.cuda.is_available():
     gpus = [0]
else:
     gpus = None

# Define model
model = models.MPNN(input_dim = dataset.node_feature_dim,
                    hidden_dim = hidden_dim,
                    edge_input_dim = dataset.edge_feature_dim,
                    num_layer = num_layer,
                    num_gru_layer = num_gru_layer,
                    num_mlp_layer = num_mlp_layer,
                    num_s2s_step = num_s2s_step)

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
                     batch_size = bs)

# Train model
for epoch in range(epochs):
    solver.train()
    val_loss = solver.evaluate("valid")
    wandb.log({
        'val_loss': val_loss
      })



'''
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
'''
