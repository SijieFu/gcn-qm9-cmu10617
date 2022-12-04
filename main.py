import sys, os
import pickle
import json
import os
import argparse
from datetime import datetime as dt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchdrug import data, datasets
from torchdrug import core, models, tasks
from torchdrug import utils
from transforms import *

# Collaborators for this team project
__author__ = 'Sijie Fu, Nicholas Hattrup, Robert MacKnight'
__email__ = 'sijief | nhattrup | rmacknig@andrew.cmu.edu'
__forkedrepo__ = 'https://github.com/DeepGraphLearning/torchdrug'

parser = argparse.ArgumentParser(description='Final team project repo for CMU 10-617 (Fall 2022)')

parser.add_argument('--dataset', type=str, default='QM9.pkl', help='path to dataset file')
parser.add_argument('--out_file', type=str, default="trained_models/my_model", help='path to saved model files')
parser.add_argument('--model', type=str, default='MPNN', help='model to train GCN or MPNN')
parser.add_argument('--hidden_dim', type=str, default="256", help='space separated string, list for GCN, [single] for MPNN')
parser.add_argument('--num_layer', type=int, default=1, help='num layers for MPNN')
parser.add_argument('--num_gru_layer', type=int, default=1, help='num gru layers for MPNN')
parser.add_argument('--num_mlp_layer', type=int, default=1, help='num mlp layer for MPNN')
parser.add_argument('--num_s2s_step', type=int, default=1, help='num s2s step for MPNN')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--gpu', action='store_true', default=False, help='use GPU')
parser.add_argument('--train_size', type=float, default=0.8, help='train size')
parser.add_argument('--include_distance', action='store_true', default=False)

args = parser.parse_args()

name = args.dataset.split(".")[0]
hidden = args.hidden_dim.split()
num_layer = args.num_layer
num_gru_layer = args.num_gru_layer
num_mlp_layer = args.num_mlp_layer
num_s2s_step = args.num_s2s_step
if len(hidden) > 1:
     hidden_dims = [int(i) for i in hidden]
     model = "GCN"
elif len(hidden) == 1:
     hidden_dim = int(hidden[0])
     model = "MPNN"
if model == args.model:
     print(f"---Training {model} model---")
else:
     print(f"Mismatch in models! -- set the '--model' flag")
     sys.exit()
lr = float(args.lr)
batch_size = int(args.batch_size)
epochs = int(args.epochs)
if args.gpu and torch.cuda.is_available():
     print("\t  Using GPU resource")
     gpus = [0]
else:
     print("\t  Using CPU only")
     gpus = None

if len(args.out_file.split("/")) > 1:
     dir_to_make = "/".join(args.out_file.split("/")[:-1])
     os.system(f"mkdir -p {dir_to_make}")
json_out = args.out_file + ".json"
pickle_out = args.out_file + ".pkl"
print(f"\t  Model configuration will be saved to {json_out}\n"
      f"\t  Solver will be saved to {pickle_out}")

# load dataset
print(f"\t  Loading {name} dataset...")
with open(args.dataset, "rb") as f:
     dataset = pickle.load(f)
print("\t  Loaded.")

if args.include_distance:
     dataset.data = [edge_importance(mol) for mol in dataset.data]
# Training
lengths = [int(args.train_size * len(dataset)), int((1 - args.train_size)/2 * len(dataset))]
lengths += [len(dataset) - sum(lengths)]
train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths)
if model == "MPNN":
     t_model = models.MPNN(input_dim = dataset.node_feature_dim,
                           hidden_dim = hidden_dim,
                           edge_input_dim = dataset.edge_feature_dim,
                           num_layer = num_layer,
                           num_gru_layer = num_gru_layer,
                           num_mlp_layer = num_mlp_layer,
                           num_s2s_step = num_s2s_step)
elif model == "GCN":
     t_model = models.GCN(input_dim = dataset.node_feature_dim,
                          hidden_dims = hidden_dims,
                          edge_input_dim = dataset.edge_feature_dim)
# Define task
task = tasks.PropertyPrediction(t_model, task=dataset.tasks)
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
with open(json_out, "w") as out_file:
     json.dump(solver.config_dict(), out_file)
solver.save(pickle_out)
