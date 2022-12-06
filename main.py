import sys, os
import pickle
import json
import os
import argparse
from datetime import datetime as dt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchdrug import data, datasets
from torchdrug import core, models, tasks
from torchdrug import utils
from utils import *
from sklearn.preprocessing import MinMaxScaler

# Collaborators for this team project
__author__ = 'Sijie Fu, Nicholas Hattrup, Robert MacKnight'
__email__ = 'sijief | nhattrup | rmacknig@andrew.cmu.edu'
__forkedrepo__ = 'https://github.com/DeepGraphLearning/torchdrug'

parser = argparse.ArgumentParser(description='Final team project repo for CMU 10-617 (Fall 2022)')

# to test if our stuff works, a set with only 100 examples will be used
parser.add_argument('--minitest', action='store_false', default=True, help='when testing the algorithm, use the mini dataset')
parser.add_argument('--dataset', type=str, default='QM9.pkl', help='path to dataset file in the subfolder ./dataset/')
parser.add_argument('--onthefly', action='store_false', default=True,  help='whether to generate new dataset/.pkl file')
parser.add_argument('--model_path', type=str, default="./models/", help='default path to save is ./models/')
parser.add_argument('--load_model', action="store_true", default=False, help='whether or not you would like to load a model')
parser.add_argument('--out_file', type=str, default="", help='name for output file (default path to save is ./models/)')
parser.add_argument('--model', type=str, default='GCN', help='model to train GCN or MPNN (default is GCN)')
parser.add_argument('--load_params', type=str, default='None', help='to load hyperparameters and not have to set them')
parser.add_argument('--hidden_dim', type=str, default="256", help='underscore separated string, list for GCN, [single] for MPNN')
parser.add_argument('--num_layer', type=int, default=1, help='num layers for MPNN')
parser.add_argument('--num_gru_layer', type=int, default=1, help='num gru layers for MPNN')
parser.add_argument('--num_mlp_layer', type=int, default=1, help='num mlp layer for MPNN')
parser.add_argument('--num_s2s_step', type=int, default=1, help='num s2s step for MPNN')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--gpu', action='store_true', default=False, help='use GPU')
parser.add_argument('--test_size', type=float, default=10000, help='test size (same for validation, rest for training) [default: 10000]')
parser.add_argument('--include_distance', action='store_true', default=False)
parser.add_argument('--param_opt', type=str, default = None, help='config file for hyperparameter optimization')

def main():
     
     global args
     # Parse arguments
     args = parser.parse_args()
     # load model hyper parameters from config
     if args.load_params != 'None':
          params_dict = json.load(open(args.load_params, "r"))
          try:
               if args.model == 'MPNN':
                    args.hidden_dim = str(params_dict["hidden_dim"])
                    args.num_layer = params_dict["num_layer"]
                    args.num_gru_layer = params_dict["num_gru_layer"]
                    args.num_mlp_layer = params_dict["num_mlp_layer"]
                    args.num_s2s_step = params_dict["num_s2s_step"]
               elif args.model == "GCN":
                    args.hidden_dim = "_".join([str(i) for i in params_dict["hidden_dims"]])
               else:
                    print(f"Mismatching config ({args.load_params}) and model ({args.model})")
          except:
               print(f"Error parsing ({args.load_params}) for model ({args.model})")
          args.lr = params_dict["lr"]
          args.batch_size = params_dict["batch_size"]
     # dataset
     name = args.dataset.split(".")[0]

     # architecture
     hidden = args.hidden_dim.split('_')
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
     
     # learning rate, batch size, epochs
     lr = float(args.lr)
     batch_size = int(args.batch_size)
     epochs = int(args.epochs)

     # GPU
     if args.gpu and torch.cuda.is_available():
          print("\t  Using GPU resource")
          gpus = [0]
     else:
          print("\t  Using CPU only")
          gpus = None
     
     # saving configureation
     if not os.path.exists(args.model_path):
          os.mkdir(args.model_path)
     out_file = model if args.out_file == "" else args.out_file
     while True:
          if out_file+'.json' in os.listdir(args.model_path) or out_file+'.pkl' in os.listdir(args.model_path):
               print(f"\t  Model out_file naming {out_file} already exists. Using {out_file}_new instead.")
               out_file += "_new"
          else:
               print(f"\t  Using {out_file} as the (new) name for outfiles. They will be in {args.model_path}")
               print(f"\t***** REMEMBER TO MANUALLY CHANGE THE OUTFILE NAMES TO AVOIND CONFUSION. *****")
               break
     json_out = args.model_path + out_file + ".json"
     pickle_out = args.model_path + out_file + ".pkl"
     print(f"\t  Model configuration will be saved to {json_out}\n"
           f"\t  Solver will be saved to {pickle_out}")
     
     # load dataset, for now, dataset will be generated on the fly by default
     print(f"\t  Loading {name} dataset...")

     path_to_dataset = args.dataset.replace('.pkl', '_mini.pkl') if args.minitest else args.dataset
     if not os.path.exists('./dataset/' + path_to_dataset) or args.onthefly:
          print(f"\t  Dataset ./dataset/{path_to_dataset} does not exist. Building from scratch.")
          # feature selection for nodes and edges
          atom_feature = "default"
          bond_feature = ["default", "length"] if args.include_distance else "default"
          dataset = datasets.QM9(path='./dataset/', node_position=True, minitest=args.minitest,
                              atom_feature=atom_feature, bond_feature=bond_feature, mol_feature=None,
                              with_hydrogen=False, kekulize=True)
          if not args.onthefly:
               with open('./dataset/' + path_to_dataset, "wb") as f:
                    pickle.dump(dataset, f)
               print(f"\t  Done building: ./dataset/{path_to_dataset}")
     else:
          with open('./dataset/' + path_to_dataset, "rb") as f:
               dataset = pickle.load(f)
          print(f"\t  Loaded dataset: ./dataset/{path_to_dataset}")
     
     # model training
     if args.minitest:
          lengths = [0.8, 0.1, 0.1]
     elif args.test_size < 0.5:
          lengths = [1.0 - 2 * args.test_size, args.test_size, args.test_size]
     elif args.test_size > 1000:
          lengths = [len(dataset) - 2 * int(args.test_size), int(args.test_size), int(args.test_size)]
     else:
          lengths = [0.8, 0.1, 0.1] if len(dataset) < 20000 else [len(dataset) - 20000, 10000, 10000]
          print(f"\t test_size argument (--test_size {args.test_size}) is illegal. Using default 10k or 0.1 for test.")
     if sum(lengths) == 1.0:
          lengths = [int(x*len(dataset)) for x in lengths]
          lengths[0] = len(dataset) - sum(lengths[1:])
     train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))

     scalers = {task: MinMaxScaler() for task in dataset.tasks}
     for task, scaler in scalers.items():
          scaler.fit(np.asarray(dataset.targets[task]).reshape(-1, 1)[train_set.indices])
          scaled_targert = scaler.transform(np.asarray(dataset.targets[task]).reshape(-1, 1)).reshape(-1).tolist()
          train_set.dataset.targets[task] = scaled_targert
          valid_set.dataset.targets[task] = scaled_targert
          test_set.dataset.targets[task] = scaled_targert
     with open(f"{args.model_path + out_file}_scalers.pkl", 'wb') as f:
          pickle.dump(scalers, f)
     print(f"\t Every target is now scaled with MinMaxScaler(). All scalers are saved to {args.model_path + out_file}_scalers.pkl")

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
     # task
     task = tasks.PropertyPrediction(t_model, task=dataset.tasks)
     # optimizer
     optimizer = torch.optim.Adam(task.parameters(), lr=lr)
     # solver (logger?)
     solver = core.Engine(task,
                          train_set,
                          valid_set,
                          test_set,
                          optimizer,
                          gpus = gpus,
                          batch_size = batch_size)
     
     # train or load model
     if args.load_model:
          pickle_in = args.model_path + out_file + ".pkl"
          solver.load(pickle_in)
          print(f"\t Loaded model: {pickle_in}")
     else:
          solver.train(num_epoch=epochs)
          # save model
          with open(json_out, "w") as out_model:
               json.dump(solver.config_dict(), out_model)
          solver.save(pickle_out)
     
     # evaluate model ... MAE and RMSE for all properties
     train_metric = solver.evaluate("train", log=True)
     with open(args.model_path + out_file + "_train_metric.json", "w") as ft:
          json_train_metric = {k: v.item() for k, v in train_metric.items()}
          json.dump(json_train_metric, ft)
     
     val_metric = solver.evaluate("valid", log=True)
     with open(args.model_path + out_file + "_val_metric.json", "w") as fval:
          json_val_metric = {k: v.item() for k, v in val_metric.items()}
          json.dump(json_val_metric, fval)
     
     test_metric = solver.evaluate("test", log=True)
     with open(args.model_path + out_file + "_test_metric.json", "w") as ftest:
          json_test_metric = {k: v.item() for k, v in test_metric.items()}
          json.dump(json_test_metric, ftest)
     print(f"\t  Saved metrics to {args.model_path + out_file}_*_metric.json")
     
if __name__ == '__main__':
    main()