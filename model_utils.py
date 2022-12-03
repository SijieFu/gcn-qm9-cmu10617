# PyTorch
import torch, gpytorch, json, warnings, pickle, sys, itertools
from botorch import fit_gpytorch_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
warnings.filterwarnings("ignore")
sys.path.append("torchdrug/")
from torchdrug import data, datasets, core, models, tasks, utils

# Define Gaussian Process Model with **kernel**
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # Use **kernel** for specific application
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        self.x = train_x
        self.y = train_y
        self.likelihood = likelihood
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
# function for training a GP
def train_gp(x, y):

    x = torch.tensor(x)
    y = torch.tensor(y)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(x, y, likelihood)
    model.train()
    likelihood.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    fit_gpytorch_model(mll)
    model.eval()
    likelihood.eval()

    prediction = model(torch.tensor(x))
    mean = prediction.mean.detach().numpy()
    
    model_scores = {"r2": r2_score(y.numpy(), mean)}

    return model, likelihood, model_scores
# make a suggestion
def suggest(test_input, model, likelihood, budget, mode = "explore"):

    prediction = model(torch.tensor(x))
    pred_mean = prediction.mean.detach().numpy()
    pred_var = prediction.variance.detach().numpy()

    if mode == "explore":
        indices = np.argsort(pred_var)[-budget:]
        additional_samples = test_input[indices]
    elif mode == "exploit":
        max_index = np.argmax(pred_mean)
        n = int(budget / 2)
        additional_samples = test_input[max_index-n:max_index+n]
    else:
        print("Unknown mode.")
    print(additional_samples)
    return pred_mean, pred_var
# create a config gile for hyper parameter sweep
def write_config(input_dim,
                 hidden_dims,
                 hidden_dim,
                 edge_input_dim=None,
                 short_cut=False,
                 batch_norm=False,
                 activation="relu",
                 concat_hidden=False,
                 readout="sum",
                 num_layer=1,
                 num_gru_layer=1,
                 num_mlp_layer=2,
                 num_s2s_step=3):

    print("Shared Parameters:\n\t'input_dim (mandatory)', 'edge_input_dim', 'short_cut', 'batch_norm', 'activation', 'conat_hidden', 'readout'")
    print("MPNN Parameters:\n\t'num_layer', 'num_gru_layer', 'num_mlp_layer', 'num_s2s_step', 'hidden_dim (mandatory)")
    print("GCN Parameters:\n\t'hidden_dims (mandatory)'")

    layer_options = [1, 2, 3, 4]
    readout_options = ["sum", "mean"]
    activation_fn = ["relu", "softmax", "sigmoid"]

    config = {"input_dim": [input_dim], "hidden_dims": [hidden_dims], "hidden_dim": [hidden_dim],
              "edge_input_dim": edge_input_dim, "short_cut": short_cut, "batch_norm": batch_norm,
              "activation": activation_fn, "concat_hidden": concat_hidden, "readout": readout_options,
              "num_layer": layer_options, "num_gru_layer": layer_options, "num_mlp_layer": layer_options,
              "num_s2s_step": layer_options}
    with open("sweep.json", "w") as f:
        json.dump(config, f)
        
# read a config file for hyper parameter sweep
def load_config(json_file):
    with open(json_file, "r") as config:
        return json.load(config)
# generate sample space
def make_sample_space(config, params_to_tune = ["num_layer"], learning_rate = [1e-4, 1e-3, 1e-2]):
    options = []
    for key in config.keys():
        if key in params_to_tune:
            if isinstance(config[key], list):
                options.append(config[key])
            else:
                options.append([config[key]])
    if isinstance(learning_rate, list):
        options.append(learning_rate)
    sample_space = list(itertools.product(*options))
    return sample_space
# add distance to edge of molecular graph
def edge_importance(molecular_graph):
    test_mol = molecular_graph
    # add dummy dimesnion to edges to replace with distances
    n_edges = molecule.num_edge
    updated_edges = torch.zeros(size = (test_mol.edge_list.size()))
    test_mol.edge_feature = torch.hstack((test_mol.edge_feature,  torch.zeros(size = (test_mol.edge_feature.size()[0], 1))))
    # iterate edges
    bond_distances = []
    for i, edge in enumerate(test_mol.edge_feature):
        nodes = test_mol.edge_list[i][:-1].tolist()
        n1_i, n2_i = nodes[0], nodes[1]
        n1_pos, n2_pos = test_mol.node_position[n1_i], test_mol.node_position[n2_i]
        distance = np.linalg.norm(n1_pos - n2_pos)
        bond_distances.append(distance)
    bond_distances = np.array(bond_distances).reshape(-1, 1)
    # reciprocal of distance
    test_mol.edge_feature[:, -1] = 1 / torch.tensor(bond_distances)[:, 0]

    return test_mol
    
