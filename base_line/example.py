import torch
import pickle
from torchdrug import datasets
from torchdrug import core, models, tasks




#dataset = datasets.QM9("~/molecule-datasets/",node_position=True)
#with open("QM9.pkl", "wb") as fout:
#    pickle.dump(dataset, fout)
#exit()
with open("QM9.pkl", "rb") as fin:
     dataset = pickle.load(fin)



lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
lengths += [len(dataset) - sum(lengths)]
train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths)

model = models.MPNN(input_dim=dataset.node_feature_dim,
                   hidden_dim=256,
                   edge_input_dim=dataset.edge_feature_dim,
                   num_layer=1, num_gru_layer=1, num_mlp_layer=2, num_s2s_step=3, short_cut=False, batch_norm=False, activation='relu', concat_hidden=False)

task = tasks.PropertyPrediction(model, task=dataset.tasks)




optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)
solver = core.Engine(task, train_set, valid_set, test_set, optimizer,
                     gpus=[0], batch_size=32)
solver.train(num_epoch=1)
#solver.evaluate("valid")
