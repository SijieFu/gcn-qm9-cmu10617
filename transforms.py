# PyTorch
import torch
import numpy as np

# add distance to edge of molecular graph                                                                                                                                                            
def edge_importance(molecular_graph):
    test_mol = molecular_graph
    # add dummy dimesnion to edges to replace with distances                                                                                                                                               
    n_edges = test_mol.num_edge
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
