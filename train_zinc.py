import argparse
import moses
import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate
from zinc.utils import *

from zinc.GraphDataset import GraphDataset

# <editor-fold desc="...">
parser = argparse.ArgumentParser(description='HyperbolicDiffusion')
parser.add_argument('--exp_name', type=str, default='Diffusion_AE_HGCN_cwitht_6')
parser.add_argument('--wandb_usr', type=str, default='elma')
parser.add_argument('--no_wandb', default=False, action='store_true', help='Disable wandb')
parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')

# Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--diffusion_steps', type=int, default=1000)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_noise_precision', type=float, default=2e-4)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
# EGNN args -->
parser.add_argument('--n_layers', type=int, default=9,
                    help='number of layers')
parser.add_argument('--nf', type=int, default=256,
                    help='dim of EGNN hidden feature')
parser.add_argument('--dim', type=int, default=6,
                    help='dim of encoder output')
# <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--train_file', type=str, default='zinc/molecules_train_zinc.json')
parser.add_argument('--test_file', type=str, default='zinc/molecules_test_zinc.json')
parser.add_argument('--dev_file', type=str, default='zinc/molecules_valid_zinc.json')
parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=1)
parser.add_argument('--data_augmentation', type=eval, default=True, help='')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='arguments : homo | lumo | alpha | gap | mu | Cv')
parser.add_argument('--resume', type=str, default=None, help='')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--n_stability_samples', type=int, default=500,
                    help='Number of samples to compute the stability')
parser.add_argument('--normalize_factors', type=eval, default=[1, 1, 1],
                    help='normalize factors for [x, categorical, integer]')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=True,
                    help='include atom charge or not')
parser.add_argument('--visualize_epoch', type=int, default=5,
                    help="Can be used to visualize multiple times per epoch")
parser.add_argument('--normalization_factor', type=float, default=1,
                    help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean"')
# zinc
parser.add_argument('--edge_type', type=int, default=6)
parser.add_argument('--num_feature', type=int, default=15)


# </editor-fold>
def collate_fn(batch):
    max_neighbor_num = -1
    for data in batch:
        for row in data['adj_mat']:
            max_neighbor_num = max(max_neighbor_num, len(row))

    for data in batch:
        # pad the adjacency list
        data['adj_mat'] = pad_sequence(data['adj_mat'], maxlen=max_neighbor_num)
        data['weight'] = pad_sequence(data['weight'], maxlen=max_neighbor_num)

        data['node'] = np.array(data['node']).astype(np.float32)
        # pad = np.zeros((max_neighbor_num-data['node'].shape[0],data['node'].shape[1]))
        # data['node'] = np.concatenate((data['node'],pad),axis=0)
        data['adj_mat'] = np.array(data['adj_mat']).astype(np.int32)
        data['weight'] = np.array(data['weight']).astype(np.float32)
        data['label'] = np.array(data['label'])
    return default_collate(batch)


args = parser.parse_args()

test_dataset = GraphDataset(args, split='test')
test_loader = DataLoader(test_dataset, batch_size=1,collate_fn=collate_fn)
for dic in test_loader:
    node,adj_mat,weight,label = dic['node'],dic['adj_mat'],dic['weight'],dic['label']
    smile = graph_to_smile(node[0],weight[0])
    metrics = moses.get_all_metrics(smile)
    print(smile)


