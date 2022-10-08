# Rdkit import should be first, do not move it
import copy
import utils.utils as utils
import argparse
import wandb

from AutoEncoder.AutoEncoder import HyperbolicAE
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9 import dataset
from qm9.models import get_optim
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
import torch
import time
import pickle
from qm9.utils import prepare_context, compute_mean_mad
from train_test import train_AE_epoch, test_AE

parser = argparse.ArgumentParser(description='E3Diffusion')
parser.add_argument('--exp_name', type=str, default='AutoEncoder')

parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--dim', type=int, default=20)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--bias', type=int, default=1)
parser.add_argument('--max_z', type=int, default=6)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--model', type=str, default='MLP',
                    help='MLP,HNN,GCN,HGCN')
parser.add_argument('--manifold', type=str, default='Euclidean',
                    help='Euclidean, Hyperboloid, PoincareBall')
parser.add_argument('--c', type=float, default=None)
parser.add_argument('--act', type=str, default='selu',
                    help='relu,silu,selu')
parser.add_argument('--local_agg', type=int, default=1)
parser.add_argument('--encdec_share_curvature', type=eval, default=False,
                    help='True | False')
# Epoch: 1 Best val loss: 10.9893 Epoch: 2, iter: 0/391, Loss 0.1327
parser.add_argument('--brute_force', type=eval, default=False,
                    help='True | False')
parser.add_argument('--actnorm', type=eval, default=True,
                    help='True | False')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')
# <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='qm9',
                    help='qm9 | qm9_second_half (train only on the last 50K samples of the training dataset)')
parser.add_argument('--datadir', type=str, default='qm9/temp',
                    help='qm9 directory')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=1)
parser.add_argument('--wandb_usr', type=str)
parser.add_argument('--no_wandb', default=True,action='store_true', help='Disable wandb')
parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--save_model', type=eval, default=True,
                    help='save model')
parser.add_argument('--generate_epochs', type=int, default=1,
                    help='save model')
parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=1)
parser.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='arguments : homo | lumo | alpha | gap | mu | Cv' )
parser.add_argument('--resume', type=str, default=None,
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--ema_decay', type=float, default=0.999,
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--n_stability_samples', type=int, default=500,
                    help='Number of samples to compute the stability')
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 1],
                    help='normalize factors for [x, categorical, integer]')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=True,
                    help='include atom charge or not')
parser.add_argument('--visualize_every_batch', type=int, default=1e8,
                    help="Can be used to visualize multiple times per epoch")
parser.add_argument('--normalization_factor', type=float, default=10,
                    help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean"')
args = parser.parse_args()

dataset_info = get_dataset_info(args.dataset, args.remove_h)

atom_encoder = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']

# args, unparsed_args = parser.parse_known_args()
args.wandb_usr = utils.get_wandb_username(args.wandb_usr)

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32

if args.resume is not None:
    exp_name = args.exp_name + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume
    wandb_usr = args.wandb_usr
    normalization_factor = args.normalization_factor
    aggregation_method = args.aggregation_method

    with open(join(args.resume, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    args.resume = resume
    args.break_train_epoch = False

    args.exp_name = exp_name
    args.start_epoch = start_epoch
    args.wandb_usr = wandb_usr

    # Careful with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = normalization_factor
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = aggregation_method

    print(args)

utils.create_folders(args)
# print(args)


# Wandb config
if args.no_wandb:
    mode = 'disabled'
else:
    mode = 'online' if args.online else 'offline'
kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': 'e3_diffusion', 'config': args,
          'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
wandb.init(**kwargs)
wandb.save('*.txt')

# Retrieve QM9 dataloaders

dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

data_dummy = next(iter(dataloaders['train']))
# print(data_dummy['one_hot'].shape) # [128, 25, 5] 只有5种原子 H,C,O,N,F padding是全false
# print(data_dummy['edge_mask']) (b*n_atom*n_atom,1) atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2) i*i=0  defined in qm9/data/collate.py
# print(data_dummy['atom_mask'].shape) (b,n_atom)

#dict_keys(['num_atoms', 'charges', 'positions', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo',
# 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv', 'omega1', 'zpve_thermo', 'U0_thermo', 'U_thermo', 'H_thermo',
# 'G_thermo', 'Cv_thermo', 'one_hot', 'atom_mask', 'edge_mask'])


if len(args.conditioning) > 0:
    print(f'Conditioning on {args.conditioning}')
    property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
    context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
    context_node_nf = context_dummy.size(2)
else:
    context_node_nf = 0
    property_norms = None

args.context_node_nf = context_node_nf


# Create EGNN flow
model = HyperbolicAE(args)  # model=EnVariationalDiffusion 包含EGNN_dynamics_QM9

model = model.to(device)

optim = get_optim(args, model)
# print(model)

gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)  # Add large value that will be flushed.


def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def main():
    if args.resume is not None:
        flow_state_dict = torch.load(join(args.resume, 'flow.npy'))
        optim_state_dict = torch.load(join(args.resume, 'optim.npy'))
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)

    # Initialize dataparallel if enabled and possible.
    if args.dp and torch.cuda.device_count() > 1:
        print(f'Training using {torch.cuda.device_count()} GPUs')
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()
    else:
        model_dp = model

    # Initialize model copy for exponential moving average of params.
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = flow_utils.EMA(args.ema_decay)

        if args.dp and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.DataParallel(model_ema)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp

    best_nll_val = 1e8
    best_nll_test = 1e8
    for epoch in range(args.start_epoch, args.n_epochs):
        start_epoch = time.time()
        train_AE_epoch(args=args, loader=dataloaders['train'], epoch=epoch, model=model, model_dp=model_dp,
                    model_ema=model_ema, ema=ema, device=device, dtype=dtype, property_norms=property_norms,
                    gradnorm_queue=gradnorm_queue, optim=optim)
        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")

        if epoch % args.test_epochs == 0:


            nll_val = test_AE(args=args, loader=dataloaders['valid'], epoch=epoch, eval_model=model_ema_dp,
                           partition='Val', device=device, dtype=dtype, property_norms=property_norms)
            # nll_test = test(args=args, loader=dataloaders['test'], epoch=epoch, eval_model=model_ema_dp,
            #                 partition='Test', device=device, dtype=dtype,
            #                 nodes_dist=nodes_dist, property_norms=property_norms)

            if nll_val < best_nll_val:
                best_nll_val = nll_val
                # best_nll_test = nll_test
                if args.save_model:
                    args.current_epoch = epoch + 1
                    utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
                    utils.save_model(model, 'outputs/%s/AE.npy' % args.exp_name)
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/AE_ema.npy' % args.exp_name)
                    with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                        pickle.dump(args, f)

                if args.save_model:
                    utils.save_model(optim, 'outputs/%s/optim_%d.npy' % (args.exp_name, epoch))
                    utils.save_model(model, 'outputs/%s/AE_%d.npy' % (args.exp_name, epoch))
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/AE_ema_%d.npy' % (args.exp_name, epoch))
                    with open('outputs/%s/args_%d.pickle' % (args.exp_name, epoch), 'wb') as f:
                        pickle.dump(args, f)
            # print('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
            print('Val loss: %.4f ' % (nll_val))
            print('Best val loss: %.4f' % (best_nll_val))
            wandb.log({"Val loss ": nll_val}, commit=True)
            # wandb.log({"Test loss ": nll_test}, commit=True)
            # wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True)


if __name__ == "__main__":
    main()
