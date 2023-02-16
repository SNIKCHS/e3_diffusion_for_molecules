from torch import nn

import wandb
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask, \
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
import numpy as np
import qm9.visualizer as vis
from qm9.analyze import analyze_stability_for_molecules
from qm9.sampling import sample_chain, sample, sample_sweep_conditional
import utils.utils as utils
import qm9.utils as qm9utils
from qm9 import losses
import time
import torch


def train_AE_epoch(args, loader, epoch, model, model_dp, model_ema, ema, device, dtype, property_norms, optim,
                   gradnorm_queue, lr_scheduler):
    # torch.autograd.set_detect_anomaly(True)
    # torch.set_printoptions(profile="full")
    model_dp.train()
    model.train()
    nll_epoch = []
    n_iterations = len(loader)
    for i, data in enumerate(loader):
        start = time.time()
        x = data['positions'].to(device, dtype)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)  # (b,n_atom,1)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)

        categories = (torch.argmax(one_hot.int(), dim=2) + 1) * node_mask.squeeze()  # (b,n_nodes) o为padding，1~5
        charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

        x = remove_mean_with_mask(x, node_mask)
        if args.data_augmentation:
            x = utils.random_rotation(x).detach()
        # eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
        # x = x + eps * args.augment_noise
        # x = remove_mean_with_mask(x, node_mask)
        h = (categories.long(), charges)

        if len(args.conditioning) > 0:
            context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
            assert_correctly_masked(context, node_mask)
        else:
            context = None

        optim.zero_grad()

        # transform batch through flow
        rec_loss, KL_loss,edge_loss = model_dp(x, h, node_mask, edge_mask)
        loss = rec_loss + args.ode_regularization*KL_loss+args.ode_regularization*edge_loss
        if torch.isnan(loss):
            raise AssertionError

        loss.backward()
        # for n,p in model.named_parameters():
        #     print(n)
        #     print(p.grad)
        if args.clip_grad:
            grad_norm = utils.gradient_clipping(model, gradnorm_queue)
        else:
            grad_norm = 0.

        optim.step()

        if args.lr_scheduler:
            lr_scheduler.step(loss)

        # Update EMA if enabled.
        if args.ema_decay > 0:
            ema.update_model_average(model_ema, model)


        print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
              f"Loss {loss.item():.4f}, rec_loss: {rec_loss.item():.4f},edge_loss: {edge_loss.item():.4f}, KL_loss: {KL_loss.item():.4f}, "
              f"GradNorm: {grad_norm:.1f},time:{time.time() - start:.4f}")
        nll_epoch.append(loss.item())
        wandb.log({"Batch NLL": loss.item(), 'rec_loss': rec_loss.item(), 'KL_loss': KL_loss.item(),'edge_loss': edge_loss.item()},
                  commit=True)
    model.show_curvatures()
    wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)


def train_HyperbolicDiffusion_epoch(args, loader, epoch, model, model_dp, model_ema, ema, device, dtype, property_norms,
                                    optim, nodes_dist, gradnorm_queue, dataset_info, prop_dist):
    # torch.autograd.set_detect_anomaly(True)
    model_dp.train()
    model.train()
    nll_epoch = []
    n_iterations = len(loader)

    for i, data in enumerate(loader):
        start = time.time()
        x = data['positions'].to(device, dtype)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)
        categories = (torch.argmax(one_hot.int(), dim=2) + 1) * node_mask.squeeze()  # (b,n_nodes) o为padding，1~5
        h = (categories.long(), charges)

        x = remove_mean_with_mask(x, node_mask)

        if args.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * args.augment_noise

        x = remove_mean_with_mask(x, node_mask)
        if args.data_augmentation:
            x = utils.random_rotation(x).detach()

        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        if len(args.conditioning) > 0:
            context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
            assert_correctly_masked(context, node_mask)
        else:
            context = None

        optim.zero_grad()

        # transform batch through flow
        nll, reg_term, mean_abs_z,error_x,error_t = losses.compute_loss_and_nll(args, model_dp, nodes_dist,
                                                                x, h, node_mask, edge_mask, context)

        # standard nll from forward KL
        loss = nll + args.ode_regularization * reg_term
        if torch.isnan(loss):
            # with torch.no_grad():
            #     t = torch.arange(0, 1, 0.1).unsqueeze(-1).to(device, dtype)
            #     c = model.dynamics.egnn.curvature_net(t)
            #     print(c)
            utils.save_model(model, 'outputs/%s/error_model.npy' % args.exp_name)
            raise AssertionError
        loss.backward()

        if args.clip_grad:
            grad_norm = utils.gradient_clipping(model, gradnorm_queue)
        else:
            grad_norm = 0.

        optim.step()

        # with torch.no_grad():
        #     t = torch.arange(0,1,0.1).unsqueeze(-1).to(device, dtype)
        #     c = model.dynamics.egnn.curvature_net(t)
        #     print(c)
        # Update EMA if enabled.
        if args.ema_decay > 0:
            ema.update_model_average(model_ema, model)

        if i % args.n_report_steps == 0:
            print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                  f"Loss {loss.item():.4f}, "
                  f"error_x: {error_x.item():.6f},error_t: {error_t.item():.6f}"
                  f"GradNorm: {grad_norm:.1f}, abs_z: {mean_abs_z.item():.6f} "
                  f"time:{time.time() - start:.4f}")

        nll_epoch.append(nll.item())
        wandb.log({'abs_z': mean_abs_z.item(),'error_x':error_x.item(),'error_t':error_t.item()}, commit=True)
        if args.break_train_epoch:
            break

    # sample_different_sizes_and_save(model_dp, nodes_dist, args, device, dataset_info,
    #                                 prop_dist, epoch=epoch)
    # vis.visualize(f"outputs/{args.exp_name}/epoch_{epoch}_", dataset_info=dataset_info, wandb=wandb)
    if args.hyp:
        for m in model.dynamics.egnn.manifolds:
            print(m.k.item())
    if epoch % args.visualize_epoch == 0 and epoch!=0:
        start = time.time()
        if len(args.conditioning) > 0:
            save_and_sample_conditional(args, device, model_ema, prop_dist, dataset_info, epoch=epoch)
        save_and_sample_chain(model_ema, args, device, dataset_info, prop_dist, epoch=epoch)

        sample_different_sizes_and_save(model_ema, nodes_dist, args, device, dataset_info,
                                        prop_dist, epoch=epoch)
        # 暂时先用model_dp
        # sample_different_sizes_and_save(model_dp, nodes_dist, args, device, dataset_info,
        #                                 prop_dist, epoch=epoch)
        print(f'Sampling took {time.time() - start:.2f} seconds')

        vis.visualize(f"outputs/{args.exp_name}/epoch_{epoch}_", dataset_info=dataset_info, wandb=wandb)
        vis.visualize_chain(f"outputs/{args.exp_name}/epoch_{epoch}_/chain/", dataset_info, wandb=wandb)
        if len(args.conditioning) > 0:
            vis.visualize_chain("outputs/%s/epoch_%d_/conditional/" % (args.exp_name, epoch), dataset_info,
                                wandb=wandb, mode='conditional')
    wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)


def train_epoch(args, loader, epoch, model, model_dp, model_ema, ema, device, dtype, property_norms, optim,
                nodes_dist, gradnorm_queue, dataset_info, prop_dist):
    model_dp.train()
    model.train()
    nll_epoch = []
    n_iterations = len(loader)
    for i, data in enumerate(loader):
        start = time.time()
        x = data['positions'].to(device, dtype)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

        x = remove_mean_with_mask(x, node_mask)

        if args.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * args.augment_noise

        x = remove_mean_with_mask(x, node_mask)
        if args.data_augmentation:
            x = utils.random_rotation(x).detach()

        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        h = {'categorical': one_hot, 'integer': charges}

        if len(args.conditioning) > 0:
            context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
            assert_correctly_masked(context, node_mask)
        else:
            context = None

        optim.zero_grad()

        # transform batch through flow
        nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(args, model_dp, nodes_dist,
                                                                x, h, node_mask, edge_mask, context)
        mean_abs_z = mean_abs_z.mean()
        # standard nll from forward KL
        loss = nll + args.ode_regularization * reg_term
        loss.backward()
        if args.clip_grad:
            grad_norm = utils.gradient_clipping(model, gradnorm_queue)
        else:
            grad_norm = 0.

        optim.step()

        # Update EMA if enabled.
        if args.ema_decay > 0:
            ema.update_model_average(model_ema, model)

        if i % args.n_report_steps == 0:
            print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                  f"Loss {loss.item():.4f}, NLL: {nll.item():.4f}, abs_z: {mean_abs_z.item():.6f} "
                  f"RegTerm: {reg_term.item():.1f}, "
                  f"GradNorm: {grad_norm:.1f}"
                  f"time:{time.time() - start:.4f}")
        nll_epoch.append(nll.item())
        wandb.log({"Batch NLL": nll.item(), 'abs_z': mean_abs_z.item()}, commit=True)
        if args.break_train_epoch:
            break

    if epoch % args.visualize_epoch == 0 and epoch != 0:
        start = time.time()
        if len(args.conditioning) > 0:
            save_and_sample_conditional(args, device, model_ema, prop_dist, dataset_info, epoch=epoch)
        save_and_sample_chain(model_ema, args, device, dataset_info, prop_dist, epoch=epoch)
        sample_different_sizes_and_save(model_ema, nodes_dist, args, device, dataset_info,
                                        prop_dist, epoch=epoch)
        print(f'Sampling took {time.time() - start:.2f} seconds')

        vis.visualize(f"outputs/{args.exp_name}/epoch_{epoch}_", dataset_info=dataset_info, wandb=wandb)
        vis.visualize_chain(f"outputs/{args.exp_name}/epoch_{epoch}_/chain/", dataset_info, wandb=wandb)
        if len(args.conditioning) > 0:
            vis.visualize_chain("outputs/%s/epoch_%d_/conditional/" % (args.exp_name, epoch), dataset_info,
                                wandb=wandb, mode='conditional')
    wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def test_AE(args, loader, epoch, eval_model, device, dtype, property_norms, partition='Test'):
    eval_model.eval()
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0

        n_iterations = len(loader)

        for i, data in enumerate(loader):
            x = data['positions'].to(device, dtype)
            batch_size = x.size(0)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            categories = (torch.argmax(one_hot.int(), dim=2) + 1) * node_mask.squeeze()  # (b,n_nodes) o为padding，1~5
            categories = categories.to(device)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

            x = remove_mean_with_mask(x, node_mask)

            h = (categories.long(), charges)
            if len(args.conditioning) > 0:
                context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
                assert_correctly_masked(context, node_mask)
            else:
                context = None

            # transform batch through flow
            rec_loss, KL_loss,edge_loss = eval_model(x, h, node_mask, edge_mask)
            # standard nll from forward KL
            nll = rec_loss + args.ode_regularization*KL_loss +args.ode_regularization*edge_loss
            # standard nll from forward KL

            nll_epoch += nll * batch_size
            n_samples += batch_size
            if i % args.n_report_steps == 0:
                print(f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                      f"NLL: {nll.item():.6f} \t node_loss: {rec_loss.item():.6f} \tedge_loss: {edge_loss.item():.6f} \t KL_loss: {KL_loss.item():.6f}")

    return nll_epoch / n_samples


def test_HyperbolicDiffusion(args, loader, epoch, eval_model, device, dtype, property_norms, nodes_dist,
                             partition='Test'):
    eval_model.eval()
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0

        n_iterations = len(loader)

        for i, data in enumerate(loader):
            x = data['positions'].to(device, dtype)
            batch_size = x.size(0)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)
            categories = (torch.argmax(one_hot.int(), dim=2) + 1) * node_mask.squeeze()  # (b,n_nodes) o为padding，1~5
            h = (categories.long(), charges)

            if args.augment_noise > 0:
                # Add noise eps ~ N(0, augment_noise) around points.
                eps = sample_center_gravity_zero_gaussian_with_mask(x.size(),
                                                                    x.device,
                                                                    node_mask)
                x = x + eps * args.augment_noise

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, one_hot, charges], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            if len(args.conditioning) > 0:
                context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
                assert_correctly_masked(context, node_mask)
            else:
                context = None

            # transform batch through flow
            nll, _, _, _, _ = losses.compute_loss_and_nll(args, eval_model, nodes_dist, x, h,
                                                    node_mask, edge_mask, context)
            # standard nll from forward KL

            nll_epoch += nll.item() * batch_size
            n_samples += batch_size
            if i % args.n_report_steps == 0:
                print(f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                      f"NLL: {nll_epoch / n_samples:.4f}")

    return nll_epoch / n_samples


def test(args, loader, epoch, eval_model, device, dtype, property_norms, nodes_dist, partition='Test'):
    eval_model.eval()
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0

        n_iterations = len(loader)

        for i, data in enumerate(loader):
            x = data['positions'].to(device, dtype)
            batch_size = x.size(0)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

            if args.augment_noise > 0:
                # Add noise eps ~ N(0, augment_noise) around points.
                eps = sample_center_gravity_zero_gaussian_with_mask(x.size(),
                                                                    x.device,
                                                                    node_mask)
                x = x + eps * args.augment_noise

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, one_hot, charges], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            h = {'categorical': one_hot, 'integer': charges}

            if len(args.conditioning) > 0:
                context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
                assert_correctly_masked(context, node_mask)
            else:
                context = None

            # transform batch through flow
            nll, _, _ = losses.compute_loss_and_nll(args, eval_model, nodes_dist, x, h,
                                                    node_mask, edge_mask, context)
            # standard nll from forward KL

            nll_epoch += nll.item() * batch_size
            n_samples += batch_size
            if i % args.n_report_steps == 0:
                print(f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                      f"NLL: {nll_epoch / n_samples:.4f}")

    return nll_epoch / n_samples


def save_and_sample_chain(model, args, device, dataset_info, prop_dist,
                          epoch=0, id_from=0, batch_id=''):
    one_hot, charges, x = sample_chain(args=args, device=device, flow=model,
                                       n_tries=1, dataset_info=dataset_info, prop_dist=prop_dist)

    vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/chain/',
                      one_hot, charges, x, dataset_info, id_from, name='chain')

    return one_hot, charges, x


def sample_different_sizes_and_save(model, nodes_dist, args, device, dataset_info, prop_dist,
                                    n_samples=5, epoch=0, batch_size=100, batch_id=''):
    batch_size = min(batch_size, n_samples)
    for counter in range(int(n_samples / batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(args, device, model, prop_dist=prop_dist,
                                                nodesxsample=nodesxsample,
                                                dataset_info=dataset_info)
        print(f"Generated molecule: Positions {x[:-1, :, :]}")
        vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/', one_hot, charges, x, dataset_info,
                          batch_size * counter, name='molecule')


def analyze_and_save(epoch, model_sample, nodes_dist, args, device, dataset_info, prop_dist,
                     n_samples=1000, batch_size=100):
    print(f'Analyzing molecule stability at epoch {epoch}...')

    # device_back = model_sample.device
    # model_sample.change_device(device)

    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    for i in range(int(n_samples / batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, _, x, node_mask = sample(args, device, model_sample, dataset_info, prop_dist,
                                          nodesxsample=nodesxsample)

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity_dict, rdkit_tuple = analyze_stability_for_molecules(molecules, dataset_info)

    wandb.log(validity_dict)
    if rdkit_tuple is not None:
        wandb.log({'epoch':epoch,'Validity': rdkit_tuple[0][0], 'Uniqueness': rdkit_tuple[0][1], 'Novelty': rdkit_tuple[0][2]})
        print('Validity:', rdkit_tuple[0][0], ' Uniqueness:', rdkit_tuple[0][1], ' Novelty:', rdkit_tuple[0][2])

    # model_sample.change_device(device_back)
    # print('Validity:', rdkit_tuple[0][0], ' Uniqueness:', rdkit_tuple[0][1], 'Novelty:', rdkit_tuple[0][2],
    #       'mol_stable:', validity_dict['mol_stable'], 'atm_stable:', validity_dict['atm_stable'])
    print(validity_dict)
    return validity_dict


def save_and_sample_conditional(args, device, model, prop_dist, dataset_info, epoch=0, id_from=0):
    one_hot, charges, x, node_mask = sample_sweep_conditional(args, device, model, dataset_info, prop_dist)

    vis.save_xyz_file(
        'outputs/%s/epoch_%d/conditional/' % (args.exp_name, epoch), one_hot, charges, x, dataset_info,
        id_from, name='conditional', node_mask=node_mask)

    return one_hot, charges, x
