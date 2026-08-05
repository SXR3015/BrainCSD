import torch
# from torch.autograd import Variable
import os
import torchvision
import torch.nn as nn
# from dipy.align import affine
from einops import rearrange, repeat
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import nibabel as nib
from diffusers.optimization import get_scheduler
import nibabel as nib
from torch.nn.functional import interpolate, cosine_similarity
# from models.scheduler import DDIMScheduler
# from models.model import unet
import time
from scipy.io import savemat,loadmat

import numpy as np
from utils import OsJoin, vgg
from torch.nn.functional import interpolate
import time
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from utils import AverageMeter,calculate_accuracy,generate_target_label,generate_neurodegeneration


def make_symmetric_lower_to_upper(tensor):
    """
    Make a tensor symmetric by copying the lower triangular part to the upper part.

    Given a tensor of shape (b, n, 160, 160), this function enforces symmetry
    across the main diagonal for each 160x160 matrix: output[i,j] = output[i,j].T.
    The diagonal and lower triangle are preserved; the upper triangle is replaced
    with the transpose of the lower triangle.

    Args:
        tensor (torch.Tensor): Input tensor of shape (b, n, 160, 160)

    Returns:
        torch.Tensor: Symmetric tensor of the same shape, where each 160x160 matrix
                      is symmetric (i.e., A = A^T).
    """
    # Create a lower triangular mask (including the diagonal) of shape (160, 160)
    lower_mask = torch.tril(torch.ones(160, 160, device=tensor.device)).bool()

    # Extract the lower triangular part of the input tensor
    # Values outside the lower triangle are set to zero
    lower_triangle = torch.where(lower_mask, tensor, torch.zeros_like(tensor))

    # Transpose the last two dimensions to get the values that should fill the upper triangle
    # This effectively mirrors the lower triangle across the diagonal
    lower_transposed = lower_triangle.transpose(-1, -2)

    # Create an upper triangular mask (excluding the diagonal) to avoid overwriting it
    upper_mask = torch.triu(torch.ones(160, 160, device=tensor.device), diagonal=1).bool()

    # Construct the symmetric result:
    # - Keep the original lower triangle (including diagonal)
    # - Fill the upper triangle (excluding diagonal) with values from the transposed lower triangle
    result = lower_triangle + torch.where(upper_mask, lower_transposed, 0)

    return result
def create_sparse_false_mask(shape=(160, 160), step=3):
    """
    Create a boolean mask of given shape where every `step`-th row and column is False,
    all other positions are True.

    Args:
        shape (tuple): The shape of the mask (height, width)
        step (int): Interval to set False (every step-th row/column)

    Returns:
        torch.BoolTensor: A mask with shape (H, W), where every step-th row/column is False
    """
    h, w = shape
    mask = torch.ones(h, w, dtype=torch.bool)  # All True initially

    # Set every 'step'-th row to False
    mask[::step, :] = False

    # Set every 'step'-th column to False
    mask[:, ::step] = False

    return mask.cuda()

def train_epoch_refine(epoch, fold_id, data_loader, model, \
                opt, epoch_logger, batch_logger, writer, optimizer_G,optimizer_D, atlas_fMRI, atlas_dti, atlas_f_mask, atlas_d_mask,atlas_f_eq,
                       atlas_d_eq, model_pretrain_f,model_pretrain_d,model_pretrian_conn_gen):
    print('train at epoch {}'.format(epoch))


    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses= AverageMeter()
    # gen_losses = AverageMeter()
    losses_discr = AverageMeter()
    writer = writer
    losses_log = 0
    end_time = time.time()
    writer_index = np.random.randint(1,len(data_loader),size=1)
    group_fc = loadmat(r'/data1/sxr/Unified_model/code/FC_SC_Refined/important_data/group.mat')
    group_mat = torch.from_numpy(np.array(group_fc['fc'])).float()


    # group_embedding = r'/data1/sxr/Unified_model/data/FC'

    for i ,(inputs) in enumerate(data_loader):
        data_time.update(time.time()-end_time)
        target_fMRI = inputs[0].float()
        affine_fMRI = inputs[1].float()
        target_dti = inputs[2].float()
        affine_dti = inputs[3].float()
        conn_tar = make_symmetric_lower_to_upper(inputs[4].float())
        group = make_symmetric_lower_to_upper(inputs[5].float())
        with torch.no_grad():
            # print(target_fMRI.shape)
            gen_fMRI = model_pretrain_f(target_fMRI.unsqueeze(1).cuda())
            gen_dti = model_pretrain_d(target_dti.unsqueeze(1).cuda())
            sc, fc, re_d, re_f = model_pretrian_conn_gen(gen_fMRI*atlas_f_mask.contiguous().unsqueeze(0).cuda(),
                                                         gen_dti*atlas_d_mask.contiguous().unsqueeze(0).cuda(), atlas_fMRI.contiguous().cuda() , atlas_dti.contiguous().cuda())
            # print(atlas_f_mask.unsqueeze(0).shape)

        if 'FC' == opt.Connect:
            group_mat_expand = group_mat.expand(target_dti.shape[0], -1, -1).unsqueeze(1).cuda()
            # fc = torch.clamp(fc,-1,1)
            noise = torch.randn((fc.shape[-1],fc.shape[-2])).cuda()
            # mask = create_sparse_false_mask()
            # noisy_fc =  noise + fc
            noisy_fc = fc
            '''
            1. directly input fc not work
            2. add guassian noise not work,try the setting and architecture of counterfactual learning, dim 32 not work
            3. mask not work
            4. Unet not work
            5. directly transformer not work
            6 multiply guassian noise not work
            7. patch size 32 not work
            8. fully noise not work
            9. 3D noise not work, try 2D noise, not work
            10. more parameter, mlp_dim 512, depth 16, not work
            11. perceptual loss needed
            12. group site mean fc not work
            '''
            # group.unsqueeze(1).cuda()
            # noisy_fc = torch.clamp(noisy_fc, -1, 1.)
            # print(noisy_fc.squeeze().unsqueeze(1).shape)
            gen_conn, loss_gen, loss_discr = model(noisy_fc.squeeze().unsqueeze(1).cuda(),conn_tar.unsqueeze(1).cuda().float(), group.unsqueeze(1).cuda())
            # gen_conn = torch.clamp(gen_conn,-1,1)
            # refine_conn = torch.clamp(refine_conn, -1, 1.)
            # conn_tar = torch.clamp(conn_tar, -1, 1.)
        else:
            # sc = torch.clamp(sc, -1, 1)
            noise = torch.randn_like(sc).cuda()
            noisy_sc = sc + noise
            # noisy_sc = torch.clamp(noisy_sc, -1, 1.)
            gen_conn, loss_gen, loss_discr   = model(noisy_sc.squeeze().unsqueeze(1))
            # refine_conn = torch.clamp(refine_conn, -1, 1.)
            # conn_tar = torch.clamp(conn_tar, -1, 1.)
            # print(index_roi)
        # re_loss = (F.mse_loss(refine_conn,conn_tar.unsqueeze(1).cuda() ,reduction='mean') +
        #               +bce_discr_loss(refine_conn,conn_tar.unsqueeze(1).cuda()))
            # handle grayscale for vgg

        # loss = loss_s + re_loss
        # loss =  re_loss + (loss_s/160)
        # loss =  re_loss
        # optimizer_D.zero_grad()
        # loss_discr.backward(retain_graph=True)
        # # if epoch % 3 == 0:
        # optimizer_D.step()
        # optimizer_G.zero_grad()
        # loss_gen.backward()
        # optimizer_G.step()
        if epoch%2==0:
            if i % 2 == 0:
                optimizer_D.zero_grad()
                loss_discr.backward(retain_graph=True)
                # if epoch % 3 == 0:
                optimizer_D.step()
                # optimizer_D.zero_grad()

            else:
                optimizer_G.zero_grad()
                loss_gen.backward(retain_graph=True)
                optimizer_G.step()


        else:
            if i % 2 == 1:
                optimizer_D.zero_grad()
                loss_discr.backward(retain_graph=True)
                # if epoch % 3 == 0:
                optimizer_D.step()
                # optimizer_D.zero_grad()

            else:
                optimizer_G.zero_grad()
                loss_gen.backward(retain_graph=True)
                optimizer_G.step()
        if i == 1 and epoch % 1 == 0:
                index = np.random.randint(0,  target_fMRI.shape[0], size=1)
                save_path = os.path.join(opt.root_path, 'Synthesis', opt.Connect)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                if 'SC' == opt.Connect:
                    gen_path =save_path + '/sc_raw_epoch%d_iter%d.mat' % (
                        epoch, i)
                    # nib.Nifti1Image(((sc[index] + 1) * 0.5).squeeze().detach().cpu().numpy(),
                    #                 affine_fMRI[index] .squeeze()).to_filename(refine_path)
                    savemat(gen_path, {'sc_raw': sc[index].squeeze().detach().cpu().numpy()})
                    noise_path = save_path + '/sc_noise_epoch%d_iter%d.mat' % (
                        epoch, i)
                    # nib.Nifti1Image(((sc[index] + 1) * 0.5).squeeze().detach().cpu().numpy(),
                    #                 affine_fMRI[index] .squeeze()).to_filename(refine_path)
                    savemat(noise_path, {'sc_noise': noisy_sc[index].squeeze().detach().cpu().numpy()})
                    refine_path =save_path + '/sc_refine_epoch%d_iter%d.mat' % (
                        epoch, i)
                    # nib.Nifti1Image(((sc[index] + 1) * 0.5).squeeze().detach().cpu().numpy(),
                    #                 affine_fMRI[index] .squeeze()).to_filename(refine_path)
                    savemat(refine_path, {'sc_refine': gen_conn[index].squeeze().detach().cpu().numpy()})
                    tar_path = save_path + '/sc_tar_epoch%d_iter%d.mat' % (
                        epoch, i)
                    # nib.Nifti1Image(((sc[index] + 1) * 0.5).squeeze().detach().cpu().numpy(),
                    #                 affine_fMRI[index] .squeeze()).to_filename(refine_path)
                    savemat(tar_path, {'sc_tar': conn_tar[index].squeeze().detach().cpu().numpy()})
                else:
                    gen_path = save_path + '/fc_raw_epoch%d_iter%d.mat' % (
                        epoch, i)
                    # nib.Nifti1Image(((sc[index] + 1) * 0.5).squeeze().detach().cpu().numpy(),
                    #                 affine_fMRI[index] .squeeze()).to_filename(refine_path)
                    savemat(gen_path, {'fc_raw': fc[index].squeeze().detach().cpu().numpy()})
                    noise_path = save_path + '/fc_noise_epoch%d_iter%d.mat' % (
                        epoch, i)
                    # nib.Nifti1Image(((sc[index] + 1) * 0.5).squeeze().detach().cpu().numpy(),
                    #                 affine_fMRI[index] .squeeze()).to_filename(refine_path)
                    savemat(noise_path, {'fc_noise': noisy_fc[index].squeeze().detach().cpu().numpy()})
                    refine_path = save_path + '/fc_refine_epoch%d_iter%d.mat' % (
                        epoch, i)
                    # nib.Nifti1Image(((sc[index] + 1) * 0.5).squeeze().detach().cpu().numpy(),
                    #                 affine_fMRI[index] .squeeze()).to_filename(refine_path)
                    savemat(refine_path, {'fc_refine': gen_conn[index].squeeze().detach().cpu().numpy()})
                    tar_path = save_path + '/fc_tar_epoch%d_iter%d.mat' % (
                        epoch, i)
                    # nib.Nifti1Image(((sc[index] + 1) * 0.5).squeeze().detach().cpu().numpy(),
                    #                 affine_fMRI[index] .squeeze()).to_filename(refine_path)
                    savemat(tar_path, {'fc_tar': conn_tar[index].squeeze().detach().cpu().numpy()})
                gen_d_path = save_path + '/gen_d_epoch%d_iter%d.nii' % (
                    epoch, i)
                # nib.save(gen_path,MRI_gen_total )
                nib.Nifti1Image((gen_dti[index].squeeze().detach().cpu().numpy()),
                                affine_dti[index] .squeeze()).to_filename(gen_d_path)
                mask_d_path = save_path + '/mask_d_epoch%d_iter%d.nii' % (
                    epoch, i)
                # nib.save(gen_path,MRI_gen_total )
                nib.Nifti1Image((gen_dti[index].squeeze().detach().cpu().numpy()*atlas_d_mask.detach().cpu().numpy()),
                                affine_dti[index].squeeze()).to_filename(mask_d_path)
                re_d_path = save_path + '/re_d_epoch%d_iter%d.nii' % (
                    epoch, i)
                # nib.save(gen_path,MRI_gen_total )
                nib.Nifti1Image((re_d[index].squeeze().detach().cpu().numpy()),
                                affine_dti[index].squeeze()).to_filename(re_d_path)
                tar_d_path = save_path + '/tar_d_epoch%d_iter%d.nii' % (
                    epoch, i)
                nib.Nifti1Image((target_dti[index].squeeze().numpy() * atlas_d_eq.numpy()),
                                affine_dti[index].squeeze()).to_filename(tar_d_path)
                gen_f_path = save_path + '/gen_f_epoch%d_iter%d.nii' % (
                    epoch, i)
                # nib.save(gen_path,MRI_gen_total )
                nib.Nifti1Image((gen_fMRI[index].squeeze().detach().cpu().numpy()),
                                affine_fMRI[index] .squeeze()).to_filename(gen_f_path)
                tar_f_path = save_path + '/tar_f_epoch%d_iter%d.nii' % (
                    epoch, i)
                nib.Nifti1Image((target_fMRI[index].squeeze().numpy() * atlas_f_eq.numpy()),
                                affine_fMRI[index].squeeze()).to_filename(tar_f_path)
                mask_f_path = save_path + '/mask_f_epoch%d_iter%d.nii' % (
                    epoch, i)
                # nib.save(gen_path,MRI_gen_total )
                nib.Nifti1Image((gen_fMRI[index].squeeze().detach().cpu().numpy()*atlas_f_mask.detach().cpu().numpy()),
                                affine_fMRI[index].squeeze()).to_filename(mask_f_path)
                re_f_path = save_path + '/re_f_epoch%d_iter%d.nii' % (
                    epoch, i)
                # nib.save(gen_path,MRI_gen_total )
                nib.Nifti1Image((re_f[index].squeeze().detach().cpu().numpy()),
                                affine_fMRI[index].squeeze()).to_filename(re_f_path)
        losses_log += loss_gen.detach().item()
        # if opt.refine == True:
        #     checkpoint = 1
        #     save_step = 200
        # else:
        #     checkpoint = 1
        #     save_step = 200
        # if opt.save_weight:
        #     if epoch % checkpoint == 0 and i % save_step == 0:
        #         # if opt.pretrain ==True:
        #         #     epoch_save = epoch +3
        #         if opt.refine == True:
        #             save_dir = os.path.join(opt.result_path, opt.result_path, 'refine',
        #                                     'weights_epoch' + str(opt.n_epochs))
        #         else:
        #             save_dir = os.path.join(opt.root_path, opt.result_path, 'foundation',
        #                                     'weights_epoch' + str(opt.n_epochs))
        #         if not os.path.exists(save_dir):
        #             os.makedirs(save_dir)
        #         save_path = OsJoin(save_dir,
        #                            'weights_epoch{}_step{}.pth'.format(epoch, i))
        #
        #         save_path_old = OsJoin(save_dir,
        #                                'weights_epoch{}_step{}.pth'.format(epoch - 1, i))
        #         if os.path.exists(save_path_old):
        #             try:
        #                 os.remove(save_path_old)
        #             except:
        #                 print('File has deleted')
        #         states = {
        #             'fold': fold_id,
        #             'epoch': epoch,
        #             'arch': opt.arch,
        #             'state_dict': model.state_dict(),
        #             'optimizer_G': optimizer_G.state_dict(),
        #             'optimizer_D': optimizer_D.state_dict(),
        #
        #         }
        #         torch.save(states, save_path)
            # acc = 1
        losses.update(loss_gen.data,inputs[0].size(0))
        losses_discr.update(loss_discr.data, inputs[0].size(0))
        # sim_losses.update((loss_s/160).data, inputs[0].size(0))

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i - 1),
            "loss_avg": losses_log / (i + 1),
            "loss": loss_gen.detach().item(),
            "lr": optimizer_G.param_groups[0]['lr']
        })

        print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_discr {discr_loss.val:.4f} ({discr_loss.avg:.4f})\t'
                  # 'Loss_sim {loss_sim.val:.4f} ({loss_sim.avg:.4f})\t'
            .format(
                epoch, i + 1, len(data_loader), batch_time=batch_time,\
                data_time=data_time, loss=losses, discr_loss=losses_discr))
        if i % writer_index == 0:
            writer.add_scalar('train/loss', losses_log / (i + 1), i + (epoch - 1) * len(data_loader))
            writer.add_scalar('train/lr', loss_gen.detach().item(), i + (epoch - 1) * len(data_loader))
        batch_time.update(time.time()-end_time)
        end_time = time.time()
    try:
        epoch_logger.log({
            'epoch': epoch,
            'loss': round(losses.avg.item(), 4),
            'lr': optimizer_G.param_groups[0]['lr']
        })
    except:
            epoch_logger.log({
                'epoch': epoch,
                'loss': round(losses.avg, 4),
                'lr': optimizer_G.param_groups[0]['lr']
            })


    # if opt.mode_net == "pretrained classifier" or opt.mode_net == 'region-specific':
    #     checkpoint = 20
    # elif opt.mode_net == 'image_generator':
    #     checkpoint = 1
    #     save_steps = 10
    if opt.save_weight:
        if epoch % 1 == 0 :
            if opt.refine == True:
                save_dir = OsJoin(opt.result_path, 'refine',
                                   'weights_epoch'+str(opt.n_epochs)
                                  )
            else:
                save_dir = OsJoin(opt.result_path, 'foundation',
                                   'weights_epoch'+str(opt.n_epochs))

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # if opt.pretrain ==True:
            #     epoch_save = epoch +3
            save_path = OsJoin(save_dir,
                               'weights_epoch{}.pth'.format(epoch))
            save_path_old = OsJoin(save_dir,
                                   'weights_epoch{}.pth'.format(epoch - 1))
            if os.path.exists(save_path_old):
                try:
                    os.remove(save_path_old)
                except:
                    print('File has deleted')
            states = {
                'fold': fold_id,
                'epoch': epoch,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
            }
            torch.save(states, save_path)

