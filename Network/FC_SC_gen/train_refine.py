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
from scipy.io import savemat

import numpy as np
from utils import OsJoin, vgg
from torch.nn.functional import interpolate
import time
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from utils import AverageMeter,calculate_accuracy,generate_target_label,generate_neurodegeneration

def log(t, eps=1e-10):
    return torch.log(t + eps)

def bce_discr_loss(fake, real):
    return (-log(1 - torch.sigmoid(fake)) - log(torch.sigmoid(real))).mean()


def nt_xent_loss(z_i, z_j, temperature=0.05):
    # Normalize the projections
    z_i = F.normalize(z_i, dim=-1)
    z_j = F.normalize(z_j, dim=-1)

    # Concatenate the two views
    representations = torch.cat([z_i, z_j], dim=0)  # Shape: (2*B, D)

    # Compute similarity matrix
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                            dim=-1)  # Shape: (2*B, 2*B)

    # Positive pairs are on the diagonal of each half of the similarity matrix
    sim_ij = torch.diag(similarity_matrix, z_i.size(0))  # Shape: (B,)
    sim_ji = torch.diag(similarity_matrix, -z_i.size(0))  # Shape: (B,)
    positives = torch.cat([sim_ij, sim_ji], dim=0)  # Shape: (2*B,)

    # Compute the logits and apply temperature scaling
    logits = positives / temperature

    # Negative pairs by excluding the diagonal
    negatives = similarity_matrix[~torch.eye(2 * z_i.size(0), dtype=bool)].view(2 * z_i.size(0),
                                                                                -1)  # Shape: (2*B, 2*B-1)

    # Combine positives and negatives to form the final logits
    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)  # Shape: (2*B, 2*B)

    # Labels are all zeros since positives are placed at index 0
    labels = torch.zeros(logits.size(0)).long().to(z_i.device)  # Shape: (2*B,)

    # Cross-entropy loss
    loss = F.cross_entropy(logits, labels)

    return loss

def train_epoch_refine(epoch, fold_id, data_loader, model, \
                opt, epoch_logger, batch_logger, writer,optimizer, atlas_fMRI, atlas_dti,atlas_f_mask, atlas_d_mask,atlas_f_eq,atlas_d_eq,model_pretrain_f,model_pretrain_d):
    print('train at epoch {}'.format(epoch))


    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses= AverageMeter()
    re_losses = AverageMeter()
    sim_losses = AverageMeter()
    writer = writer
    losses_log = 0
    end_time = time.time()
    writer_index = np.random.randint(1,len(data_loader),size=1)
    index_train = np.random.randint(1,len(data_loader),size=10)

    for i ,(inputs) in enumerate(data_loader):
        data_time.update(time.time()-end_time)
        target_fMRI = inputs[0]
        affine_fMRI = inputs[1]
        target_dti = inputs[2]
        affine_dti = inputs[3]
        with torch.no_grad():
            # print(target_fMRI.shape)
            gen_fMRI = model_pretrain_f(target_fMRI.unsqueeze(1).cuda())
            gen_dti = model_pretrain_d(target_dti.unsqueeze(1).cuda())
            # print(atlas_f_mask.unsqueeze(0).shape)
        sc, fc, re_d, re_f = model (gen_fMRI*atlas_f_mask.unsqueeze(0).cuda(), gen_dti*atlas_d_mask.unsqueeze(0).cuda(), atlas_fMRI.cuda() , atlas_dti.cuda()  )
        loss_s = 0
        for region in range(1,161,1):
            # fc_region = fc[...,region-1]
            # # print(fc_region.shape)
            # sc_region = sc[...,region-1]
            # # print(sc_region.shape)
            # mask_region_f = (atlas_fMRI == region/160).float()
            # mask_region_d = (atlas_dti == region / 160).float()
            # image_region_f = (target_fMRI * mask_region_f).view(target_fMRI.shape[0],-1)
            # image_region_d = (target_dti * mask_region_d).view(target_dti.shape[0],-1)
            # # print(image_region_f.shape)
            # simalirity_img = cosine_similarity(image_region_f.mean(dim=(-1)),image_region_d.mean(dim=(-1)), dim=-1)
            # simalirity_connect = cosine_similarity(fc_region,sc_region,dim=-1)
            # loss_sim = (simalirity_connect-simalirity_img).mean()
            # loss_s += loss_sim

            fc_region = fc[...,region-1]
            # print(fc_region[0,...])
            sc_region = sc[...,region-1]
            # region_indices = (mask_flatten == cur_region).float()
            # nonzero_idx = region_indices.bool()
            # region_values_compressed = arr[:, nonzero_idx]
            # print(sc_region.shape)
            mask_region_f = (atlas_fMRI.view(-1) == region).cuda()
            # print(mask_region_f)
            mask_region_d = (atlas_dti.view(-1)  == region).cuda()
            image_region_f = target_fMRI.view(target_fMRI.shape[0],-1).cuda()
            image_region_d = target_dti.view(target_dti.shape[0],-1).cuda()
            index_roi= 0
            if mask_region_f.numel() > 0:  # 检查是否有元素
                region_values_compressed_f = image_region_f[:, mask_region_f]
                # print(region_values_compressed_f.shape)
                index_roi += region_values_compressed_f.shape[1]
            else:
                index_f = torch.where(mask_region_f == region)
                # print('test')
                range_ = 0.001
                while (len(index_f) <= 8):
                    index_f = torch.where((mask_region_f > (region - range_)) & (mask_region_f < (region + range_)))[0]
                    # print(index_f)
                    # index_roi_1 = index_f
                    range_ += 0.0001
                region_values_compressed_f = image_region_f[:, index_f]  # 返回一个 dummy 张量
                index_roi += len(index_f)

            if mask_region_d.numel() > 0:  # 检查是否有元素
                region_values_compressed_d = image_region_d[:, mask_region_d]
                index_roi += region_values_compressed_d.shape[1]
                # print(region_values_compressed_d)
            else:
                index_d = torch.where(mask_region_d == region)
                range_ = 0.001
                while (len(index_d) <= 8):
                    index_d = torch.where((mask_region_d > (region - range_)) & (mask_region_d < (region + range_)))[0]
                    range_ += 0.0001
                region_values_compressed_d = image_region_d[:, index_d]  # 返回一个 dummy 张量
                index_roi += len(index_d)

            # region_values_compressed_d = image_region_d[:, mask_region_d]
            #
            # print(region_values_compressed_f)
            # print(index_roi)
            if region_values_compressed_f.shape[1] < region_values_compressed_d.shape[1]:
                  simalirity_img = cosine_similarity(region_values_compressed_f, region_values_compressed_d[:,0:region_values_compressed_f.shape[1] ], dim=-1)
            else:
                simalirity_img = cosine_similarity(region_values_compressed_f[:, 0:region_values_compressed_d.shape[1]],
                                                   region_values_compressed_d,
                                                   dim=-1)
            # print(simalirity_img)
            simalirity_connect = cosine_similarity(fc_region, sc_region, dim=-1)
            # print(simalirity_connect)
            loss_sim = nt_xent_loss(simalirity_img.unsqueeze(1).cuda(), simalirity_connect.unsqueeze(1).cuda())
            # loss_s += (loss_sim /((len(index_f)+len(index_d))/2))
            loss_s += (loss_sim / (( index_roi*1)))
            # print(index_roi)
        re_loss = (F.smooth_l1_loss(re_f,target_fMRI.unsqueeze(1).cuda() * atlas_f_eq.unsqueeze(0).cuda()*10,reduction='mean') +
                      F.smooth_l1_loss(re_d,target_dti.unsqueeze(1).cuda() * atlas_d_eq.unsqueeze(0).cuda()*50,reduction='mean')
                   +bce_discr_loss(re_f,target_fMRI.unsqueeze(1).cuda() * atlas_f_eq.unsqueeze(0).cuda()*10)+
                     bce_discr_loss(re_d,target_dti.unsqueeze(1).cuda() * atlas_d_eq.unsqueeze(0).cuda()*50))
        # loss = loss_s + re_loss
        loss =  re_loss + (loss_s/160)
        # loss =  re_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i == 1 and epoch % 1 == 0:
                index = np.random.randint(0,  target_fMRI.shape[0], size=1)
                save_path = os.path.join(opt.root_path, 'Synthesis')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                refine_path =save_path + '/sc_refine_epoch%d_iter%d.mat' % (
                    epoch, i)
                # nib.Nifti1Image(((sc[index] + 1) * 0.5).squeeze().detach().cpu().numpy(),
                #                 affine_fMRI[index] .squeeze()).to_filename(refine_path)
                savemat(refine_path, {'sc': sc[index].squeeze().detach().cpu().numpy()})
                refine_path = save_path + '/fc_refine_epoch%d_iter%d.mat' % (
                    epoch, i)
                # nib.Nifti1Image(((sc[index] + 1) * 0.5).squeeze().detach().cpu().numpy(),
                #                 affine_fMRI[index] .squeeze()).to_filename(refine_path)
                savemat(refine_path, {'fc': fc[index].squeeze().detach().cpu().numpy()})
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
        losses_log += loss.detach().item()
        if opt.refine == True:
            checkpoint = 1
            save_step = 200
        else:
            checkpoint = 1
            save_step = 200
        if opt.save_weight:
            if epoch % checkpoint == 0 and i % save_step == 0:
                # if opt.pretrain ==True:
                #     epoch_save = epoch +3
                if opt.refine == True:
                    save_dir = os.path.join(opt.result_path, opt.result_path, 'refine',
                                            'weights_epoch' + str(opt.n_epochs))
                else:
                    save_dir = os.path.join(opt.root_path, opt.result_path, 'foundation',
                                            'weights_epoch' + str(opt.n_epochs))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = OsJoin(save_dir,
                                   'weights_epoch{}_step{}.pth'.format(epoch, i))

                save_path_old = OsJoin(save_dir,
                                       'weights_epoch{}_step{}.pth'.format(epoch - 1, i))
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
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(states, save_path)
            # acc = 1
        losses.update(loss.data,inputs[0].size(0))
        re_losses.update(re_loss.data, inputs[0].size(0))
        sim_losses.update((loss_s/160).data, inputs[0].size(0))

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i - 1),
            "loss_avg": losses_log / (i + 1),
            "loss": loss.detach().item(),
            "lr": optimizer.param_groups[0]['lr']
        })

        print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_re {re_loss.val:.4f} ({re_loss.avg:.4f})\t'
                  'Loss_sim {loss_sim.val:.4f} ({loss_sim.avg:.4f})\t'
            .format(
                epoch, i + 1, len(data_loader), batch_time=batch_time,\
                data_time=data_time, loss=losses, re_loss=re_losses, loss_sim=sim_losses))
        if i % writer_index == 0:
            writer.add_scalar('train/loss', losses_log / (i + 1), i + (epoch - 1) * len(data_loader))
            writer.add_scalar('train/lr', loss.detach().item(), i + (epoch - 1) * len(data_loader))
        batch_time.update(time.time()-end_time)
        end_time = time.time()
    try:
        epoch_logger.log({
            'epoch': epoch,
            'loss': round(losses.avg.item(), 4),
            'lr': optimizer.param_groups[0]['lr']
        })
    except:
            epoch_logger.log({
                'epoch': epoch,
                'loss': round(losses.avg, 4),
                'lr': optimizer.param_groups[0]['lr']
            })


    if opt.mode_net == "pretrained classifier" or opt.mode_net == 'region-specific':
        checkpoint = 20
    elif opt.mode_net == 'image_generator':
        checkpoint = 1
        save_steps = 10
    if opt.save_weight:
        if epoch % checkpoint == 0 :
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
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_path)

