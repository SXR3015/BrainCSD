import torch
# from torch.autograd import Variable
import os
import torchvision
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import nibabel as nib
from diffusers.optimization import get_scheduler
import nibabel as nib
from torch.nn.functional import interpolate
# import io
# from PIL import Image
import numpy as np
from utils import OsJoin, vgg

import time
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
# from models.my_model  import DM_MRI
from utils import AverageMeter,calculate_accuracy,generate_target_label,generate_neurodegeneration
# from models.scheduler import DDIMScheduler
# from models import ema

'''
best performance at 83 epoch
'''
def log(t, eps=1e-10):
    return torch.log(t + eps)

def bce_discr_loss(fake, real):
    return (-log(1 - torch.sigmoid(fake)) - log(torch.sigmoid(real))).mean()

def train_epoch(epoch, fold_id, data_loader, model,
                opt, epoch_logger, batch_logger, writer,optimizer, atlas_fMRI, atlas_dti):
    print('train at epoch {}'.format(epoch))
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses= AverageMeter()
    writer = writer
    losses_log = 0
    end_time = time.time()
    writer_index = np.random.randint(1,len(data_loader),size=1)
    index_train = np.random.randint(1,len(data_loader),size=1)

    # print(atlas.shape)
    '''
    network-aware start at epoch 46
    '''

    for i ,(inputs) in enumerate(data_loader):
        # if 8825<i <8840 or 3620< i< 3640 or 160<i<170 or 320<i<340:
        #     continue
        # if i < 390:
        #     continue
        torch.cuda.empty_cache()
        data_time.update(time.time()-end_time)
        target_dti = inputs[2]
        affine_dti = inputs[3]

        if opt.mode_net == 'Foundation' :
                   re_dti= model(target_dti.unsqueeze(1).cuda())
                   # print(re_dti.shape, re_fMRI.shape)
                   # print((target_fMRI.unsqueeze(1)*atlas_fMRI.unsqueeze(0)).cuda().shape)
                   # loss = (F.l1_loss(re_fMRI,(target_fMRI.unsqueeze(1)*atlas_fMRI.unsqueeze(0)).cuda())
                   #            + F.l1_loss(re_dti,(target_dti.unsqueeze(1)*atlas_dti.unsqueeze(0)).cuda()))
                   loss =(
                          F.mse_loss(re_dti,(target_dti.unsqueeze(1)*atlas_dti.unsqueeze(0)).cuda()) +
                            bce_discr_loss(re_dti,(target_dti.unsqueeze(1)*atlas_dti.unsqueeze(0)).cuda())
                          # + F.mse_loss(re_dti*atlas_dti.unsqueeze(0).cuda(),(target_dti.unsqueeze(1)*atlas_dti.unsqueeze(0)).cuda())
                          #    + bce_discr_loss(re_dti*atlas_dti.unsqueeze(0).cuda(),(target_dti.unsqueeze(1)*atlas_dti.unsqueeze(0)).cuda())
                                )
                   # loss = F.l1_loss(re_fMRI*atlas_fMRI.unsqueeze(0).cuda(),(target_fMRI.unsqueeze(1)*atlas_fMRI.unsqueeze(0)).cuda())
        if i ==1 and epoch % 1 == 0:
                    index = np.random.randint(0, re_dti.shape[0], size=1)
                    save_path = opt.root_path +'/Synthesis/'+opt.structure
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    gen_path = save_path + '/train_dMRI_epoch%d_iter%d.nii' % (
                    epoch, i)
                    # print(affine_fMRI[index].shape)
                    nib.Nifti1Image(re_dti[index].detach().squeeze().cpu().numpy() ,
                                    affine_dti[index].detach().squeeze().cpu().numpy()).to_filename(gen_path)
                    # gen_path = save_path + '/train_dmri_epoch%d_iter%d.nii' % (
                    # epoch, i)
                    # nib.Nifti1Image(re_dti[index].detach().squeeze().cpu().numpy() ,
                    #                 affine_dti[index].detach().squeeze().cpu().numpy()).to_filename(gen_path)
                    tar_path = save_path + '/train_dMRI_tar_epoch%d_iter%d.nii' % (epoch, i)
                    nib.Nifti1Image((target_dti.unsqueeze(1)*atlas_dti.unsqueeze(0))[index].squeeze().cpu().numpy(),
                        affine_dti[index].detach().squeeze().cpu().numpy()).to_filename(tar_path)
                    # tar_path = save_path + '/train_dmri_tar_epoch%d_iter%d.nii' % (epoch, i)
                    # nib.Nifti1Image((target_dti.unsqueeze(1)*atlas_dti.unsqueeze(0))[index].squeeze().cpu().numpy(),
                    #                 affine_dti[index].detach().squeeze().cpu().numpy()).to_filename(tar_path)
                    source_path = save_path + '/train_dMRI_source_epoch%d_iter%d.nii' % (epoch, i)
                    nib.Nifti1Image((target_dti.unsqueeze(1))[index].squeeze().cpu().numpy(),
                                    affine_dti[index].detach().squeeze().cpu().numpy()).to_filename(source_path)
                    # source_path = save_path + '/train_dmri_source_epoch%d_iter%d.nii' % (epoch, i)
                    # nib.Nifti1Image((target_dti.unsqueeze(1))[index].squeeze().cpu().numpy(),
                    #                 affine_dti[index].detach().squeeze().cpu().numpy()).to_filename(source_path)
        '''
        easy to lead to OOM and nan loss
        '''

        # if i  == 1:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if torch.isnan(loss):
            print('nan in loss at step {}'.format(i))
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses_log += loss.detach().item()
        if opt.refine == True:
            checkpoint = 1
            save_step = 50
        else:
            checkpoint = 1
            save_step = 50
        if opt.save_weight:
            if epoch % checkpoint == 0 and i % save_step == 0:
                # if opt.pretrain ==True:
                #     epoch_save = epoch +3
                if opt.refine == True:
                    save_dir = os.path.join(opt.result_path, opt.result_path, 'refine', opt.structure, 'weights_epoch'+str(opt.n_epochs))
                else:
                    save_dir = os.path.join(opt.root_path, opt.result_path, 'foundation', opt.structure, 'weights_epoch'+str(opt.n_epochs))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = OsJoin(save_dir,
                                   'weights_epoch{}_step{}.pth'.format(epoch,i))

                save_path_old = OsJoin(save_dir,
                                       'weights_epoch{}_step{}.pth'.format(epoch-1, i))
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
        batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i - 1),
                "loss_avg": losses_log / (i + 1),
                "loss": loss.detach().item(),
                 "lr": optimizer.param_groups[0]['lr'],
            })
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        .format(
            epoch, i + 1, len(data_loader), batch_time=batch_time,
            data_time=data_time, loss=losses))
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

    checkpoint = 1
    if opt.save_weight:
        if epoch % checkpoint == 0 :
            if opt.refine == True:
                save_dir = OsJoin(opt.result_path, 'refine', opt.structure,
                                   'weights_epoch'+str(opt.n_epochs)
                                  )
            else:
                save_dir = OsJoin(opt.result_path, 'foundation', opt.structure,
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

