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
from torch.nn.functional import interpolate, cosine_similarity
# from models.scheduler import DDIMScheduler
# from models.model import unet
import time
from models.foundation import ViT_dualModal
import numpy as np
from utils import OsJoin, vgg
from torch.nn.functional import interpolate
import time
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from utils import AverageMeter,calculate_accuracy,generate_target_label,generate_neurodegeneration

def train_epoch_refine(epoch, fold_id, data_loader, model, \
                opt, epoch_logger, batch_logger, writer,optimizer, atlas_fMRI, atlas_dti,atlas_f_mask, atlas_d_mask):
    print('train at epoch {}'.format(epoch))


    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses= AverageMeter()
    writer = writer
    losses_log = 0
    end_time = time.time()
    writer_index = np.random.randint(1,len(data_loader),size=1)
    index_train = np.random.randint(1,len(data_loader),size=10)
    model_pretrain = ViT_dualModal(dim=opt.dim_foundation, depth=opt.depth_foundation,heads=8,mlp_dim=opt.mlpdim_foundation)
    model_pretrain = model_pretrain.cuda()
    # model_pretrain.eval()
    with torch.no_grad():
        checkpoint = torch.load(opt.pretrain_foundation_path, map_location='cuda:0')['state_dict']
        print('loading pretrained model {}'.format(opt.pretrain_diffusion_path))
        for key in list(checkpoint.keys()):
            if 'module.' in key:
                checkpoint[key.replace('module.', '')] = checkpoint[key]
                del checkpoint[key]
        model_pretrain.load_state_dict(checkpoint)
    for i ,(inputs) in enumerate(data_loader):
        data_time.update(time.time()-end_time)
        target_fMRI = inputs[0]
        target_dti = inputs[2]
        with torch.no_grad():
            gen_fMRI, gen_dti = model_pretrain (target_fMRI, target_dti)
        sc, fc = model (gen_fMRI * atlas_f_mask.unsqueeze(0).cuda(), gen_dti * atlas_d_mask.unsqueeze(0).cuda())
        loss = 0
        for region in range(1,161,1):
            fc_region = fc[...,region]
            sc_region = sc[...,region]
            mask_region_f = (atlas_fMRI == region/160).float()
            mask_region_d = (atlas_dti == region / 160).float()
            image_region_f = target_fMRI * mask_region_f
            image_region_d = target_dti * mask_region_d
            simalirity_img = cosine_similarity(image_region_f,image_region_d )
            simalirity_connect = cosine_similarity(fc_region,sc_region)
            loss_sim = (simalirity_connect-simalirity_img).mean()
            loss += loss_sim
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i in index_train and epoch % 1 == 0:
                index = np.random.randint(0,  target_fMRI.shape[0], size=1)
                save_path = os.path.join(opt.root_path, 'Synthesis')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                refine_path =opt.root_path + '/f_refine_epoch%d_iter%d.nii' % (
                    epoch, i)

                # nib.save(gen_path,MRI_gen_total )
                # nib.Nifti1Image(((gen_images['gen_fmri'][index] + 1) * 0.5).squeeze().detach().cpu().numpy(),
                #                 affine_fMRI[index] .squeeze()).to_filename(refine_path)
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
            .format(
                epoch, i + 1, len(data_loader), batch_time=batch_time,\
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

