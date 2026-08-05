import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from opts import parse_opts
from model import generate_model
from dataset import TrainSet
import nibabel as nib
from utils import Logger, OsJoin
from models.foundation_f import ViT_dualModal_f
from models.foundation_d import ViT_dualModal_d
from models.Connect_gen_raw import Conn_gen_raw
from torch.nn.functional import interpolate
from train import train_epoch
from train_refine import  train_epoch_refine
# from validation import val_epoch
# from test import test_epoch
import random
import numpy as np
from dataset import TestSet
from tensorboardX import SummaryWriter
# from models import ema
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
# os.environ["CUDA_VISIBLE_DEVICES"]= '3,4'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:6144"
torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
torch.autograd.set_detect_anomaly(True)

def re_mask(atlas_arr, opt):
    atlas_fla = atlas_arr.flatten()
    atlas_re = np.zeros_like(atlas_fla)
    atlas_eq = np.zeros_like(atlas_fla)
    atlas_mask = np.zeros_like(atlas_fla)
    for i in range(1,161,1):
        index = np.where(np.array(atlas_fla) == i)
        range_ = 0.001
        while (len(index)<=8):
            index = np.where((atlas_fla > (i - range_)) & (atlas_fla < (i + range_)))[0]
            # print(len(index))
            range_ +=0.0001
        # print(len(index))
        if opt.refine == False:
            atlas_re[index] = 1
        else:
            atlas_re[index] = i
            atlas_eq[index] = 1
            atlas_mask[index] = 1
        if 1<=i<=5 or 35<=i<=40 or 54<=i<=58 or 86 <=i<= 90 or 119<=i<=123 or 141 <=i<= 145:
            atlas_mask[index] = 0
    # print(atlas_arr.shape)
    atlas_re = atlas_re.reshape(atlas_arr.shape[2], atlas_arr.shape[3], atlas_arr.shape[4])
    atlas_mask = atlas_mask.reshape(atlas_arr.shape[2], atlas_arr.shape[3], atlas_arr.shape[4])
    atlas_eq = atlas_eq.reshape(atlas_arr.shape[2], atlas_arr.shape[3], atlas_arr.shape[4])
    if opt.refine == False:
        return torch.from_numpy(atlas_re).float()
    else:
        return torch.from_numpy(atlas_re).float(), torch.from_numpy(atlas_mask).float(), torch.from_numpy(atlas_eq).float()

def run(fold_id, opt):
    assert  opt.Connect != 'FC' or opt.Connect != 'SC', "Connectcome kind must be FC or SC"
    if opt.root_path != '':
        result_path = OsJoin(opt.root_path, opt.result_path)
        event_path = OsJoin(opt.root_path, opt.event_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
    if opt.refine == True:
        opt.arch ='{}-{}-{}-{}'.format('foundation', opt.depth_refine, opt.dim_refine, opt.mlp_dim_refine)
    else:
        opt.arch = '{}-{}-{}-{}'.format('refine', opt.depth_foundation, opt.dim_foundation, opt.mlpdim_foundation)
    #print(opt)

    print('-'*75, 'RUN ', '-'*75)

    model, parameters = generate_model(opt)

    if torch.cuda.device_count() > 1 and opt.DDP ==True:
        model.to(device)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank,
                                                      find_unused_parameters= True
                                                      )
        training_data = TrainSet(fold_id=fold_id)
        train_loader = DataLoader(training_data, batch_size=opt.batch_size,
                                  num_workers=8, pin_memory=True, sampler=DistributedSampler(training_data))
    else:
        model.cuda()
        training_data = TrainSet(fold_id=fold_id)
        train_loader = DataLoader(training_data, batch_size=opt.batch_size,
                                  num_workers=8, pin_memory=True)

    if opt.refine == True:
        log_path = OsJoin(result_path, 'refine', opt.structure,
                          'logs_epoch%d' % (opt.n_epochs))
        event_path = OsJoin(event_path, 'refine',opt.structure,
                             'logs_epoch%d' % (opt.n_epochs))
    else:
        log_path = OsJoin(result_path, 'foundation',opt.structure,
                          'logs_epoch%d' % (opt.n_epochs))
        event_path = OsJoin(event_path, 'foundation',opt.structure,
                             'logs_epoch%d' % (opt.n_epochs))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if opt.mode_net == 'Foundation':
        train_logger = Logger(
            OsJoin(log_path,'train.log'),
            # ['epoch','loss','acc','lr',])
        ['epoch', 'loss','lr'])

        if opt.refine == True:
            train_batch_logger = Logger(
                OsJoin(log_path, 'train_batch.log'),
                ['epoch','batch','iter','loss_avg', 'loss','lr'])
        else:
            train_batch_logger = Logger(
                OsJoin(log_path, 'train_batch.log'),
                ['epoch', 'batch', 'iter', 'loss_avg', 'loss', 'lr'])
    if opt.refine == True:
        generator_parameters = model.module.generator.parameters()
        discriminator_parameters = model.module.discriminator.parameters()
        # params_G = [
        #     {'params': filter(lambda p: p.requires_grad, generator_parameters['base_parameters']),
        #      'lr': opt.learning_rate * 0.001},
        #     {'params': filter(lambda p: p.requires_grad, generator_parameters['new_parameters']),
        #      'lr': opt.learning_rate}
        # ]
        # params_D = [
        #     {'params': filter(lambda p: p.requires_grad, discriminator_parameters['base_parameters']),
        #      'lr': opt.learning_rate * 0.001},
        #     {'params': filter(lambda p: p.requires_grad, discriminator_parameters['new_parameters']),
        #      'lr': opt.learning_rate}
        # ]
        optimizer_G = torch.optim.Adam(generator_parameters, lr=opt.learning_rate, weight_decay= opt.weight_decay)
        optimizer_D = torch.optim.Adam(discriminator_parameters, lr=opt.learning_rate, weight_decay= opt.weight_decay)

    else:
        optimizer = torch.optim.AdamW(parameters, lr=opt.learning_rate, weight_decay= opt.weight_decay)
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min',
        #                                            factor=opt.lr_decay_factor, patience =opt.lr_patience)
    if opt.refine == True:
        model_pretrain_f = ViT_dualModal_f(dim=opt.dim_foundation, depth=opt.depth_foundation, heads=8,
                                           mlp_dim=opt.mlpdim_foundation)
        model_pretrain_f = model_pretrain_f.cuda()
        model_pretrain_d = ViT_dualModal_d(dim=opt.dim_foundation, depth=opt.depth_foundation, heads=8,
                                           mlp_dim=opt.mlpdim_foundation)
        model_pretrain_d = model_pretrain_d.cuda()
        model_pretrian_conn_gen = Conn_gen_raw(dim=opt.dim_raw, depth=opt.depth_raw,heads=8,mlp_dim=opt.mlpdim_raw)
        model_pretrian_conn_gen = model_pretrian_conn_gen.cuda()
        # model_pretrain_f.eval()
        with torch.no_grad():
            checkpoint = torch.load(opt.pretrain_foundation_path_f,weights_only=True, map_location='cuda:0')['state_dict']
            print('loading pretrained model {}'.format(opt.pretrain_foundation_path_f))
            for key in list(checkpoint.keys()):
                if 'module.' in key:
                    checkpoint[key.replace('module.', '')] = checkpoint[key]
                    del checkpoint[key]
            model_pretrain_f.load_state_dict(checkpoint)
            print('Load fMRI Model successfully')
        with torch.no_grad():
            checkpoint = torch.load(opt.pretrain_foundation_path_d,weights_only=True, map_location='cuda:1')['state_dict']
            print('loading pretrained model {}'.format(opt.pretrain_foundation_path_d))
            for key in list(checkpoint.keys()):
                if 'module.' in key:
                    checkpoint[key.replace('module.', '')] = checkpoint[key]
                    del checkpoint[key]
            model_pretrain_d.load_state_dict(checkpoint)
            print('Load dMRI Model successfully')
        with torch.no_grad():
            checkpoint = torch.load(opt.pretrain_raw_path, weights_only=True, map_location='cuda:2')['state_dict']
            print('loading pretrained model {}'.format(opt.pretrain_raw_path))
            for key in list(checkpoint.keys()):
                if 'module.' in key:
                    checkpoint[key.replace('module.', '')] = checkpoint[key]
                    del checkpoint[key]
            model_pretrian_conn_gen.load_state_dict(checkpoint)
            print('Load raw gen Model successfully')
    writer = SummaryWriter(logdir=event_path)
    scheduler_G = lr_scheduler.StepLR(optimizer_G, step_size=200, gamma=0.1)
    scheduler_D = lr_scheduler.StepLR(optimizer_D, step_size=200, gamma=0.1)
    atlas_arr = nib.load(opt.mask_path).get_fdata()
    affine =  nib.load(opt.mask_path).affine
    atlas = torch.from_numpy(np.array(atlas_arr)).type(torch.FloatTensor)
    atlas_fMRI = interpolate(atlas.unsqueeze(0).unsqueeze(1),
                             [64, 64, 40])
    atlas_dti = interpolate(atlas.unsqueeze(0).unsqueeze(1),
                            [96, 96, 60])
    # if opt.refine == True:
    #     atlas_fMRI, atlas_f_mask, atlas_f_eq = re_mask(atlas_fMRI, opt)
    #     atlas_dti, atlas_d_mask, atlas_d_eq = re_mask(atlas_dti, opt)
    #     save_path = r'/data1/sxr/Unified_model/code/FC_SC_Refined/important_data'
    #     save_atlas_fMRI_path = save_path + '/atlas_fMRI.nii'
    #     # nib.save(gen_path,MRI_gen_total )
    #     nib.Nifti1Image(atlas_fMRI.numpy(),
    #                     affine).to_filename(save_atlas_fMRI_path)
    #     save_atlas_f_mask_path = save_path + '/atlas_f_mask.nii'
    #     # nib.save(gen_path,MRI_gen_total )
    #     nib.Nifti1Image(atlas_f_mask.numpy(),
    #                     affine).to_filename(save_atlas_f_mask_path)
    #     save_atlas_f_eq_path = save_path + '/atlas_f_eq.nii'
    #     # nib.save(gen_path,MRI_gen_total )
    #     nib.Nifti1Image(atlas_f_eq.numpy(),
    #                     affine).to_filename(save_atlas_f_eq_path)
    #     save_atlas_dti_path = save_path + '/atlas_dti.nii'
    #     # nib.save(gen_path,MRI_gen_total )
    #     nib.Nifti1Image(atlas_dti.numpy(),
    #                     affine).to_filename(save_atlas_dti_path)
    #     save_atlas_d_mask_path = save_path + '/atlas_d_mask.nii'
    #     # nib.save(gen_path,MRI_gen_total )
    #     nib.Nifti1Image(atlas_d_mask.numpy(),
    #                     affine).to_filename(save_atlas_d_mask_path)
    #     save_atlas_d_eq_path = save_path + '/atlas_d_eq.nii'
    #     # nib.save(gen_path,MRI_gen_total )
    #     nib.Nifti1Image(atlas_d_eq.numpy(),
    #                     affine).to_filename(save_atlas_d_eq_path)
    #     print('save nii files successfully')
    # else:
    #     atlas_fMRI = re_mask(atlas_fMRI, opt)
    #     atlas_dti = re_mask(atlas_dti, opt)
    atlas_fMRI_raw = nib.load(r'/data1/sxr/Unified_model/code/FC_SC_Refined/important_data/atlas_fMRI.nii').get_fdata()
    atlas_fMRI = torch.from_numpy(np.array(atlas_fMRI_raw)).type(torch.FloatTensor)
    atlas_dti_raw = nib.load(r'/data1/sxr/Unified_model/code/FC_SC_Refined/important_data/atlas_dti.nii').get_fdata()
    atlas_dti = torch.from_numpy(np.array(atlas_dti_raw)).type(torch.FloatTensor)
    atlas_f_mask = nib.load(r'/data1/sxr/Unified_model/code/FC_SC_Refined/important_data/atlas_f_mask.nii').get_fdata()
    atlas_f_mask  = torch.from_numpy(np.array(atlas_f_mask)).type(torch.FloatTensor)
    atlas_d_mask = nib.load(r'/data1/sxr/Unified_model/code/FC_SC_Refined/important_data/atlas_d_mask.nii').get_fdata()
    atlas_d_mask  = torch.from_numpy(np.array(atlas_d_mask)).type(torch.FloatTensor)
    atlas_d_eq = nib.load(r'/data1/sxr/Unified_model/code/FC_SC_Refined/important_data/atlas_d_eq.nii').get_fdata()
    atlas_d_eq  = torch.from_numpy(np.array(atlas_d_eq)).type(torch.FloatTensor)
    atlas_f_eq = nib.load(r'/data1/sxr/Unified_model/code/FC_SC_Refined/important_data/atlas_f_eq.nii').get_fdata()
    atlas_f_eq  = torch.from_numpy(np.array(atlas_f_eq)).type(torch.FloatTensor)
    for i in range(1, opt.n_epochs+1):
        
        torch.cuda.empty_cache()
        # if i < 2:
        #     continue
        if not opt.no_train and opt.refine == False:
            train_epoch(i, fold_id, train_loader, model, opt,
                        train_logger, train_batch_logger, writer,optimizer, atlas_fMRI, atlas_dti)
        elif not opt.no_train and opt.refine == True:
            train_epoch_refine(i, fold_id, train_loader, model, opt,
                        train_logger, train_batch_logger, writer, optimizer_G,optimizer_D, atlas_fMRI, atlas_dti,
                               atlas_f_mask, atlas_d_mask,atlas_f_eq,atlas_d_eq, model_pretrain_f,model_pretrain_d,model_pretrian_conn_gen )
       # if not opt.no_val:
       #     validation_loss = val_epoch(i,val_loader, model, criterion, opt, val_logger, writer,optimizer,\
       #                                 global_step=global_step_,
       #                 noise_scheduler_ =noise_schedul, \
       #                                 scaler_=scaler, lr_scheduler_=lr_scheduler,\
        #                Ema_=ema_, gamma_=gamma_)
        if not opt.no_train and not opt.no_val:
            lr = optimizer_G.param_groups[0]["lr"]
            writer.add_scalar('lr', lr, i)
        # global_step_ = global_step_ +1
        scheduler_G.step()
        scheduler_D.step()
    writer.close()
    # test_data = TestSet()
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size = opt.batch_size, shuffle=False,
    #                                                         num_workers = 0, pin_memory=True)
    # if opt.mode_net == 'pretrained classifier' or opt.mode_net == 'region-specific':
    #     test_logger = Logger(OsJoin(log_path, 'test.log'),
    #                         ['epoch', 'loss', 'acc', 'recall', 'precision', 'f1', 'sensitivity', 'specificity'])
    # elif opt.mode_net == 'Foundation':
    #     test_logger = Logger(OsJoin(log_path, 'test.log'),
    #                         ['epoch', 'loss_G', 'loss_D', 'acc', 'recall', 'precision', 'f1', 'sensitivity',
    #                          'specificity'])
    # test_epoch(1, test_loader, model, writer, fold_id, criterion, opt, test_logger)
    # print('-'*47, 'FOLD %s FINISHED'%str(fold_id), '-'*48)


if __name__ == '__main__':
    opt = parse_opts()
    # 交叉验证
    for fold_id in range(1, opt.n_fold + 1):
        if fold_id > 1:
             break
        run(fold_id, opt)
