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
os.environ["CUDA_VISIBLE_DEVICES"]= '5,6,7'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:6144"
torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
torch.autograd.set_detect_anomaly(True)

def re_mask(atlas_arr, opt):
    atlas_fla = atlas_arr.flatten()
    atlas_re = np.zeros_like(atlas_fla)
    atlas_mask = np.zeros_like(atlas_fla)
    for i in range(1,161,1):
        index = np.where(np.array(atlas_fla) == i)
        range_ = 0.001
        while (len(index)<=8):
            index = np.where((atlas_fla > (i - range_)) & (atlas_fla < (i + range_)))[0]
            # print(len(index))
            range_ +=0.0001
        if opt.refine == False:
            atlas_re[index] = 1
        else:
            atlas_re[index] = i
        if 1<=i<=5 or 35<=i<=40 or 54<=i<=58 or 86 <=i<= 90 or 119<=i<=123 or 141 <=i<= 145:
            atlas_mask[index] = 1
    # print(atlas_arr.shape)
    atlas_re = atlas_re.reshape(atlas_arr.shape[2], atlas_arr.shape[3], atlas_arr.shape[4])
    atlas_mask = atlas_mask.reshape(atlas_arr.shape[2], atlas_arr.shape[3], atlas_arr.shape[4])
    if opt.refine == False:
        return torch.from_numpy(atlas_re).float()
    else:
        return torch.from_numpy(atlas_re).float(), torch.from_numpy(atlas_mask).float()

def run(fold_id, opt):
    if opt.root_path != '':
        result_path = OsJoin(opt.root_path, opt.result_path)
        event_path = OsJoin(opt.root_path, opt.event_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
    if opt.refine == True:
        opt.arch ='{}-{}-{}-{}'.format('foundation', opt.depth_refine, opt.dim_refine, opt.mlpdim_refine)
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
    if opt.mode_net == 'pretrained classifier' or opt.mode_net == 'region-specific':
        parameters = model.parameters()
    elif opt.mode_net == 'Foundation':
        try:

            parameters =model.module.parameters()
            # print('load module ')
        except:
            parameters = model.parameters()


    if opt.refine == True:
        optimizer = torch.optim.AdamW(parameters, lr=opt.learning_rate_refine, weight_decay= opt.weight_decay)
    else:
        optimizer = torch.optim.AdamW(parameters, lr=opt.learning_rate, weight_decay= opt.weight_decay)
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min',
        #                                            factor=opt.lr_decay_factor, patience =opt.lr_patience)
    writer = SummaryWriter(logdir=event_path)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    atlas_arr = nib.load(opt.mask_path).get_fdata()
    atlas = torch.from_numpy(np.array(atlas_arr)).type(torch.FloatTensor)
    atlas_fMRI = interpolate(atlas.unsqueeze(0).unsqueeze(1),
                             [64, 64, 40])
    atlas_dti = interpolate(atlas.unsqueeze(0).unsqueeze(1),
                            [96, 96, 60])
    if opt.refine == True:
        atlas_fMRI, atlas_f_mask = re_mask(atlas_fMRI, opt)
        atlas_dti, atlas_d_mask= re_mask(atlas_dti, opt)
    else:
        atlas_fMRI = re_mask(atlas_fMRI, opt)
        atlas_dti = re_mask(atlas_dti, opt)
    for i in range(1, opt.n_epochs+1):
        
        torch.cuda.empty_cache()
        # if i < 2:
        #     continue
        if not opt.no_train and opt.refine == False:
            train_epoch(i, fold_id, train_loader, model, opt,
                        train_logger, train_batch_logger, writer,optimizer, atlas_fMRI, atlas_dti)
        elif not opt.no_train and opt.refine == True:
            train_epoch_refine(i, fold_id, train_loader, model, opt,
                        train_logger, train_batch_logger, writer, optimizer, atlas_fMRI, atlas_dti, atlas_f_mask, atlas_d_mask)
       # if not opt.no_val:
       #     validation_loss = val_epoch(i,val_loader, model, criterion, opt, val_logger, writer,optimizer,\
       #                                 global_step=global_step_,
       #                 noise_scheduler_ =noise_schedul, \
       #                                 scaler_=scaler, lr_scheduler_=lr_scheduler,\
        #                Ema_=ema_, gamma_=gamma_)
        if not opt.no_train and not opt.no_val:
            lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar('lr', lr, i)
        # global_step_ = global_step_ +1
        scheduler.step()
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
