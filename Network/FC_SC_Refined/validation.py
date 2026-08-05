import torch
# from torch.autograd import Variable
import os
import torch.nn.functional as F
import nibabel as nib
from diffusers.optimization import get_scheduler
# import io
# from PIL import Image
import numpy as np
from utils import OsJoin
import time
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
# from models.my_model  import DM_MRI
from utils import AverageMeter,calculate_accuracy,generate_target_label,generate_neurodegeneration
from models.scheduler import DDIMScheduler
from models import ema
n_inference_timesteps = 250
def plt_image(array):
    figure = plt.figure()
    plt.imshow(array)
    plt.axis('off')

    plt.close(figure)
    return figure
def normalize(array):
    max_ = np.max(np.max(array))
    min_ = np.min(np.min(array))
    return (array-max_)/(max_-min_)
def val_epoch(epoch,data_loader,model,criterion,opt,logger, writer,optimizer,\
              noise_scheduler_f2d, noise_scheduler_d2f, \
              global_step, scaler_, lr_scheduler_,Ema_, gamma_):
    print('validation at epoch {}'.format(epoch) )
    model.train()
    if opt.mode_net == 'image_generator':
       model.train()

    batch_time =AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_discr = AverageMeter()
    accuracies = AverageMeter()
    recalls = AverageMeter()
    precisions = AverageMeter()
    f1s = AverageMeter()
    sensitivitys = AverageMeter()
    specificitys = AverageMeter()
    writer = writer
    # MRI_gen_total =torch.zeros((1,160,160))
    # gen_target_total = torch.zeros((1, 160, 160))
    if opt.n_classes == 3:
        labels_total = torch.zeros((1,3))
    elif opt.n_classes == 2:
        labels_total = torch.zeros((1,2))
    end_time = time.time()
    writer_index = np.random.randint(1,len(data_loader),size=1)
    losses_log = 0
    index_gen = np.random.randint(0,len(data_loader),size=1)
    for i, (inputs,labels) in enumerate(data_loader):
       # if i > 10:
        #     break
        
        data_time.update(time.time() - end_time)
        # labels = list(map(int,labels))
        # inputs= (torch.unsqueeze(input,1) for input in inputs)
        target_fMRI = inputs[0]
        noise_fMRI = inputs[1]
        affine_fMRI = inputs[2]
        target_dti = inputs[3]
        noise_dti = inputs[4]
        affine_dti = inputs[5]
        # labels = labels.type(torch.FloatTensor)
        # target_MRI = [target_dti, target_fMRI]
        # inputs_noise = [noise_fMRI, noise_dti]
        # affine = [affine_dti, affine_fMRI]
        # inputs_target = inputs[1].type(torch.FloatTensor).unsqueeze(1)
        # inputs = [inputs_noise, inputs_target]
        labels = labels.type(torch.FloatTensor)
        target_MRI = [target_dti, target_fMRI ]
        inputs_noise = [noise_fMRI,noise_dti]
        affine = [affine_dti,affine_fMRI]
        if opt.mode_net == 'pretrained classifier' or opt.mode_net == 'region-specific':
            loss, outputs = model([inputs,labels])
        elif opt.mode_net == 'image_generator' :
            noise_fMRI = torch.randn(target_fMRI.shape)
            noise_dti = torch.randn(target_dti.shape)
            timesteps = torch.randint(0,
                                      noise_scheduler_f2d.num_train_timesteps,
                                      (opt.batch_size,)).long()
            noisy_images_fMRI =noise_scheduler_f2d.add_noise(target_fMRI, noise_fMRI,
                                                     timesteps)
            noisy_images_dti = noise_scheduler_d2f.add_noise(target_dti, noise_dti,
                                                     timesteps)

            optimizer.zero_grad()
            with autocast(enabled=opt.fp16_precision):
                noise_pred_dti = model(noisy_images_dti.type(torch.FloatTensor).unsqueeze(1).cuda(), \
                                        timesteps)["sample"]
                # noise_pred_fMRI = model(noisy_images_fMRI.type(torch.FloatTensor).unsqueeze(1).cuda(), \
                #                         timesteps)["sample"]
                loss = F.l1_loss(noise_pred_dti, noise_dti.cuda().unsqueeze(1))
                # loss = F.l1_loss(noise_pred_fMRI, noise_fMRI.cuda().unsqueeze(1)) + \
                #             F.l1_loss(noise_pred_dti, noise_dti.cuda().unsqueeze(1))

            scaler_.scale(loss).backward()
            scaler_.step(optimizer)
            scaler_.update()
            Ema_.update_params(gamma_)
            # Ema_d2f.update_params(gamma_d2f)
            gamma_ = Ema_.update_gamma(global_step)
            # gamma_d2f = Ema_d2f.update_gamma(global_step)
            lr_scheduler_.step()
            losses_log += loss.detach().item()
            losses.update(loss.data, inputs[0].size(0))
            if i ==index_gen and epoch %10 ==0:
                model.eval()
                with torch.no_grad():
                    generator = torch.manual_seed(0)
                    generated_images_d2f = noise_scheduler_d2f.generate(
                        model,
                        num_inference_steps=n_inference_timesteps,
                        generator=generator,
                        eta=1.0,
                        batch_size=opt.batch_size,
                        mode= 'd2f')
                    # generated_images_f2d = noise_scheduler_f2d.generate(
                    #     Ema_.ema_model,
                    #     num_inference_steps=n_inference_timesteps,
                    #     generator=generator,
                    #     eta=1.0,
                    #     batch_size=opt.batch_size,
                    #     mode= 'f2d')
                    try:

                        index_d2f = np.random.randint(0,generated_images_d2f["sample"].shape[0], size=1)
                        # index_f2d = np.random.randint(0, generated_images_f2d["sample"].shape[0], size=1)
                        max_fmri = np.max(np.max(np.max(target_fMRI[index_d2f].squeeze().detach().cpu().numpy())))
                        # max_dti = np.max(np.max(np.max(target_dti[index_f2d].squeeze().detach().cpu().numpy())))
                        save_path = r'/home/b23sxr/fmri_dti_synthesis/code/network_generation_distribution2/Synthesis'
                        if not os.path.exists(save_path):
                          os.makedirs(save_path)
                        gen_path = r'/home/b23sxr/fmri_dti_synthesis/code/network_generation_distribution2/Synthesis'+'/gen_epoch%d_iter%d_d2f.nii'%(epoch,i)
                    # nib.save(gen_path,MRI_gen_total )
                        nib.Nifti1Image(generated_images_d2f['sample'][index_d2f].squeeze()*max_fmri, affine_dti[index_d2f].squeeze()).to_filename(gen_path)
                        tar_path = r'/home/b23sxr/fmri_dti_synthesis/code/network_generation_distribution2/Synthesis'+'/tar_epoch%d_iter%d_d2f.nii'%(epoch,i)
                        nib.Nifti1Image(target_dti[index_d2f].squeeze().detach().cpu().numpy(), affine_dti[index_d2f].squeeze()).to_filename(tar_path)
                        gen_path = r'/home/b23sxr/fmri_dti_synthesis/code/network_generation_distribution2/Synthesis'+'/gen_epoch%d_iter%d_f2d.nii'%(epoch,i)
                    # nib.save(gen_path,MRI_gen_total )
                    #     nib.Nifti1Image(generated_images_f2d['sample'][index_f2d].squeeze()*max_dti, affine_dti[index_f2d].squeeze()).to_filename(gen_path)
                    #     tar_path = r'/home/b23sxr/fmri_dti_synthesis/code/network_generation_distribution2/Synthesis'+'/tar_epoch%d_iter%d_f2d.nii'%(epoch,i)
                    #     nib.Nifti1Image(target_dti[index_f2d].squeeze().detach().cpu().numpy(), affine_dti[index_f2d].squeeze()).to_filename(tar_path)
                    except:
                                 print(index_d2f )
                                 print(generated_images_d2f['sample'].shape)
                                 continue
                   



        # elif opt.mode_net == 'text-image generator':
#        aucs.update(auc, inputs[0][0].size(0))
        batch_time.update(time.time() - end_time)
        end_time = time.time()
# need to check the data,the data of loss is the format of tensor array

        if  opt.mode_net == 'image_generator':
            logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i - 1),
                "loss_avg": losses_log / (i + 1),
                "loss": loss.detach().item(),
                "lr": lr_scheduler_.get_last_lr()[0],
                "step": global_step,
                "gamma": gamma_
            })
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i + 1, len(data_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            if i % writer_index == 0:
                writer.add_scalar('train/loss', losses_log / (i + 1), i + (epoch - 1) * len(data_loader))
                writer.add_scalar('train/lr', loss.detach().item(), i + (epoch - 1) * len(data_loader))
    if opt.mode_net == 'image_generator':
        logger.log({'epoch': epoch, 'loss': losses_log / (i + 1), 'lr': lr_scheduler_.get_last_lr()[0],
                    } )
    return losses
