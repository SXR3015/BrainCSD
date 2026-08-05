from torch.autograd import Variable
import time
from utils import OsJoin
from utils import AverageMeter, calculate_accuracy, calculate_recall,generate_target_label,generate_neurodegeneration
import torch
from torch.utils.data import DataLoader
from opts import parse_opts
import numpy as np
from dataset import ValidSet
from model import generate_model
from dataset import TestSet
from utils import Logger
from torch import nn
import os
from models.model import unet_refine
import nibabel as nib
from models.scheduler import DDIMScheduler
from opts import parse_opts
from models.model import unet
import matplotlib.pyplot as plt
opt = parse_opts()
n_timesteps = 1000
n_inference_timesteps = 250
def normalize(array):
    max_ = np.max(np.max(array))
    min_ = np.min(np.min(array))
    return (array-max_)/(max_-min_)
def plt_image(array):
    figure = plt.figure()
    plt.imshow(array)
    plt.axis('off')
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    # image = Image.open(buf)
    plt.close(figure)
    return figure

def test_epoch(test_loader, model):
    noise_scheduler_ = DDIMScheduler(num_train_timesteps=n_timesteps,
                                    beta_schedule="cosine")
    model.eval()
    affine_dti = np.load('/data1/sxr/fmri_dti_synthesis/code/network_loss/mean_dti.npy')
    affine_fmri = np.load('/data1/sxr/fmri_dti_synthesis/code/network_loss/mean_fmri.npy')
    with torch.no_grad():
        for i, (inputs,labels) in enumerate(test_loader):
            target_fMRI = inputs[0]
            noise_fMRI = inputs[1]
            # affine_fMRI = inputs[2]
            target_dti = inputs[3]
            noise_dti = inputs[4]
            # affine_dti = inputs[5]
            # generated_images_f2d = noise_scheduler_f2d.generate(
            #     Ema_.ema_model,
            #     num_inference_steps=n_inference_timesteps,
            #     generator=generator,
            #     eta=1.0,
            #     batch_size=opt.batch_size,
            #     mode= 'f2d')
            noise_pred = model(target_fMRI.type(torch.FloatTensor).unsqueeze(1).cuda(), \
                                    target_dti.type(torch.FloatTensor).unsqueeze(1).cuda()

                               )
            index_d2f = np.random.randint(0, noise_pred["gen_fmri"].shape[0], size=1)
            index_f2d = np.random.randint(0, noise_pred["gen_dti"].shape[0], size=1)
            save_path = (r'/data1/sxr/fmri_dti_synthesis/code/U-net/'
                         r'/Synthesis_gen')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            gen_path = (r'/data1/sxr/fmri_dti_synthesis/code/U-net/'
                        r'/Synthesis_gen') + '/gen_sample%d_d2f.nii' % (i)
            # nib.save(gen_path,MRI_gen_total )
            # print(affine_fmri.squeeze().shape)
            nib.Nifti1Image(((noise_pred['gen_fmri'] + 1)* 0.5).cpu().numpy().squeeze(), affine_fmri.squeeze()).to_filename(gen_path)
            tar_path = r'/data1/sxr/fmri_dti_synthesis/code/U-net/Synthesis_gen/' + '/tar_iter%d_d2f.nii' % ( i)
            nib.Nifti1Image((target_fMRI.squeeze().detach().cpu().numpy() + 1) * 0.5,
                            affine_fmri.squeeze()).to_filename(tar_path)
            gen_path = (r'/data1/sxr/fmri_dti_synthesis/code/U-net/'
                        r'/Synthesis_gen') + '/gen_sample%d_f2d.nii' % (
                 i)
            # nib.save(gen_path,MRI_gen_total )
            nib.Nifti1Image(((noise_pred['gen_dti']+1)* 0.5).cpu().numpy().squeeze(), affine_dti.squeeze()).to_filename(gen_path)
            tar_path = r'/data1/sxr/fmri_dti_synthesis/code/U-net/Synthesis_gen/' + 'tar_iter%d_f2d.nii' % (
            i)
            nib.Nifti1Image((target_dti.squeeze().detach().cpu().numpy() + 1) * 0.5,
                            affine_dti.squeeze()).to_filename(tar_path)
            print('generate sample %d' %i)

opt = parse_opts()
model = unet_refine.UNet(1, hidden_dims=[64, 128, 256, 512],
                  use_flash_attn=True)
checkpoint = torch.load(opt.resume_path, map_location='cuda:0')['state_dict']
print('loading pretrained model {}'.format(opt.resume_path))
for key in list(checkpoint.keys()):
    if 'module.' in key:
        checkpoint[key.replace('module.', '')] = checkpoint[key]
        del checkpoint[key]
# opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
# assert opt.arch == checkpoint['arch']
# model.load_state_dict(checkpoint['state_dict'])
model.load_state_dict(checkpoint)
test_data = ValidSet(fold_id=1)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle=False,
                                                        num_workers = 0, pin_memory=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
print('Load Model successfully')
test_epoch(test_loader, model)

