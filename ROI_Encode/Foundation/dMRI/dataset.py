import os.path

import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import csv
import os
import scipy.io as sio
from opts import parse_opts
from utils import OsJoin
from PIL import Image
import codecs
import pickle
opt = parse_opts()
data_type = opt.data_type
if opt.refine == False:
  csv_dir = OsJoin(opt.data_root_path, 'csv', opt.mode_net)
else:
    csv_dir = OsJoin(opt.data_root_path, 'csv', opt.Connect)

def nii_loader(path):
    # path = path.replace('/home/b23sxr', '/data1/sxr')
    try :
        img_pil = nib.load(path)
    except:
        path.replace('nii.gz','nii')
        img_pil = nib.load(path)
    affine = img_pil.affine
    img_arr = np.array(img_pil.get_fdata())

    if len(img_arr.shape) > 3:
        img_arr = np.sum(img_arr, axis=3)
    # img_arr = np.array(img_pil.get_fdata()) change the function get_data() to get_fdata()
    img_arr_cleaned = np.nan_to_num(img_arr)  # Replace NaN with zero and infinity with large finite numbers.
    max_ = np.max(np.max(np.max(img_arr_cleaned)))
    min_ = np.min(np.min(np.min(img_arr_cleaned)))
    range_ = max_ - min_
    img_arr_cleaned = (img_arr_cleaned-min_) /range_
    img_arr_cleaned = img_arr_cleaned
    # img_pil = torch.from_numpy(img_arr_cleaned).dtype(torch.float)
    img_pil = torch.from_numpy(img_arr_cleaned).float()
    return img_pil,affine

def mat_loader(path):
    # path = path.replace('/home/b23sxr', '/data1/sxr')
    if 'FC' in path:
       mat_data = sio.loadmat(path)['FC']
    else:
        mat_data = sio.loadmat(path)['SC']
    # img_arr = np.array(img_pil.get_fdata()) change the function get_data() to get_fdata()
    img_arr_cleaned = np.nan_to_num(mat_data)  # Replace NaN with zero and infinity with large finite numbers.
    max_ = np.max(np.max(img_arr_cleaned))
    min_ = np.min(np.min(img_arr_cleaned))
    range_ = max_ - min_
    img_arr_cleaned = (img_arr_cleaned-min_) /range_
    img_arr_cleaned = img_arr_cleaned*2-1
    # noise = np.random.normal(0,1,img_arr.shape)
    # noise_img = img_arr_cleaned + noise
    # if path.split('/')[-1] == 's20090904_03_ZangYF_LSF_LiuZhuo-0003-00001-000128-01.nii' or 's20090526_09_ZangYF_LSF_ShuiCaiKong-0003-00001-000128-01.nii':
    #     img_arr_cleaned.resize((256,256,128))   # resize bad samples
    img_pil = torch.from_numpy(img_arr_cleaned)
    # max_ = torch.max(torch.max(torch.max(img_pil )))
    # img_pil = img_pil / max_
    return img_pil

class TrainSet(Dataset):

    def __init__(self, fold_id, nii_loader=nii_loader):
        with open(csv_dir + '/train_fold%s.csv'%str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            #for col in reader:
            '''
            need to complete, the structure is stubborn,
            need to re-open file
            '''
            file_train_fmri = ([OsJoin(opt.data_root_path, row[0]) for row in reader])
        with open(csv_dir + '/train_fold%s.csv' % str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            file_train_diffusion= [row[1] for row in reader]
        self.image = np.array([np.flip(file_train_fmri), np.flip(file_train_diffusion)])
        if opt.refine == True:
            with open(csv_dir + '/train_fold%s.csv' % str(fold_id), 'r') as csvfile:#t1 file is row 2. ignore
                reader = csv.reader(csvfile)
                label_train_connect = [row[2] for row in reader]
            self.image = np.array([np.flip(file_train_fmri), np.flip(file_train_diffusion),
                                   np.flip(label_train_connect)])
        # self.image_fmri = file_train_fmri
        self.nii_loader = nii_loader
        self.mat_loader = mat_loader
    def __getitem__(self, index):
        fn = self.image.transpose()[index]
        img_arr=[]
        for file in fn:
            if "diffusion" in file:
              diff_img, affine_diff = self.nii_loader(file)
              img_arr.append(diff_img)
              img_arr.append(affine_diff)

            elif "fMRI" in file:
              fmri_img, fmri_affine = self.nii_loader(file)
              img_arr.append(fmri_img)
              img_arr.append(fmri_affine)
            else:
                Connect = self.mat_loader(file)
                img_arr.append(Connect)
        return img_arr

    def __len__(self):
        # return len(self.image)
        return len(self.image.transpose())

class ValidSet(Dataset):
    def __init__(self, fold_id, nii_loader=nii_loader):
        with open(csv_dir + '/val_fold%s.csv' % str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            # for col in reader:
            '''
            need to complete, the structure is stubborn,
            need to re-open file
            '''
            file_val_fmri = ([OsJoin(opt.data_root_path, row[0]) for row in reader])
        with open(csv_dir + '/val_fold%s.csv' % str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            file_val_diffusion = [row[1] for row in reader]
        self.image = np.array([np.flip(file_val_fmri), np.flip(file_val_diffusion)])
        if opt.refine == True:
            with open(csv_dir + '/val_fold%s.csv' % str(fold_id), 'r') as csvfile:
                reader = csv.reader(csvfile)
                file_val_connect = [row[2] for row in reader]
            self.image = np.array([np.flip(file_val_fmri), np.flip(file_val_diffusion)
                                   , np.flip(file_val_connect)])
        # self.image_diff = file_val_diffusion
        # self.image_fmri = file_val_fmri

        self.nii_loader = nii_loader
        self.mat_loader = mat_loader

    def __getitem__(self, index):
        fn = self.image.transpose()[index]
        img_arr = []
        for file in fn:
            if "diffusion" in file:
              diff_img, affine_diff = self.nii_loader(file)
              img_arr.append(diff_img)
              img_arr.append(affine_diff)
            elif "fMRI" in file :
              fmri_img, fmri_affine = self.nii_loader(file)
              img_arr.append(fmri_img)
              img_arr.append(fmri_affine)
            else:
                Connect = self.mat_loader(file)
                img_arr.append(Connect)
        return img_arr

    def __len__(self):
        # return len(self.image)
        return len(self.image.transpose())##data_set

class TestSet(Dataset):

    def __init__(self, fold_id, nii_loader=nii_loader):
        with open(csv_dir + '/test.csv' % str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            # for col in reader:
            '''
            need to complete, the structure is stubborn,
            need to re-open file
            '''
            file_test_fmri = ([OsJoin(opt.data_root_path, row[0]) for row in reader])
        with open(csv_dir + '/test.csv' % str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            file_test_diffusion = [row[1] for row in reader]
        with open(csv_dir + '/test.csv' % str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            label_test_1 = [row[3] for row in reader]
        with open(csv_dir + '/test.csv' % str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            label_test_2 = [row[4] for row in reader]
        label_test = np.array([label_test_1, label_test_2])
        if opt.n_classes == 3:
            with open(csv_dir + '/test.csv' % str(fold_id), 'r') as csvfile:
                reader = csv.reader(csvfile)
                label_test_3 = [row[5] for row in reader]
                label_test = np.array([label_test_1, label_test_2, label_test_3])
        if opt.n_classes == 4:
            with open(csv_dir + '/test.csv' % str(fold_id), 'r') as csvfile:
                reader = csv.reader(csvfile)
                label_test_3 = [row[5] for row in reader]
            with open(csv_dir + '/test.csv' % str(fold_id), 'r') as csvfile:
                reader = csv.reader(csvfile)
                label_test_4 = [row[6] for row in reader]
                label_test = np.array([label_test_1, label_test_2, label_test_3, label_test_4])
        # self.image_diff = file_test_diffusion
        # self.image_fmri = file_test_fmri
        self.image = np.array([file_test_fmri, file_test_diffusion])
        self.label = label_test
        self.nii_loader = nii_loader

    def __getitem__(self, index):
        fn = self.image.transpose()[index]
        img_arr = []
        for file in fn:
            if "diffusion" in file:
              diff_img, noise_img_diff, affine_diff = self.nii_loader(file)
              img_arr.append(diff_img)
              img_arr.append(noise_img_diff)
              img_arr.append(affine_diff)
            elif "fmri" in file :
              fmri_img, noise_img_fmri, fmri_affine = self.nii_loader(file)
              img_arr.append(fmri_img)
              img_arr.append(noise_img_fmri)
              img_arr.append(fmri_affine)
        label = self.label.transpose()[index]
        label = np.array([int(label_str) for label_str in label])
        return img_arr, label

    def __len__(self):
        # return len(self.image)
        return len(self.image.transpose())
