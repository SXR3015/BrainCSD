import csv
import numpy as np
import os
from math import ceil
from utils import OsJoin
from sklearn.model_selection import KFold
from opts import parse_opts
import pandas as pd
import fnmatch
opt = parse_opts()
root = opt.data_root_path

#health_dir = opt.data_type + '_healthy'
#MCI_dir = opt.data_type + '_MCI'
csv_save_dir = OsJoin('csv/', opt.Connect)
test_ratio = 0.0001
n_fold = 300
FC_path = r'/data1/sxr/Unified_model/data/FC'
SC_path = r'/data1/sxr/Unified_model/data/SC'
Subs=list()
least_subs_name=r'/data1/sxr/Unified_model/data/Diffusion'
# def search_session(sub, files_list):
#     index = list()
#     for i in range(len(files_list)):
#         if sub in files_list[i]:
#             index.append(i)
#         else:
#             continue
#     return index
# def search_near_session(sub_fea, session_index, session_names):
#     Current_ses = sub_fea.split('_d')[1]
#     if '_diffusion' in sub_fea:
#         ses_cur_number = Current_ses
#     if '_T1' in sub_fea:
#         ses_cur_number = Current_ses.split('_T1')[0]
#     if '_rsfMRI' in sub_fea:
#         ses_cur_number = Current_ses.split('_rsfMRI')[0]
#     index = session_index[0]
#     gap = 1000
#     for i in range(len(session_index)):
#         session_name_total = session_names[session_index[i]]
#         session_name_num = session_name_total .split('_UDSb4_d')[1]
#         gap_tmp = abs(int(session_name_num) - int(ses_cur_number))
#         if gap_tmp < gap:
#             gap = gap_tmp
#             index = session_index[i]
#     return index





    # files_list.index(sub)
# for filename in os.listdir(least_subs_name):
    # if '_diffusion' in filename:
    #     sub = filename.split('_diffusion')[0]
    #     if 'Hospital' in sub:
    #         sub_new = sub.split('Hospital_')[1]
    #     if 'ADNI' in sub:
    #         sub_new = sub.split('ADNI_')[1]
    #     if 'OASIS' in sub:
    #         sub_new_OA = sub.split('OASIS-3_')[1]
    #         # sub_new = sub_new_OA.split('_')[0]
    #         sub_new = sub_new_OA
    #     Subs.append(sub_new)
    # else:
    #   sub = filename.split('zfc_Covswra_')[1].split('_rsfMRI_timeseries_Dosenbach164.mat')[0]
      # sub_name_delPRE = sub.split('_PRE')[0]
      # Subs.append(filename)
# sub_info_excel_ADNI = pd.read_csv(r'/home/b23sxr/fmri_dti_synthesis/code/network_generation_distribution2_scan/Label_information/ADNI.csv', \
#                                    usecols=[1, 2])
# sub_info_excel_OASIS = pd.read_csv(r'/home/b23sxr/fmri_dti_synthesis/code/network_generation_distribution2_scan/Label_information/OASIS3.csv', \
#                                    usecols=[0,1,4,18])
# data_health = []

# sub_num=len(Subs)
# data_health=np.empty([sub_num, fea_i])
# data_MCI=np.empty([sub_num, fea_i])
# data_SCD=np.empty([sub_num, fea_i])
# sub_n=0
# print(Subs)
Subs_had_FC = list()
Subs_had_SC = list()
subs_rsfmri = list()
subs_diff = list()
if 'FC' in opt.Connect:
    fc_files = os.listdir(FC_path)
    for fc_file in fc_files:
        fc_file_path = os.path.join(FC_path, fc_file)
        Subs_had_FC.append(fc_file_path)
    Subs_connect =  Subs_had_FC
    for sub in Subs_had_FC:
        sub_rsfmri = sub.replace('_FC.mat', '_rsfMRI.nii')
        sub_rsfmri = sub_rsfmri.replace('/FC/', '/Function/')
        subs_rsfmri.append(sub_rsfmri)
        subs_dff = sub.replace('_FC.mat', '_diffusion.nii')
        subs_dff = subs_dff.replace('/FC/', '/Diffusion/')
        subs_diff.append(subs_dff)
if 'SC' in opt.Connect:
    sc_files = os.listdir(SC_path)
    for sc_file in sc_files:
        sc_file_path = os.path.join(SC_path, sc_file)
        Subs_had_SC.append(sc_file_path)
    Subs_connect = Subs_had_SC
    for sub in Subs_had_SC:
        sub_rsfmri = sub.replace('_SC.mat', '_rsfMRI.nii')
        sub_rsfmri = sub_rsfmri.replace('/SC/', '/Function/')
        subs_rsfmri.append(sub_rsfmri)
        subs_dff = sub.replace('_SC.mat', '_diffusion.nii')
        subs_dff = subs_dff.replace('/SC/', '/Diffusion/')
        subs_diff.append(subs_dff)
    # print(subs_rsfmri)
# subs_had = list()


# if csv_contain == True:
#     fea_num = fea_num-1
np.random.seed(opt.manual_seed)
if len(Subs_connect) > 0:
    # data = np.array(Subs).reshape(-1, 2)
    # print(np.array(Subs).unsqueeze(1).shape)
    # health_list = np.concatenate((data_health, np.array(label_health).reshape(int(HC_num / fea_num), 1)), axis=1)
    data_list = np.vstack((np.array(subs_rsfmri), np.array(subs_diff), np.array(Subs_connect))).T
    # print(data_list.shape)
    np.random.shuffle(data_list)
    n_test = ceil(data_list.shape[0] * test_ratio)
    n_train_val = data_list.shape[0] - n_test
    train_val_list =  data_list[0:n_train_val, :]
    test_list = data_list[n_train_val: data_list.shape[0], :]
    # print(train_val_list.shape)
if 'SC' in opt.Connect:
    split = 250
else:
    split = 300
kf = KFold(n_splits=split, shuffle=False)
n = 0
names = locals()
print(train_val_list.shape)

if len(data_list) > 0:
    for train_index, val_index in kf.split(train_val_list):
        n += 1
        names['train_fold%s'%n] = train_val_list[train_index]
        names['val_fold%s' % n] = train_val_list[val_index]

names2 = locals()
for i in range(1, split+1):
    # names2['train_list_fold%s'%i] = np.vstack((names2.get('train_fold%s'%i)))
    # names2['val_list_fold%s'%i] = np.vstack((names2.get('val_fold%s'%i)))
    # names2['train_list_fold%s'%i] = names.get('train_fold%s'%i)
    # names2['val_list_fold%s'%i] = names.get('val_fold%s'%i)
    # test_list = np.vstack((test_list))
    np.random.seed(opt.manual_seed)
    np.random.shuffle(names2['train_fold%s'%i])
    np.random.shuffle(names2['val_fold%s'%i])

   # 按行堆叠
np.random.seed(opt.manual_seed)
np.random.shuffle(test_list)

csv_save_path = OsJoin(root, csv_save_dir)
if not os.path.exists(csv_save_path):
    os.makedirs(csv_save_path)

for i in range(1, split+1):
    with open(OsJoin(csv_save_path, 'train_fold%s.csv'%i), 'w', newline='') as f:  # 设置文件对象
        f_csv = csv.writer(f)
        f_csv.writerows(names2.get('train_fold%s'%i))
    with open(OsJoin(csv_save_path, 'val_fold%s.csv'%i), 'w', newline='') as f:  # 设置文件对象
        f_csv = csv.writer(f)
        f_csv.writerows(names2.get('val_fold%s'%i))


with open(OsJoin(csv_save_path, 'test.csv'), 'w', newline='') as f:  # 设置文件对象
    f_csv = csv.writer(f)
    f_csv.writerows(test_list)