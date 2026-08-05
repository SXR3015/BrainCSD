import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_path', default=r'/data1/sxr/Unified_model/code/FC_SC_gen'
                              , type=str, help='Root directory path of code')
    parser.add_argument(
        '--data_root_path', default=r'/data1/sxr/Unified_model/data', type=str, help='Root directory path of data')
    parser.add_argument(
        '--mode_net', default=r'Foundation', type=str, help='project mode: pretrained classifier, image_generator, or region-specific')
    parser.add_argument(
        '--Connect', default=r'FC', type=str,
        help='Connectivity')
    parser.add_argument(
        '--pretrain', default=False, type=bool, help='path of pretrained classifier weight')
    # parser.add_argument(
    #     '--pretrain_path', default='/data1/sxr/Unified_model/code/Transformer_foundation2'
    #                                '/results/foundation/weights_epoch300'
    #                                '/weights_epoch300.pth', type=str, help='path of pretrained classifier weight')
    parser.add_argument(
        '--pretrain_path', default='', type=str, help='path of pretrained foundation weight')
    parser.add_argument(
        '--pretrain_refine_path', default='/data1/sxr/Unified_model/code/FC_SC_gen/results/refine/weights_epoch500/weights_epoch174.pth',
                                       type=str, help='path of pretrained refined foundation weight')
    parser.add_argument(
        '--pretrain_foundation_path_f', default='/data1/sxr/Unified_model/code/fMRI_foundation/results/foundation/Max_min_add/weights_epoch500'
                                                '/weights_epoch500.pth', type=str, help='path of pretrained fMRI ROI foundation model weight')
    parser.add_argument(
        '--pretrain_foundation_path_d', default='/data1/sxr/Unified_model/code/dMRI_foundation/results/foundation/Max_min_add/weights_epoch1000'
                                   '/weights_epoch139.pth', type=str, help='path of pretrained dti ROI foundation model weight')
    parser.add_argument(
        '--resume_path', default='', type=str, help='path of pretrained connectcome generation weight')
    parser.add_argument(
        '--result_path', default='results', type=str, help='Result directory path')
    parser.add_argument(
        '--structure', default='Max_min_add', type=str, help='the weighted embeddiing structure (None, Max, Min, Max_min,Max_min_sub, Max_min_add, Max_max_mut)')
    parser.add_argument(
        '--event_path', default='events', type=str, help='Result directory path')
    parser.add_argument(
        '--fold_id', default='2', type=str, help='Different data type directory')
    parser.add_argument(
        '--data_type', default='DFC_CLINICAL', type=str, help='FC or JPEG')
    # parser.add_argument(
    # #     '--category', default='', type=str, help='Different data type directory')
    # parser.add_argument(
    #     '--features', default='ALFF_DFC_FA_FC', type=str, help='Different data type directory')
    # parser.add_argument(
    #     '--n_classes', default=4, type=int, help='Number of classes (an: 2, tri: 3)')
    parser.add_argument(
        '--n_fold', default=5, type=int, help='Number of cross validation fold')
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument(
        '--refine', default=True, type=bool, help='Refine the ROI data to connectcome matrix')
    parser.add_argument(
        '--manual_seed', default=1680, type=int, help='Manually set random seed')#1024
    parser.add_argument(
        '--learning_rate', default=1e-4, type=float, help= 'Initial learning rate')#学习率
    parser.add_argument(
        '--learning_rate_refine', default=1e-3, type=float, help= 'Initial learning rate')#学习率
    parser.add_argument(
        '--lr_decay_factor', default=0.02, type=float,
        help=' Factor by which the learning rate will be reduced. new_lr = lr * factor')
    parser.add_argument(
        '--weight_decay', default=1e-5, type=float, help='Weight Decay')
    parser.add_argument(
        '--lr_patience', default=10, type=int, help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument(
        '--batch_size', default=240, type=int, help='Batch Size')
    # parser.add_argument('--temperature', default=0.07, type=float,
    #                     help='softmax temperature (default: 0.07)')
    parser.add_argument('--n_views', default=512, type=int, metavar='N',
                        help='Number of views for contrastive learning training.')
    parser.add_argument(
        '--n_epochs', default=20, type=int, help='Number of total epochs to run')
    # parser.add_argument(
    #     '--n_epochs_pretrain', default=60, type=int, help='Number of total epochs to run')
    # parser.add_argument(
    #     '--n_epochs_pretrain', default=20, type=int, help='Number of total epochs to run')
    parser.add_argument(
        '--save_weight', default=True, type=int, help='wheather save the Trained model or not.')
    parser.add_argument(
        '--no_train', action='store_true', help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument(
        '--no_val', action='store_true', help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument(
        '--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=True)
    parser.add_argument(
        '--drop_rate', default=1e-5, type=int, help='drop rate')
    parser.add_argument(
        '--perceptual_loss', default=False, type=bool, help='drop rate')
    parser.add_argument(
        '--norm_groups', default=48, type=int, help='1')
    parser.add_argument("--mask_path", type=str, default='/home/b23sxr/fmri_dti_synthesis/mask/Dosenbach160_3mm.nii')
    parser.add_argument("--dim_foundation", type=int, default=256)
    parser.add_argument("--depth_foundation", type=int, default=1)
    parser.add_argument("--mlpdim_foundation", type=int, default=512)
    parser.add_argument("--dim_refine", type=int, default=256)
    parser.add_argument("--depth_refine", type=int, default=4)
    parser.add_argument("--mlpdim_refine", type=int, default=512)
    parser.add_argument(
        '--DDP', default=True, type=bool, help='multi-gpu training')
    parser.set_defaults(no_cuda=False)
    args = parser.parse_args()

    return args
