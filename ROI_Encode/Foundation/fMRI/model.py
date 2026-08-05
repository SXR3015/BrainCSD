import torch
from torch import nn
from models.foundation import ViT_dualModal
from models.Connect_gen import Conn_gen
# from models import resnet, pre_resnet, wide_resnet, densenet, simpleCNN
# from models import my_model
def generate_model(opt):
    # assert opt.mode in ['score', 'feature']

    if opt.refine == True:
        model = Conn_gen(dim=opt.dim_refine, depth=opt.depth_refine,heads=8,mlp_dim=opt.mlpdim_refine)
    else:
        model= ViT_dualModal(dim=opt.dim_foundation, depth=opt.depth_foundation,heads=8,mlp_dim=opt.mlpdim_foundation)
            # noise_scheduler = DDIMScheduler(num_train_timesteps=n_timesteps,
            #                                 beta_schedule="cosine")

    # if not opt.no_cuda:
    #     model = model.cuda()
    #     #model = nn.DataParallel(model, device_ids=None)
    #     net_dict = model.state_dict()
    # else:
    #     net_dict = model.state_dict()

    # load pretrain
    if opt.pretrain ==True and opt.refine == False:
        print('loading pretrained model {}'.format(opt.pretrain_path))
        if opt.pretrain == True:
            # try:
                checkpoint = torch.load(opt.pretrain_path, weights_only=True, map_location='cuda:0')['state_dict']
                for key in list(checkpoint.keys()):
                    if 'module.' in key:
                        checkpoint[key.replace('module.', '')] = checkpoint[key]
                        del checkpoint[key]
                # opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
                # assert opt.arch == checkpoint['arch']
                # model.load_state_dict(checkpoint['state_dict'])
                model.load_state_dict(checkpoint)
                print('Load Model successfully')
    if opt.refine == True and opt.resume_path != '':
        checkpoint = torch.load(opt.resume_path, weights_only=True, map_location='cuda:0')['state_dict']
        for key in list(checkpoint.keys()):
            if 'module.' in key:
                checkpoint[key.replace('module.', '')] = checkpoint[key]
                del checkpoint[key]
        # opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
        # assert opt.arch == checkpoint['arch']
        # model.load_state_dict(checkpoint['state_dict'])
        model.load_state_dict(checkpoint)
        print('loading pretrained model {}'.format(opt.resume_path))
        print('Load Model successfully')
        # pretrain = torch.load(opt.pretrain_path)
        # pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
        #
        # net_dict.update(pretrain_dict)
        # model.load_state_dict(net_dict)
        #
        # new_parameters = []
        # for pname, p in model.named_parameters():
        #     for layer_name in opt.new_layer_names:
        #         if pname.find(layer_name) >= 0:
        #             new_parameters.append(p)
        #             break

        # except:
        #     print('Load Model unsuccessfully')

    return model, model.parameters()