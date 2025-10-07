# BrainCSD
A implementation of BrainCSD. The implementation can be three stages: (1) ROI activation (2) Encode MOE activation (3) finetune.

Stage 1

Set the transformer embedding parameter (dim, mlp_dim, depth, e.g.) in the roi activation files. Asve the model weights for the training of stage 2 and 3. The fMRI and dMRI used parameter should be different, since their resolution is different.
Since the signals in atlas \times dMRI is low, the signals should be enced by 10* when inputing the target

Stage 2

Before setting the parameter of MOE  transformer, you should load the pretrained model from Stage 1. Then if you want to design the numbers of expert, you can design them in the connect_gen.py in the models directory of Encode files. We do not set 
a paramerter in the opt.py

Stage 3

load the pretrained model from stage 1 and stage 2, set the refine mode as FC/SC. Remember to choose the optimal parameter of transformer embbeding layers
