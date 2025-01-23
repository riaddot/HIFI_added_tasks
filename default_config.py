#!/usr/bin/env python3

import os 

"""
Default arguments from [1]. Entries can be manually overriden via
command line arguments in `train.py`.

[1]: arXiv 2006.09965
"""

class ModelTypes(object):
    COMPRESSION = 'compression'
    COMPRESSION_GAN = 'compression_gan'

class ModelModes(object):
    TRAINING = 'training'
    VALIDATION = 'validation'
    EVALUATION = 'evaluation'  # actual entropy coding

class Datasets(object):
    LFWPeople = 'lfw'
    OPENIMAGES = 'openimages'
    CITYSCAPES = 'cityscapes'
    JETS = 'jetimages'
    FaceForensicsDataset = "ff++"

class DatasetPaths(object):
    LFWPeople = "/home/bellelbn/DL/datasets/lfw" #'/home/bellelbn/DL/datasets/lfw'
    OPENIMAGES = '/home/bellelbn/DL/datasets/OID'
    CITYSCAPES = ''
    JETS = ''
    FaceForensicsDataset = "/home/bellelbn/DL/datasets/Faceforensics"

class directories(object):
    experiments = '/home/bellelbn/DL/datapart/jpegai_experiments/experiments'
    baseline_experiments = os.path.join(experiments, "baseline")

class args(object):
    """
    Shared config
    """
    name = 'hific_v0.1'
    silent = True
    n_epochs = 40
    n_steps = 1e6
    batch_size = 32
    boost_compression = 3
    log_interval = 100
    save_interval = 50000
    gpu = 0
    multigpu = True
    dataset = Datasets.FaceForensicsDataset #OPENIMAGES
    dataset_path = DatasetPaths.FaceForensicsDataset #OPENIMAGES
    shuffle = True
    
    fake_type = None
    dataset_type = 'original'
    nframes = 32

    # GAN params
    discriminator_steps = 0
    model_mode = ModelModes.TRAINING
    sample_noise = False
    noise_dim = 32

    # Architecture params - defaults correspond to Table 3a) of [1]
    latent_channels = 220
    n_residual_blocks = 9           # Authors use 9 blocks, performance saturates at 5
    lambda_B = 2**(-4)              # Loose rate
    k_M = 0.075 * 2**(-5)           # Distortion
    k_P = 1.                        # Perceptual loss
    beta = 0.15                     # Generator loss
    use_channel_norm = True
    likelihood_type = 'gaussian'    # Latent likelihood model
    normalize_input_image = False   # Normalize inputs to range [-1,1]
    
    # Shapes
    crop_size = 256
    image_dims = (3,256,256)
    latent_dims = (latent_channels,16,16)
    
    # Optimizer params
    learning_rate = 5e-5
    weight_decay = 1e-6

    # Scheduling
    lambda_schedule = dict(vals=[1., 1.], steps=[500000])
    lr_schedule = dict(vals=[1., 0.1], steps=[500000])
    target_schedule = dict(vals=[1., 0.20/0.14], steps=[500000])  # Rate allowance
                                                #steps=[50000]
    ignore_schedule = False

    # match target rate to lambda_A coefficient
    regime = 'high'  # -> 0.45
    target_rate_map = dict(low=0.14, med=0.3, high=0.45)
    lambda_A_map = dict(low=2**1, med=2**0, high=2**(-1))
    target_rate = target_rate_map[regime]
    lambda_A = lambda_A_map[regime]

    double_compression = 1

    dataset_type = "original"
    nframes = 32
    
    # DLMM
    use_latent_mixture_model = False
    mixture_components = 4
    latent_channels_DLMM = 64
    
    default_task = "HiFiC"
    image_dir = "/home/bellelbn/DL/datasets/jpegai_test"
    checkpoint = "/home/bellelbn/DL/datapart/models/hific_hi.pt"

    hific_hi_checkpoint = "/home/bellelbn/DL/datapart/models/hific_hi.pt"
    hific_med_checkpoint = "/home/bellelbn/DL/datapart/models/hific_mi.pt"
    hific_low_checkpoint = "/home/bellelbn/DL/datapart/models/hific_low.pt"

    hific_zoom = "/home/bellelbn/DL/datapart/jpegai_experiments/experiments/lfw_compression_high_adapt_linear_origW_2024_09_07_02_12/HiFiC_Zoom"
    hific_ffx = "/home/bellelbn/DL/datapart/jpegai_experiments/experiments/lfw_compression_high_adapt_linear_origW_2024_09_07_17_04/HiFiC_FFX"
    
    # all these checkpoints are obtained from a HiFiC Hi codec targeting different bpp
#     checkpoint_paths_for_pruning = {
#     "encoder_0.45_bpp": "/home/bellelbn/DL/datapart/jpegai_experiments/experiments/lfw_compression_ln_adapt_linear_origW_2024_08_13_12_16/HiFiC_Zoom_FFX/best_checkpoint.pt",
#     "encoder_0.30_bpp": "/home/bellelbn/DL/datapart/jpegai_experiments/experiments/lfw_compression_med_adapt_linear_origW_2024_09_11_22_41/HiFiC_Zoom_FFX/best_checkpoint.pt",
#     "encoder_0.14_bpp": "/home/bellelbn/DL/datapart/jpegai_experiments/experiments/lfw_compression_low_adapt_linear_origW_2024_09_11_22_41/HiFiC_Zoom_FFX/best_checkpoint.pt"
# }
    
    checkpoint_paths_for_pruning = {
    "encoder_0.45_bpp": hific_hi_checkpoint,
    # "encoder_0.30_bpp": hific_med_checkpoint,
    # "encoder_0.14_bpp": hific_low_checkpoint
}
    
"""
Specialized configs
"""

class mse_lpips_args(args):
    """
    Config for model trained with distortion and 
    perceptual loss only.
    """
    model_type = ModelTypes.COMPRESSION

class hific_args(args):
    """
    Config for model trained with full generative
    loss terms.
    """
    model_type = ModelTypes.COMPRESSION_GAN
    gan_loss_type = 'non_saturating'  # ('non_saturating', 'least_squares')
    discriminator_steps = 1
    sample_noise = False
