"""
Learnable generative compression model modified from [1], 
implemented in Pytorch.

Example usage:
python3 train.py -h

[1] Mentzer et. al., "High-Fidelity Generative Image Compression", 
    arXiv:2006.09965 (2020).
"""
import numpy as np
from PIL import Image
import os, glob, time, datetime
import logging, pickle, argparse
import functools, itertools

from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Custom modules
from src.model import Model
from src.helpers import utils, datasets
from default_config import hific_args, mse_lpips_args, directories, ModelModes, ModelTypes

# go fast boi!!
torch.backends.cudnn.benchmark = True

def create_model(args, device, logger, storage, storage_test):

    start_time = time.time()
    model = Model(args, logger, storage, storage_test, model_type=args.model_type)
    logger.info(model)
    logger.info('Trainable parameters:')
    
    for n, p in model.named_parameters():
        logger.info('{} - {}'.format(n, p.shape))

    # Freeze the Generator : 
    model.Decoder.eval()
    for param in model.Decoder.parameters():
        param.requires_grad = False 

    logger.info("Number of trainable parameters: {}".format(utils.count_parameters(model)))
    logger.info("Estimated size (under fp32): {:.3f} MB".format(utils.count_parameters(model) * 4. / 10**6))
    logger.info('Model init {:.3f}s'.format(time.time() - start_time))

    return model

def optimize_loss(loss, opt, retain_graph=False):
    loss.backward(retain_graph=retain_graph)
    opt.step()
    opt.zero_grad()

def optimize_compression_loss(compression_loss, amortization_opt, hyperlatent_likelihood_opt):
    compression_loss.backward()
    amortization_opt.step()
    hyperlatent_likelihood_opt.step()
    amortization_opt.zero_grad()
    hyperlatent_likelihood_opt.zero_grad()

def test(args, model, epoch, idx, data, test_data, test_bpp, device, epoch_test_loss, storage, best_test_loss, 
         start_time, epoch_start_time, logger, train_writer, test_writer):

    model.eval()  
    with torch.no_grad():
        data = data.to(device, dtype=torch.float)

        losses, intermediates = model(data, return_intermediates=True, writeout=False)
        utils.save_images(train_writer, model.step_counter, intermediates.input_image, intermediates.reconstruction,
            fname=os.path.join(args.figures_save, 'recon_epoch{}_idx{}_TRAIN_{:%Y_%m_%d_%H:%M}.jpg'.format(epoch, idx, datetime.datetime.now())))

        test_data = test_data.to(device, dtype=torch.float)
        losses, intermediates = model(test_data, return_intermediates=True, writeout=True)
        utils.save_images(test_writer, model.step_counter, intermediates.input_image, intermediates.reconstruction,
            fname=os.path.join(args.figures_save, 'recon_epoch{}_idx{}_TEST_{:%Y_%m_%d_%H:%M}.jpg'.format(epoch, idx, datetime.datetime.now())))
    
        compression_loss = losses['compression'] 
        epoch_test_loss.append(compression_loss.item())
        mean_test_loss = np.mean(epoch_test_loss)
        
        best_test_loss = utils.log(model, storage, epoch, idx, mean_test_loss, compression_loss.item(), 
                                     best_test_loss, start_time, epoch_start_time, 
                                     batch_size=data.shape[0], avg_bpp=test_bpp.mean().item(),header='[TEST]', 
                                     logger=logger, writer=test_writer)
        
    return best_test_loss, epoch_test_loss

def load_generator(model,path):
    """ Loading the pretrained weights from the HIFI ckpt to the Generator 
    
    path : path of HIFI ckpt
    model : Generator 

    """
    load = torch.load(path)

    new_state_dict = {}
    for name, weight in load['model_state_dict'].items():
        if 'Generator' in name:
            new_state_dict[name] = weight

    model.eval()
    for param in model.parameters():
        param.requires_grad = False 

    return model.load_state_dict(new_state_dict,strict=False)


def eval_jpegai(model, image_dir):
    
    ssim_rec = utils.AverageMeter()
    ssim_zoom = utils.AverageMeter()
    
    psnr_rec = utils.AverageMeter()
    psnr_zoom = utils.AverageMeter()

    model.eval()

    images = glob.glob(os.path.join(image_dir, "*.png"))
    with torch.no_grad():
        for i, img_name in enumerate(tqdm(images)):
            image = Image.open(img_name)

            image_array = np.array(image)
            image_array = image_array.transpose(2,0,1)
            
            image_array = image_array / 255.0
            image_array = (image_array-0.5)/0.5
            image = torch.zeros(1,3, image_array.shape[1], image_array.shape[2])
            tensor_img = torch.from_numpy(image_array)
            image[0] = tensor_img

            image = F.upsample(image, size=(image.size(2)//2, image.size(3)//2), mode='bicubic')
            image = image.to(device)
            print(image.shape)
            # losses = model(image)
            # model.model_mode = 'validation'
            losses = model(image, return_intermediates=False, writeout=False)

            compression_loss = losses['compression']

            if "HiFiC" in args.tasks:
                
                ssim = losses['perceptual rec']
                ssim_rec.update(ssim, image.size(0))

                psnr = losses['psnr rec']
                psnr_rec.update(psnr, image.size(0))

            if "Zoom" in args.tasks:
                ssim = losses['perceptual zoom']
                ssim_zoom.update(ssim, image.size(0))

                psnr = losses['psnr zoom']
                psnr_zoom.update(psnr.item(), image.size(0))
                
            if i % 5 == 0:
                print('Test_JPEGAI: [{0}/{1}]\t'
                        'psnr_rec {psnr_rec.val:.4f} ({psnr_rec.avg:.4f})\t'
                        # 'mse_rec {mse_rec.val:.4f} ({mse_rec.avg:.4f})\t'
                        'ssim_rec {ssim_rec.val:.3f} ({ssim_rec.avg:.3f})\t'.format(
                        i, len(images), psnr_rec = psnr_rec, ssim_rec = ssim_rec))

        print('Test_JPEGAI: [{0}/{1}]\t'
                    'psnr_rec {psnr_rec.val:.4f} ({psnr_rec.avg:.4f})\t'
                    # 'mse_rec {mse_rec.val:.4f} ({mse_rec.avg:.4f})\t'
                    'ssim_rec {ssim_rec.val:.3f} ({ssim_rec.avg:.3f})\t'.format(
                    i+1, len(images), psnr_rec = psnr_rec, ssim_rec = ssim_rec))

        
        # return psnr_rec.avg, ssim_rec.avg


def train(args, model, train_loader, test_loader, jpeg_loader, device, logger, optimizers):

    start_time = time.time()
    test_loader_iter = iter(test_loader)
    # jpeg_loader_iter = iter(jpeg_loader)
    current_D_steps, train_generator = 0, True
    best_loss, best_test_loss, mean_epoch_loss = np.inf, np.inf, np.inf     
    train_writer = SummaryWriter(os.path.join(args.tensorboard_runs, 'train'))
    test_writer = SummaryWriter(os.path.join(args.tensorboard_runs, 'test'))
    jpegai_writer = SummaryWriter(os.path.join(args.tensorboard_runs, 'jpegai'))
    storage, storage_test = model.storage_train, model.storage_test

    amortization_opt, hyperlatent_likelihood_opt = optimizers['amort'], optimizers['hyper']

    if model.use_discriminator is True:
        disc_opt = optimizers['disc']

    for epoch in trange(args.n_epochs, desc='Epoch'):
        
        ssim_rec_train = utils.AverageMeter()
        ssim_zoom_train = utils.AverageMeter()
        psnr_rec_train = utils.AverageMeter()
        psnr_zoom_train = utils.AverageMeter()
        cosine_ffx_train = utils.AverageMeter()

        # ssim_rec_test = utils.AverageMeter()
        # ssim_zoom_test = utils.AverageMeter()
        # psnr_rec_test = utils.AverageMeter()
        # psnr_zoom_test = utils.AverageMeter()
        # cosine_ffx_test = utils.AverageMeter()

        epoch_loss, epoch_test_loss = [], []  
        epoch_start_time = time.time()
        
        if epoch > 0:
            ckpt_path = utils.save_model(model, optimizers, mean_epoch_loss, epoch, device, args=args, logger=logger)
        
        model.train()

        for idx, (data, bpp) in enumerate(tqdm(train_loader, desc='Train'), 0):

            data = data.to(device, dtype=torch.float)
            
            # eval_jpegai(model, args.image_dir)

            try:
                if model.use_discriminator is True:
                    # Train D for D_steps, then G, using distinct batches
                    losses = model(data, train_generator=train_generator)
                    compression_loss = losses['compression']
                    disc_loss = losses['disc']

                    if train_generator is True:
                        optimize_compression_loss(compression_loss, amortization_opt, hyperlatent_likelihood_opt)
                        train_generator = False
                    else:
                        optimize_loss(disc_loss, disc_opt)
                        current_D_steps += 1

                        if current_D_steps == args.discriminator_steps:
                            current_D_steps = 0
                            train_generator = True

                        continue
                else:
                    # Rate, perceptual (SSIM) only
                    losses = model(data, train_generator=True)
                    compression_loss = losses['compression']
                    optimize_compression_loss(compression_loss, amortization_opt, hyperlatent_likelihood_opt)
 
            except KeyboardInterrupt:
                # Note: saving not guaranteed!
                if model.step_counter > args.log_interval+1:
                    logger.warning('Exiting, saving ...')
                    ckpt_path = utils.save_model(model, optimizers, mean_epoch_loss, epoch, device, args=args, logger=logger)
                    return model, ckpt_path
                else:
                    return model, None
            
            update_performance(args, ssim_rec_train, ssim_zoom_train, psnr_rec_train, psnr_zoom_train, cosine_ffx_train, storage)

            # Plot on Tensorboard per iteration
            # utils.log_summaries(args, train_writer, storage, ssim_rec_train, ssim_zoom_train, psnr_rec_train, psnr_zoom_train, cosine_ffx_train, idx, mode = 'train_lfw', use_discriminator=model.use_discriminator)

            if model.step_counter % args.log_interval == 1:
                
                epoch_loss.append(compression_loss.item())
                mean_epoch_loss = np.mean(epoch_loss)

                # best_loss = utils.log(model, storage, epoch, idx, mean_epoch_loss, compression_loss.item(),
                #                 best_loss, start_time, epoch_start_time, batch_size=data.shape[0],
                #                 avg_bpp=bpp.mean().item(), logger=logger, writer=train_writer)
                try:
                    test_data, test_bpp = next(test_loader_iter)
                    # jpeg_test_data, jpeg_test_bpp = next(jpeg_loader_iter)
                except StopIteration:
                    test_loader_iter = iter(test_loader)
                    test_data, test_bpp = next(test_loader_iter)

                # best_test_loss, epoch_test_loss = test(args, model, epoch, idx, data, test_data, test_bpp, device, epoch_test_loss, storage_test,
                #      best_test_loss, start_time, epoch_start_time, logger, train_writer, test_writer)

                with open(os.path.join(args.storage_save, 'storage_{}_tmp.pkl'.format(args.name)), 'wb') as handle:
                    pickle.dump(storage, handle, protocol=pickle.HIGHEST_PROTOCOL)

                model.train()

                # LR scheduling
                utils.update_lr(args, amortization_opt, model.step_counter, logger)
                utils.update_lr(args, hyperlatent_likelihood_opt, model.step_counter, logger)
                if model.use_discriminator is True:
                    utils.update_lr(args, disc_opt, model.step_counter, logger)

                if model.step_counter > args.n_steps:
                    logger.info('Reached step limit [args.n_steps = {}]'.format(args.n_steps))
                    break
                
                print_performance(args, len(train_loader), bpp.mean().item(), storage, ssim_rec_train, ssim_zoom_train, psnr_rec_train, psnr_zoom_train, cosine_ffx_train, idx, epoch)

            if (idx % args.save_interval == 1) and (idx > args.save_interval):
                ckpt_path = utils.save_model(model, optimizers, mean_epoch_loss, epoch, device, args=args, logger=logger)
            
        # End epoch

        # Plot on Tensorboard per Epoch
        utils.log_summaries(args, train_writer, storage, ssim_rec_train, ssim_zoom_train, psnr_rec_train, psnr_zoom_train, cosine_ffx_train, epoch, mode = 'train_lfw', use_discriminator=model.use_discriminator)

        # eval_lfw(model, args.image_dir)
        #Update performance for lfw test loop
        # update_performance(args, ssim_rec_test, ssim_zoom_test, psnr_rec_test, psnr_zoom_test, cosine_ffx_test, losses)
        # print_performance(args, len(train_loader), bpp.mean().item(), storage, ssim_rec_test, ssim_zoom_test, psnr_rec_test, psnr_zoom_test, cosine_ffx_test, idx, epoch)
        # utils.log_summaries(args, test_writer, storage, ssim_rec_test, ssim_zoom_test, psnr_rec_test, psnr_zoom_test, cosine_ffx_test, epoch, mode = 'val_lfw', use_discriminator=model.use_discriminator)

        # eval_jpegai(model, args.image_dir)
        #Update performance for jpegai test loop
        # update_performance(args, ssim_rec_jpegai, ssim_zoom_jpegai, psnr_rec_jpegai, psnr_zoom_jpegai, None, losses)
        # print_performance(args, len(train_loader), bpp.mean().item(), storage, ssim_rec_jpegai, ssim_zoom_jpegai, psnr_rec_jpegai, psnr_zoom_jpegai, None, idx, epoch)
        # utils.log_summaries(args, jpegai_writer, storage, ssim_rec_jpegai, ssim_zoom_jpegai, psnr_rec_jpegai, psnr_zoom_jpegai, None, epoch, mode = 'jpegai_test', use_discriminator=model.use_discriminator)

        mean_epoch_loss = np.mean(epoch_loss)
        mean_epoch_test_loss = np.mean(epoch_test_loss)

        
        
        logger.info('===>> Epoch {} | Mean train loss: {:.3f} | Mean test loss: {:.3f}'.format(epoch, 
            mean_epoch_loss, mean_epoch_test_loss))    
        
        # logger.info('===>> Epoch {}\tSSIM Train: {:.3f}({:.3f})\tSSIM Test: {:.3f}({:.3f})'.format(epoch, 
        #     1 - mean_epoch_loss, 1 - mean_epoch_test_loss))  

        if model.step_counter > args.n_steps:
            break
    
    with open(os.path.join(args.storage_save, 'storage_{}_{:%Y_%m_%d_%H:%M:%S}.pkl'.format(args.name, datetime.datetime.now())), 'wb') as handle:
        pickle.dump(storage, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    ckpt_path = utils.save_model(model, optimizers, mean_epoch_loss, epoch, device, args=args, logger=logger)
    args.checkpoint = ckpt_path
    logger.info("Training complete. Time elapsed: {:.3f} s. Number of steps: {}".format((time.time()-start_time), model.step_counter))
    
    return model, ckpt_path

def update_performance(args, ssim_rec, ssim_zoom, psnr_rec, psnr_zoom, cosine_ffx, store):
    if "HiFiC" in args.tasks:
        ssim = store['perceptual rec'][-1]
        ssim_rec.update(ssim, args.batch_size)

        psnr = store['psnr rec'][-1]
        psnr_rec.update(psnr, args.batch_size)

    if "Zoom" in args.tasks:
        ssim = store['perceptual zoom'][-1]
        ssim_zoom.update(ssim, args.batch_size)

        psnr = store['psnr zoom'][-1]
        psnr_zoom.update(psnr, args.batch_size)

    if "FFX" in args.tasks and cosine_ffx is not None:
        cosine = store['cosine sim'][-1]
        cosine_ffx.update(cosine, args.batch_size)


def print_performance(args, data_size, avg_bpp, storage, ssim_rec, ssim_zoom, psnr_rec, psnr_zoom, cosine_ffx, idx, epoch):
    
    display = '\nEPOCH: [{0}][{1}/{2}]'.format(epoch, idx, data_size)
    if "HiFiC" in args.tasks:
        display += '\tpsnr_rec {psnr_rec.val:.3f} ({psnr_rec.avg:.3f})\t ssim_rec {ssim_rec.val:.3f} ({ssim_rec.avg:.3f})'.format(psnr_rec = psnr_rec, ssim_rec = ssim_rec)
    
    if "Zoom" in args.tasks:
        display += '\tpsnr_zoom {psnr_zoom.val:.3f} ({psnr_zoom.avg:.3f})\t ssim_zoom {ssim_zoom.val:.3f} ({ssim_zoom.avg:.3f})'.format(psnr_zoom = psnr_zoom, ssim_zoom = ssim_zoom)
          
    if "FFX" in args.tasks and cosine_ffx is not None:
        display += '\tcosine_ffx {cosine_ffx.val:.3f} ({cosine_ffx.avg:.3f})'.format(cosine_ffx = cosine_ffx)
    
    if "HiFiC" in args.tasks:
        display += '\n'
        display += "Rate-Distortion:\n"
        display += "Weighted Rate: {:.3f} | Perceptual: {:.3f} | Rate Penalty: {:.3f}".format(storage['weighted_rate'][-1], 
                                            storage['perceptual'][-1], storage['rate_penalty'][-1])

        display += '\n'
        display += "Rate Breakdown:\n"
        display += "avg. original bpp: {:.3f} | n_bpp (total): {:.3f} | q_bpp (total): {:.3f} | n_bpp (latent): {:.3f} | q_bpp (latent): {:.3f} | n_bpp (hyp-latent): {:.3f} | q_bpp (hyp-latent): {:.3f}".format(avg_bpp, storage['n_rate'][-1], storage['q_rate'][-1], 
                storage['n_rate_latent'][-1], storage['q_rate_latent'][-1], storage['n_rate_hyperlatent'][-1], storage['q_rate_hyperlatent'][-1])
        
    display += '\n'
    display += '=' * 180
    
    print(display)


if __name__ == '__main__':

    description = "Learnable generative compression."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General options - see `default_config.py` for full options
    general = parser.add_argument_group('General options')
    general.add_argument("-n", "--name", default=None, help="Identifier for checkpoints and metrics.")
    general.add_argument("-mt", "--model_type", required=True, default=ModelTypes.COMPRESSION, choices=(ModelTypes.COMPRESSION, ModelTypes.COMPRESSION_GAN), 
        help="Type of model - with or without GAN component")
    general.add_argument("-regime", "--regime", choices=('low','med','high'), default='high', help="Set target bit rate - Low (0.14), Med (0.30), High (0.45)")
    general.add_argument("-gpu", "--gpu", type=int, default=0, help="GPU ID.")
    general.add_argument("-log_intv", "--log_interval", type=int, default=hific_args.log_interval, help="Number of steps between logs.")
    general.add_argument("-save_intv", "--save_interval", type=int, default=hific_args.save_interval, help="Number of steps between checkpoints.")
    general.add_argument("-multigpu", "--multigpu", help="Toggle data parallel capability using torch DataParallel", action="store_true")
    general.add_argument("-norm", "--normalize_input_image", help="Normalize input images to [-1,1]", action="store_true")
    general.add_argument('-bs', '--batch_size', type=int, default=hific_args.batch_size, help='input batch size for training')
    general.add_argument('--save', type=str, default='experiments', help='Parent directory for stored information (checkpoints, logs, etc.)')
    general.add_argument("-lt", "--likelihood_type", choices=('gaussian', 'logistic'), default='gaussian', help="Likelihood model for latents.")
    general.add_argument("-force_gpu", "--force_set_gpu", help="Set GPU to given ID", action="store_true")
    general.add_argument("-LMM", "--use_latent_mixture_model", help="Use latent mixture model as latent entropy model.", action="store_true")

    # Optimization-related options
    optim_args = parser.add_argument_group("Optimization-related options")
    optim_args.add_argument('-steps', '--n_steps', type=float, default=hific_args.n_steps, 
        help="Number of gradient steps. Optimization stops at the earlier of n_steps/n_epochs.")
    optim_args.add_argument('-epochs', '--n_epochs', type=int, default=hific_args.n_epochs, 
        help="Number of passes over training dataset. Optimization stops at the earlier of n_steps/n_epochs.")
    optim_args.add_argument("-lr", "--learning_rate", type=float, default=hific_args.learning_rate, help="Optimizer learning rate.")
    optim_args.add_argument("-wd", "--weight_decay", type=float, default=hific_args.weight_decay, help="Coefficient of L2 regularization.")

    # Architecture-related options
    arch_args = parser.add_argument_group("Architecture-related options")
    arch_args.add_argument('-lc', '--latent_channels', type=int, default=hific_args.latent_channels,
        help="Latent channels of bottleneck nominally compressible representation.")
    arch_args.add_argument('-nrb', '--n_residual_blocks', type=int, default=hific_args.n_residual_blocks,
        help="Number of residual blocks to use in Generator.")
    
    arch_args.add_argument('-t', '--tasks', choices=['HiFiC', 'Zoom','FFX'], nargs='+', default=hific_args.default_task, help="Choose which task to add into the MTL framework")

    # Warmstart adversarial training from autoencoder/hyperprior
    warmstart_args = parser.add_argument_group("Warmstart options")
    warmstart_args.add_argument("-warmstart", "--warmstart", help="Warmstart adversarial training from autoencoder + hyperprior ckpt.", action="store_true")
    warmstart_args.add_argument("-ckpt", "--checkpoint", default=hific_args.checkpoint, help="Path to autoencoder + hyperprior ckpt.")

    cmd_args = parser.parse_args()

    if (cmd_args.gpu != 0) or (cmd_args.force_set_gpu is True):
        torch.cuda.set_device(cmd_args.gpu)

    if cmd_args.model_type == ModelTypes.COMPRESSION:
        args = mse_lpips_args
    elif cmd_args.model_type == ModelTypes.COMPRESSION_GAN:
        args = hific_args

    start_time = time.time()
    device = utils.get_device()

    # Override default arguments from config file with provided command line arguments
    dictify = lambda x: dict((n, getattr(x, n)) for n in dir(x) if not (n.startswith('__') or 'logger' in n))
    args_d, cmd_args_d = dictify(args), vars(cmd_args)
    args_d.update(cmd_args_d)
    args = utils.Struct(**args_d)
    args = utils.setup_generic_signature(args, special_info=args.model_type)
    args.target_rate = args.target_rate_map[args.regime]
    args.lambda_A = args.lambda_A_map[args.regime]
    args.n_steps = int(args.n_steps)

    storage = defaultdict(list)
    storage_test = defaultdict(list)
    logger = utils.logger_setup(logpath=os.path.join(args.snapshot, 'logs'), filepath=os.path.abspath(__file__))

    # if args.warmstart is True:
    #     assert args.warmstart_ckpt is not None, 'Must provide checkpoint to previously trained AE/HP model.'
    #     logger.info('Warmstarting discriminator/generator from autoencoder/hyperprior model.')
    #     if args.model_type != ModelTypes.COMPRESSION_GAN:
    #         logger.warning('Should warmstart compression-gan model.')
    #     args, model, optimizers = utils.load_model(args.warmstart_ckpt, logger, device, 
    #         model_type=args.model_type, current_args_d=dictify(args), strict=False, prediction=False)
    # else:
    model = create_model(args, device, logger, storage, storage_test)
    model = model.to(device)
    amortization_parameters = itertools.chain.from_iterable(
        [am.parameters() for am in model.amortization_models])

    hyperlatent_likelihood_parameters = model.Hyperprior.hyperlatent_likelihood.parameters()

    amortization_opt = torch.optim.Adam(amortization_parameters,
        lr=args.learning_rate)
    hyperlatent_likelihood_opt = torch.optim.Adam(hyperlatent_likelihood_parameters, 
        lr=args.learning_rate)
    optimizers = dict(amort=amortization_opt, hyper=hyperlatent_likelihood_opt)

    if model.use_discriminator is True:
        discriminator_parameters = model.Discriminator.parameters()
        disc_opt = torch.optim.Adam(discriminator_parameters, lr=args.learning_rate)
        optimizers['disc'] = disc_opt

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1 and args.multigpu is True:
        # Not supported at this time
        raise NotImplementedError('MultiGPU not supported yet.')
        logger.info('Using {} GPUs.'.format(n_gpus))
        model = nn.DataParallel(model)

    logger.info('MODEL TYPE: {}'.format(args.model_type))
    logger.info('MODEL MODE: {}'.format(args.model_mode))
    logger.info('TASKS: {}'.format(args.tasks))
    logger.info('BITRATE REGIME: {}'.format(args.regime))
    logger.info('SAVING LOGS/CHECKPOINTS/RECORDS TO {}'.format(args.snapshot))
    logger.info('USING DEVICE {}'.format(device))
    logger.info('USING GPU ID {}'.format(args.gpu))
    logger.info('USING DATASET: {}'.format(args.dataset))

    test_loader = datasets.get_dataloaders(args.dataset,
                                root=args.dataset_path,
                                batch_size=args.batch_size,
                                logger=logger,
                                split='test',
                                shuffle=True,
                                normalize=args.normalize_input_image)

    train_loader = datasets.get_dataloaders(args.dataset,
                                root=args.dataset_path,
                                batch_size=args.batch_size,
                                logger=logger,
                                split='train',
                                shuffle=True,
                                normalize=args.normalize_input_image)


    jpeg_loader = datasets.get_dataloaders('evaluation', root=args.image_dir, batch_size=args.batch_size,
                                           logger=logger, shuffle=False, normalize=args.normalize_input_image)


    args.n_data = len(train_loader.dataset)
    args.image_dims = train_loader.dataset.image_dims
    logger.info("=" * 50)
    logger.info('Training elements: {}'.format(args.n_data))
    logger.info('Testing elements: {}'.format(len(test_loader.dataset)))
    logger.info('JPEGAI elements: {}'.format(len(jpeg_loader.dataset)))
    logger.info('Input Dimensions: {}'.format(args.image_dims))
    logger.info('Batch size: {}'.format(args.batch_size))
    logger.info('Optimizers: {}'.format(optimizers))
    logger.info('Using device {}'.format(device))
    logger.info("=" * 50)


    metadata = dict((n, getattr(args, n)) for n in dir(args) if not (n.startswith('__') or 'logger' in n))
    logger.info(metadata)

    """
    Train
    """
    model, ckpt_path = train(args, model, train_loader, test_loader, jpeg_loader, device, logger, optimizers=optimizers)

    """
    TODO
    Generate metrics
    """
