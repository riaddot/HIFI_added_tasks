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
import copy

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms

# Custom modules
from src.model import Model
from src.helpers import utils, datasets
from default_config import hific_args, mse_lpips_args, directories, ModelModes, ModelTypes


# go fast boi!!
torch.backends.cudnn.benchmark = True

def create_model(args, device, logger, storage, storage_val, storage_test):

    start_time = time.time()
    model = Model(args, logger, storage, storage_val, storage_test, model_type=args.model_type)
    # logger.info(model)
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

def test(args, model, epoch, idx, data, test_data, test_bpp, device, epoch_test_loss, storage, best_val_loss, 
         start_time, epoch_start_time, logger, train_writer, val_writer):

    model.eval()  
    with torch.no_grad():
        data = data.to(device, dtype=torch.float)

        losses, intermediates = model(data, return_intermediates=True, writeout=False)
        utils.save_images(train_writer, model.step_counter, intermediates.input_image, intermediates.reconstruction,
            fname=os.path.join(args.figures_save, 'recon_epoch{}_idx{}_TRAIN_{:%Y_%m_%d_%H:%M}.jpg'.format(epoch, idx, datetime.datetime.now())))

        test_data = test_data.to(device, dtype=torch.float)
        losses, intermediates = model(test_data, return_intermediates=True, writeout=True)
        utils.save_images(val_writer, model.step_counter, intermediates.input_image, intermediates.reconstruction,
            fname=os.path.join(args.figures_save, 'recon_epoch{}_idx{}_TEST_{:%Y_%m_%d_%H:%M}.jpg'.format(epoch, idx, datetime.datetime.now())))
    
        compression_loss = losses['compression'] 
        epoch_test_loss.append(compression_loss.item())
        mean_test_loss = np.mean(epoch_test_loss)
        
        best_val_loss = utils.log(model, storage, epoch, idx, mean_test_loss, compression_loss.item(), 
                                     best_val_loss, start_time, epoch_start_time, 
                                     batch_size=data.shape[0], bpp=test_bpp.mean().item(),header='[TEST]', 
                                     logger=logger, writer=val_writer)
        
    return best_val_loss, epoch_test_loss

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


def eval_lfw_jpegai(args, epoch, model, val_loader, device, writer, dataset = "lfw"):

    val_loss = utils.AverageMeter()
    ssim_rec_val = utils.AverageMeter()
    ssim_zoom_val = utils.AverageMeter()
    psnr_rec_val = utils.AverageMeter()
    psnr_zoom_val = utils.AverageMeter()
    cosine_ffx_val = utils.AverageMeter()

    compression_loss_sans_G = utils.AverageMeter()
    weighted_rate = utils.AverageMeter()
    perceptual = utils.AverageMeter()
    rate_penalty = utils.AverageMeter()
    bpp = utils.AverageMeter()
    n_rate = utils.AverageMeter()
    q_rate = utils.AverageMeter()
    n_rate_latent = utils.AverageMeter()
    q_rate_latent = utils.AverageMeter()
    n_rate_hyperlatent = utils.AverageMeter()
    q_rate_hyperlatent = utils.AverageMeter()


    metrics = {
    'loss': val_loss,
    'ssim_rec': ssim_rec_val,
    'ssim_zoom': ssim_zoom_val,
    'psnr_rec': psnr_rec_val,
    'psnr_zoom': psnr_zoom_val,
    'cosine_ffx': cosine_ffx_val,
    'compression_loss_sans_G' : compression_loss_sans_G,
    'weighted_rate' : weighted_rate,
    'perceptual' : perceptual,
    'rate_penalty' : rate_penalty,
    'bpp' : bpp,
    'n_rate' : n_rate,
    'q_rate' : q_rate,
    'n_rate_latent' : n_rate_latent,
    'q_rate_latent' : q_rate_latent,
    'n_rate_hyperlatent' : n_rate_hyperlatent,
    'q_rate_hyperlatent' : q_rate_hyperlatent
    }
    
    model.eval()

    model.val_data = dataset

    with torch.no_grad():
        if dataset == "lfw":
            
            storage = model.storage_val

            for idx, (data, bpp) in enumerate(tqdm(val_loader, desc='Val'), 0):

                data = data.to(device, dtype=torch.float)

                losses, intermediates = model(data, return_intermediates=True, writeout=True)
                storage["bpp"].append(bpp.mean().item())
                # update_performance(args, val_loss, ssim_rec_val, ssim_zoom_val, psnr_rec_val, psnr_zoom_val, cosine_ffx_val, storage)
                update_performance(args, metrics, storage)

                # if idx % 100 == 0:
                #     print_performance(args, len(val_loader), bpp.mean().item(), storage, val_loss, ssim_rec_val, ssim_zoom_val, psnr_rec_val, psnr_zoom_val, cosine_ffx_val, idx, epoch, model.training)

        elif dataset == "jpegai":
            
            storage = model.storage_test
            model.model_mode = ModelModes.EVALUATION

            device = torch.device("cpu")
            model.to(device)
            for idx, (data, bpp, filename) in enumerate(tqdm(val_loader, desc='Val'), 0):
                data = data.to(device, dtype=torch.float)

                outputs, intermediates = model(data, return_intermediates=True, writeout=True)
                if "Zoom" in args.tasks or args.test_task:
                    reconst, reconst_zoom = outputs
                    
                    fname=os.path.join(args.figures_save, 'zoom_{}.png'.format(filename[0]))
                    save_image(fname, reconst_zoom)
                else:
                    reconst = outputs

                fname=os.path.join(args.figures_save, 'recon_{}.png'.format(filename[0]))
                logger.info(fname)
                save_image(fname, reconst)

                storage["bpp"].append(bpp.mean().item())
                # update_performance(args, None, ssim_rec_val, ssim_zoom_val, psnr_rec_val, psnr_zoom_val, None, storage)
                metrics['loss'] = None
                metrics['cosine_ffx'] = None
                update_performance(args, metrics, storage, "jpegai")

                print_performance(args, len(val_loader), metrics, idx, epoch, model.training)

            model.model_mode = ModelModes.TRAINING
        else:
            raise Exception("dataset {} not found. Please choose 'lfw' or 'jpegai'".format(dataset))
    
    
    # print_performance(args, len(val_loader), bpp.mean().item(), storage, val_loss, ssim_rec_val, ssim_zoom_val, psnr_rec_val, psnr_zoom_val, cosine_ffx_val, idx, epoch, model.training)
    print_performance(args, len(val_loader), metrics, idx, epoch, model.training)

    if writer is not None:
        # utils.log_summaries(args, writer, storage, val_loss, ssim_rec_val, ssim_zoom_val, psnr_rec_val, psnr_zoom_val, cosine_ffx_val, epoch, mode = 'lfw', use_discriminator=model.use_discriminator)
        utils.log_summaries(args, writer, metrics, epoch, mode = 'lfw', use_discriminator=model.use_discriminator)

    return val_loss.avg, ssim_rec_val.avg, ssim_zoom_val.avg, cosine_ffx_val.avg

def save_image(filename, reconst):
    transform = transforms.ToPILImage()
    image = transform(reconst.squeeze())
    # Save the image
    image.save(filename)


def compare_params(initial_params, current_params):
    for key in initial_params.keys():
        if not torch.equal(initial_params[key], current_params[key]):
            return False
    return True


def train(args, model, train_loader, val_loader, jpeg_loader, device, logger, optimizers):

    start_time = time.time()

    current_D_steps, train_generator = 0, True
    best_loss, best_val_loss, mean_epoch_loss = np.inf, np.inf, np.inf     
    train_writer = SummaryWriter(os.path.join(args.tensorboard_runs, 'train'))
    val_writer = SummaryWriter(os.path.join(args.tensorboard_runs, 'val'))
    jpegai_writer = SummaryWriter(os.path.join(args.tensorboard_runs, 'jpegai'))

    amortization_opt, hyperlatent_likelihood_opt = optimizers['amort'], optimizers['hyper']
    
    if model.use_discriminator is True:
        disc_opt = optimizers['disc']


    lambda_lr = lambda epoch: 0.95 ** epoch
    scheduler_amortization = LambdaLR(amortization_opt, lambda_lr)
    scheduler_hyperlatent_likelihood = LambdaLR(hyperlatent_likelihood_opt, lambda_lr)
    
    
    if args.evaluate:    

        model.load_checkpoint(args.checkpoint)
        model = model.to(device)

        model.args.norm_loss = False

        logger.info("LFW evaluation")
        val_loss, _, _, _ = eval_lfw_jpegai(args, 0, model, val_loader, device, None)

        logger.info("=" * 150)
        logger.info("\n")

        logger.info("JPEGAI evaluation")
        val_loss, _, _, _ = eval_lfw_jpegai(args, 0, model, jpeg_loader, device, None, "jpegai")

        logger.info("=" * 150)
        logger.info("\n")
        logger.info("End of Evaluation")

        return
    
    if args.norm_loss:
        val_loss, a, b, c = eval_lfw_jpegai(args, 0, model, val_loader, device, None)

        model.a = a
        model.b = b
        model.c = c

        logger.info("Loss normalizatin weights : a {}, b {}, c {}".format(a, b, c))


    for epoch in trange(args.n_epochs, desc='Epoch'):
        
        storage = model.storage_train

        loss_train = utils.AverageMeter() 
        ssim_rec_train = utils.AverageMeter()
        ssim_zoom_train = utils.AverageMeter()
        psnr_rec_train = utils.AverageMeter()
        psnr_zoom_train = utils.AverageMeter()
        cosine_ffx_train = utils.AverageMeter()
        compression_loss_sans_G = utils.AverageMeter()
        weighted_rate = utils.AverageMeter()
        perceptual = utils.AverageMeter()
        rate_penalty = utils.AverageMeter()
        bpp = utils.AverageMeter()
        n_rate = utils.AverageMeter()
        q_rate = utils.AverageMeter()
        n_rate_latent = utils.AverageMeter()
        q_rate_latent = utils.AverageMeter()
        n_rate_hyperlatent = utils.AverageMeter()
        q_rate_hyperlatent = utils.AverageMeter()

        metrics = {
        'loss': loss_train,
        'ssim_rec': ssim_rec_train,
        'ssim_zoom': ssim_zoom_train,
        'psnr_rec': psnr_rec_train,
        'psnr_zoom': psnr_zoom_train,
        'cosine_ffx': cosine_ffx_train,
        'compression_loss_sans_G': compression_loss_sans_G,
        'weighted_rate': weighted_rate,
        'perceptual': perceptual,
        'rate_penalty': rate_penalty,
        'bpp': bpp,
        'n_rate': n_rate,
        'q_rate': q_rate,
        'n_rate_latent': n_rate_latent,
        'q_rate_latent': q_rate_latent,
        'n_rate_hyperlatent': n_rate_hyperlatent,
        'q_rate_hyperlatent': q_rate_hyperlatent,
        }

        model.train()

        logger.info("\n")
        logger.info("=" * 150)
        logger.info("\n")

        
        for idx, (data, bpp) in enumerate(tqdm(train_loader, desc='Train'), 0):

            data = data.to(device, dtype=torch.float)

            storage["bpp"].append(bpp.mean().item())

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
                # if model.step_counter > args.log_interval+1:
                #     logger.warning('Exiting, saving ...')
                #     ckpt_path = utils.save_model(model, optimizers, mean_epoch_loss, epoch, device, args=args, logger=logger)
                #     return model, ckpt_path
                # else:
                    return model, None
            
            # update_performance(args, loss_train, ssim_rec_train, ssim_zoom_train, psnr_rec_train, psnr_zoom_train, cosine_ffx_train, storage)
            update_performance(args, metrics, storage)

            # if model.step_counter % args.log_interval == 1:
            #     print_performance(args, len(train_loader), bpp.mean().item(), storage, loss_train, ssim_rec_train, ssim_zoom_train, psnr_rec_train, psnr_zoom_train, cosine_ffx_train, idx, epoch, model.training)

        # End epoch
        # print_performance(args, len(train_loader), bpp.mean().item(), storage, loss_train, ssim_rec_train, ssim_zoom_train, psnr_rec_train, psnr_zoom_train, cosine_ffx_train, idx, epoch, model.training)
        print_performance(args, len(train_loader), metrics, idx, epoch, model.training)

        # Plot on Tensorboard per Epoch
        # utils.log_summaries(args, train_writer, storage, loss_train, ssim_rec_train, ssim_zoom_train, psnr_rec_train, psnr_zoom_train, cosine_ffx_train, epoch, mode = 'lfw', use_discriminator=model.use_discriminator)
        utils.log_summaries(args, train_writer, metrics, epoch, mode = 'lfw', use_discriminator=model.use_discriminator)

        val_loss, *_ = eval_lfw_jpegai(args, epoch, model, val_loader, device, val_writer)

        if val_loss < best_val_loss:
            logger.info('===>> Loss imporved from {:.3f} to {:.3f}'.format(best_val_loss, val_loss))
            ckpt_path = utils.save_model(model, optimizers, mean_epoch_loss, epoch, device, args=args, logger=logger)
            args.checkpoint = ckpt_path
            best_val_loss = val_loss

        scheduler_amortization.step()
        scheduler_hyperlatent_likelihood.step()
        logger.info("Adjusting learning rate to {:.7f}".format(scheduler_amortization.get_lr()[0]))
        
        logger.info("=" * 150)
        logger.info("\n")
    
        if model.step_counter > args.n_steps:
            break

    with open(os.path.join(args.storage_save, 'storage_{}_{:%Y_%m_%d_%H:%M:%S}.pkl'.format(args.name, datetime.datetime.now())), 'wb') as handle:
        pickle.dump(storage, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    logger.info("=" * 150)
    logger.info("\n")
    logger.info("Loading best checkpoint")

    model.load_checkpoint(args.checkpoint)
    model = model.to(device)

    model.args.norm_loss = False
    eval_lfw_jpegai(args, 0, model, val_loader, device, None)
    eval_lfw_jpegai(args, 0, model, jpeg_loader, device, None, "jpegai")
    logger.info("Training complete. Time elapsed: {:.3f} s. Number of steps: {}".format((time.time()-start_time), model.step_counter))
    
    return model, ckpt_path


def update_performance(args, metrics, store, dataset = "lfw"):
# def update_performance(args, loss, ssim_rec, ssim_zoom, psnr_rec, psnr_zoom, cosine_ffx, store):

    loss = metrics['loss']
    ssim_rec = metrics['ssim_rec']
    ssim_zoom = metrics['ssim_zoom']
    psnr_rec = metrics['psnr_rec']
    psnr_zoom = metrics['psnr_zoom']
    cosine_ffx = metrics['cosine_ffx']

    batch_size = args.batch_size
    if dataset == "jpegai":
        batch_size = 1

    if loss is not None:
        loss.update(store["weighted_compression_loss"][-1], batch_size)

    if args.default_task in args.tasks:
        ssim = store['perceptual rec'][-1]
        ssim_rec.update(ssim, batch_size)

        psnr = store['psnr rec'][-1]
        psnr_rec.update(psnr, batch_size)

        metrics['compression_loss_sans_G'].update(store['compression_loss_sans_G'][-1], batch_size)
        metrics['weighted_rate'].update(store['weighted_rate'][-1], batch_size)
        metrics['perceptual'].update(store['perceptual'][-1], batch_size)
        metrics['rate_penalty'].update(store['rate_penalty'][-1], batch_size)

        metrics['bpp'].update(store['bpp'][-1], batch_size)
        metrics['n_rate'].update(store['n_rate'][-1], batch_size)
        metrics['q_rate'].update(store['q_rate'][-1], batch_size)
        metrics['n_rate_latent'].update(store['n_rate_latent'][-1], batch_size)
        metrics['q_rate_latent'].update(store['q_rate_latent'][-1], batch_size)
        metrics['n_rate_hyperlatent'].update(store['n_rate_hyperlatent'][-1], batch_size)
        metrics['q_rate_hyperlatent'].update(store['q_rate_hyperlatent'][-1], batch_size)

    if "Zoom" in args.tasks or args.test_task:
        ssim = store['perceptual zoom'][-1]
        ssim_zoom.update(ssim, batch_size)

        psnr = store['psnr zoom'][-1]
        psnr_zoom.update(psnr, batch_size)

    if ("FFX" in args.tasks  or args.test_task) and cosine_ffx is not None :
        cosine = store['cosine sim'][-1]
        cosine_ffx.update(cosine, batch_size)


def print_performance(args, data_size, metrics, idx, epoch, training):
# def print_performance(args, data_size, bpp, storage, loss, ssim_rec, ssim_zoom, psnr_rec, psnr_zoom, cosine_ffx, idx, epoch, training):
    
    loss = metrics['loss']
    ssim_rec = metrics['ssim_rec']
    ssim_zoom = metrics['ssim_zoom']
    psnr_rec = metrics['psnr_rec']
    psnr_zoom = metrics['psnr_zoom']
    cosine_ffx = metrics['cosine_ffx']
 
    if training:
        display = '\nEPOCH: [{0}][{1}/{2}]'.format(epoch, idx, data_size)
    else:
        display = '\nValidation: [{0}/{1}]'.format(idx, data_size)

    if loss is not None:
        display += '\tloss {loss.val:.3f} ({loss.avg:.3f})'.format(loss = loss)

    if args.default_task in args.tasks:
        display += '\tpsnr_rec {psnr_rec.val:.3f} ({psnr_rec.avg:.3f})\t ssim_rec {ssim_rec.val:.3f} ({ssim_rec.avg:.3f})'.format(psnr_rec = psnr_rec, ssim_rec = ssim_rec)
    
    if "Zoom" in args.tasks or args.test_task:
        display += '\tpsnr_zoom {psnr_zoom.val:.3f} ({psnr_zoom.avg:.3f})\t ssim_zoom {ssim_zoom.val:.3f} ({ssim_zoom.avg:.3f})'.format(psnr_zoom = psnr_zoom, ssim_zoom = ssim_zoom)
          
    if ("FFX" in args.tasks or args.test_task) and cosine_ffx is not None:
        display += '\tcosine_ffx {cosine_ffx.val:.3f} ({cosine_ffx.avg:.3f})'.format(cosine_ffx = cosine_ffx)
    
    if args.default_task in args.tasks:
        display += '\n'
        display += "Rate-Distortion:\n"
        display += "Weighted Rate: {:.3f} ({:.3f}) | Perceptual: {:.3f} ({:.3f}) | Rate Penalty: {:.3f} ({:.3f})".format(metrics['weighted_rate'].val, metrics['weighted_rate'].avg, 
                                            metrics['perceptual'].val, metrics['perceptual'].avg, metrics['rate_penalty'].val, metrics['rate_penalty'].avg)

        # display += "Weighted Rate: {:.3f} | Perceptual: {:.3f} | Rate Penalty: {:.3f}".format(storage['weighted_rate'][-1], 
        #                                     storage['perceptual'][-1], storage['rate_penalty'][-1])

        display += '\n'
        display += "Rate Breakdown:\n"
        display += "avg. original bpp: {:.3f} ({:.3f}) | n_bpp (total): {:.3f} ({:.3f}) | q_bpp (total): {:.3f} ({:.3f}) | n_bpp (latent): {:.3f} ({:.3f}) | q_bpp (latent): {:.3f} ({:.3f}) | n_bpp (hyp-latent): {:.3f} ({:.3f}) | q_bpp (hyp-latent): {:.3f} ({:.3f})".format(metrics['bpp'].val, metrics['bpp'].avg, metrics['n_rate'].val, metrics['n_rate'].avg, metrics['q_rate'].val, metrics['q_rate'].avg, 
                metrics['n_rate_latent'].val, metrics['n_rate_latent'].avg, metrics['q_rate_latent'].val, metrics['q_rate_latent'].avg, metrics['n_rate_hyperlatent'].val, metrics['n_rate_hyperlatent'].avg, metrics['q_rate_hyperlatent'].val, metrics['q_rate_hyperlatent'].avg)

        # display += "avg. original bpp: {:.3f} | n_bpp (total): {:.3f} | q_bpp (total): {:.3f} | n_bpp (latent): {:.3f} | q_bpp (latent): {:.3f} | n_bpp (hyp-latent): {:.3f} | q_bpp (hyp-latent): {:.3f}".format(bpp, storage['n_rate'][-1], storage['q_rate'][-1], 
        #         storage['n_rate_latent'][-1], storage['q_rate_latent'][-1], storage['n_rate_hyperlatent'][-1], storage['q_rate_hyperlatent'][-1])
        
    display += '\n'
    
    logger.info(display)


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
    general.add_argument("-os_gpu", "--os_gpu", type=int, default=0, help="GPU ID.")
    general.add_argument("-log_intv", "--log_interval", type=int, default=hific_args.log_interval, help="Number of steps between logs.")
    general.add_argument("-save_intv", "--save_interval", type=int, default=hific_args.save_interval, help="Number of steps between checkpoints.")
    general.add_argument("-multigpu", "--multigpu", help="Toggle data parallel capability using torch DataParallel", action="store_true")
    general.add_argument("-norm", "--normalize_input_image", help="Normalize input images to [-1,1]", action="store_true")
    general.add_argument("-lnorm", "--norm_loss", help="Normalize the global loss", action="store_true")
    general.add_argument("-eval", "--evaluate", help="Evaluate the framework before training", action="store_true")
    general.add_argument("-tt", "--test_task", help="Test task only, train a new task on freezed latent", action="store_true")
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

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cmd_args.os_gpu)

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
    storage_val = defaultdict(list)
    storage_test = defaultdict(list)
    logger = utils.logger_setup(logpath=os.path.join(args.snapshot, 'logs.txt'), filepath=os.path.abspath(__file__))

    model = create_model(args, device, logger, storage, storage_val, storage_test)
    model = model.to(device)
    
    multi_gpu = torch.cuda.device_count() > 1 if torch.cuda.is_available() else False
    if multi_gpu:
        model = nn.DataParallel(model)


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

    val_loader = datasets.get_dataloaders(args.dataset,
                                root=args.dataset_path,
                                batch_size=args.batch_size,
                                logger=logger,
                                split='test',
                                shuffle=False,
                                normalize=args.normalize_input_image)

    train_loader = datasets.get_dataloaders(args.dataset,
                                root=args.dataset_path,
                                batch_size=args.batch_size,
                                logger=logger,
                                split='train',
                                shuffle=True,
                                normalize=args.normalize_input_image)


    jpeg_loader = datasets.get_dataloaders('evaluation', root=args.image_dir, batch_size=1,
                                           logger=logger, shuffle=False, normalize=args.normalize_input_image)


    args.n_data = len(train_loader.dataset)
    args.image_dims = train_loader.dataset.image_dims
    logger.info("=" * 50)
    logger.info('Training elements: {}'.format(args.n_data))
    logger.info('Testing elements: {}'.format(len(val_loader.dataset)))
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
    model, ckpt_path = train(args, model, train_loader, val_loader, jpeg_loader, device, logger, optimizers=optimizers)

    """
    TODO
    Generate metrics
    """
