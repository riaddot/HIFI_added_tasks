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

from src.pruning.pruning_utils import *
from src.pruning.pruning_analysis import verify_pruned_weights

from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

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

def optimize_compression_loss(compression_loss, amortization_opt, hyperlatent_likelihood_opt, gammas_opt = None):
    compression_loss.backward()
    amortization_opt.step()
    hyperlatent_likelihood_opt.step()
    amortization_opt.zero_grad()
    hyperlatent_likelihood_opt.zero_grad()

    if gammas_opt is not None:
        gammas_opt.step()
        gammas_opt.zero_grad()
        


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
    gamma1 = utils.AverageMeter()
    gamma2 = utils.AverageMeter()
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
    'gamma1' : gamma1,
    'gamma2' : gamma2,
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

                losses, intermediates = model(data, return_intermediates=True, writeout=True, upsample = False, )
                storage["bpp"].append(bpp.mean().item())
                # update_performance(args, val_loss, ssim_rec_val, ssim_zoom_val, psnr_rec_val, psnr_zoom_val, cosine_ffx_val, storage)
                update_performance(args, metrics, storage)

                # if idx % 100 == 0:
                #     print_performance(args, len(val_loader), bpp.mean().item(), storage, val_loss, ssim_rec_val, ssim_zoom_val, psnr_rec_val, psnr_zoom_val, cosine_ffx_val, idx, epoch, model.training)

        elif dataset.lower() == "ff++":
            
            storage = model.storage_val
            model.model_mode = ModelModes.EVALUATION

            for idx, (data, bpp, filename) in enumerate(tqdm(val_loader, desc='Val'), 0):
                
                data = data.to(device, dtype=torch.float)
                data_prime = data
                upsample = True
                for i in range(args.double_compression):

                    outputs, intermediates = model(data, return_intermediates=True, writeout=True, upsample = False, x_hr_prime = data_prime)

                    if "Zoom" in args.tasks or args.test_task:
                        reconst, reconst_zoom = outputs
                    else:
                        reconst = outputs
                        
                    data = reconst

                    if args.normalize_input_image:
                        data = (data - 0.5) * 2

                    upsample = False

                if "Zoom" in args.tasks or args.test_task:
                    reconst, reconst_zoom = outputs

                    for im_idx in range(args.batch_size):
                        splited = filename[im_idx].split("/")
                        fname = splited[-2] + "_" + splited[-1].split(".")[0]
                        fname=os.path.join(args.figures_save, 'zoom_{}_comp{}.png'.format(fname, i))
                        # logger.info(fname)
                        save_image(fname, reconst_zoom[im_idx])

                else:
                    reconst = outputs

                    for im_idx in range(len(data)):
                        splited = filename[im_idx].split("/")
                        fname = splited[-2] + "_" + splited[-1].split(".")[0]
                        fname=os.path.join(args.figures_save, 'recon_{}_comp{}.png'.format(fname, i))
                        # logger.info(fname)
                        save_image(fname, reconst[im_idx])

                    metrics['loss'] = None
                    storage["bpp"].append(bpp.mean().item())
                    # update_performance(args, val_loss, ssim_rec_val, ssim_zoom_val, psnr_rec_val, psnr_zoom_val, cosine_ffx_val, storage)
                    update_performance(args, metrics, storage)

        elif dataset == "jpegai":
            
            storage = model.storage_test
            model.model_mode = ModelModes.EVALUATION

            device = torch.device("cpu")
            model.to(device)
            
            for idx, (data, bpp, filename) in enumerate(tqdm(val_loader, desc='Val'), 0):
                data = data.to(device, dtype=torch.float)
                data_prime = data
                upsample = True
                for i in range(args.double_compression):
                    outputs, intermediates = model(data, return_intermediates=True, writeout=True, upsample = upsample, x_hr_prime = data_prime)
                    if "Zoom" in args.tasks or args.test_task:
                        reconst, reconst_zoom = outputs
                        
                        fname=os.path.join(args.figures_save, 'zoom_{}_comp{}.png'.format(filename[0], i))
                        save_image(fname, reconst_zoom)
                    else:
                        reconst = outputs

                    fname=os.path.join(args.figures_save, 'recon_{}_comp{}.png'.format(filename[0], i))
                    logger.info(fname)
                    save_image(fname, reconst)

                    storage["bpp"].append(bpp.mean().item())
                    # update_performance(args, None, ssim_rec_val, ssim_zoom_val, psnr_rec_val, psnr_zoom_val, None, storage)
                    metrics['loss'] = None
                    
                    metrics['cosine_ffx'] = None
                    update_performance(args, metrics, storage, "jpegai")

                    print_performance(args, len(val_loader), metrics, idx, epoch, model.training)

                    data = reconst

                    if args.normalize_input_image:
                        data = (data - 0.5) * 2
                    upsample = False

            model.model_mode = ModelModes.TRAINING

    
    # print_performance(args, len(val_loader), bpp.mean().item(), storage, val_loss, ssim_rec_val, ssim_zoom_val, psnr_rec_val, psnr_zoom_val, cosine_ffx_val, idx, epoch, model.training)
    print_performance(args, len(val_loader), metrics, idx, epoch, model.training)

    if writer is not None:
        # utils.log_summaries(args, writer, storage, val_loss, ssim_rec_val, ssim_zoom_val, psnr_rec_val, psnr_zoom_val, cosine_ffx_val, epoch, mode = 'lfw', use_discriminator=model.use_discriminator)
        utils.log_summaries(args, writer, metrics, epoch, mode = 'lfw', use_discriminator=model.use_discriminator)
        
    return val_loss.avg, ssim_rec_val.avg, ssim_zoom_val.avg, cosine_ffx_val.avg, weighted_rate.avg



def process_batch_hific(args, model, data, device):
    """
    Function to process a single batch of data, perform double compression, and compute residuals.
    
    Args:
        args: Argument parser object containing configurations.
        model: Compression model instance.
        data: A single batch of input data (tensor).
        device: Device to run the model on (e.g., 'cuda' or 'cpu').

    Returns:
        decompressed_images (torch.Tensor): Tensor of decompressed images after double compression.
        residual_maps (torch.Tensor): Tensor of residual maps between input and decompressed images.
    """
    model.eval()
    data = data.to(device, dtype=torch.float)
    data_prime = data  # Preserve the original data for residual computation
    
    model.model_mode = ModelModes.EVALUATION

    with torch.no_grad():
        # Perform double compression
        for i in range(args.double_compression):
            outputs, bpp = model(
                data,
                return_intermediates=True,
                writeout=False,
                upsample=False,
                x_hr_prime=data_prime
            )

            # outputs = intermediates.reconstruction
            decompressed = outputs  # Assuming `outputs` are the decompressed images
            data = decompressed  # Feed decompressed images back for second compression

            # Normalize if required
            if args.normalize_input_image:
                data = (data - 0.5) * 2

        # Compute residual maps
        residual = (data_prime * 0.5 + 0.5) - (data * 0.5 + 0.5)

    return residual, data


def train_svm_with_hific(args, hific_model, classifier, train_loader, val_loader, fakes_test_set, optimizer, epochs, device, logger):
    """
    Train an SVM classifier using residuals from the HiFiC model.
    
    Args:
        args: Arguments containing model configurations.
        hific_model: HiFiC compression model.
        train_loader: DataLoader for training data (contains real and fake images).
        val_loader: DataLoader for validation data.
        device: Device to run the HiFiC model ('cuda' or 'cpu').
        epochs: Number of epochs to train the SVM.
    
    Returns:
        classifier: Trained SVM classifier.
    """

    train_writer = SummaryWriter(os.path.join(args.tensorboard_runs, 'train'))
    val_writer = SummaryWriter(os.path.join(args.tensorboard_runs, 'val'))
    # jpegai_writer = SummaryWriter(os.path.join(args.tensorboard_runs, 'jpegai'))


    hific_model.eval()
    
    train_accuracy = utils.AverageMeter() 
    
    train_acc_log = []
    val_acc_log = []

    # Collect training data
    logger.info("Generating training residuals...")
    
    for epoch in range(epochs):

        storage, metrics = init_train_config()

        model.to(device)
        model.train()

        logger.info("\n")
        logger.info("=" * 150)
        logger.info("\n")


        for batch_idx, (images, labels, *_) in enumerate(tqdm(train_loader)):
            # Get residuals from HiFiC
            residuals, decompressed = process_batch_hific(args, hific_model, images, device)

            if args.fft:
                residuals = torch.abs(torch.fft.fft2(residuals, norm="ortho")).detach()

            if args.classifier == "svm" or args.classifier == "kmeans":
                classifier.fit(residuals.view(residuals.size(0), -1).cpu().numpy(), labels.cpu().numpy())  # You can call fit on mini-batches
                pred = classifier.predict(residuals.view(residuals.size(0), -1).cpu().numpy())
            
            elif args.classifier == "cnn":
                # Train CNN
                classifier = classifier.to(device)  # Move model to the device
                residuals = residuals.to(device)    # Move input tensor to the device
                labels = labels.to(device)          # Move labels tensor to the device

                classifier.train()
                optimizer.zero_grad()
                outputs = classifier(residuals)
                outputs.to(device)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                pred = torch.argmax(outputs, dim=1).cpu().numpy()

            # Calculate batch accuracy
            # update_performance(args, metrics, storage, dataset = args.dataset)
            train_accuracy.update(accuracy_score(labels.cpu().numpy(), pred), len(images))

            if (batch_idx+1) % 100 == 0:
                # print_performance(args, len(train_loader), metrics, batch_idx, epoch, True)
                logger.info(f"EPOCH [{epoch}][{batch_idx+1}/{len(train_loader)}]: Train Accuracy: {train_accuracy.avg * 100:.2f}%")

        # print_performance(args, len(train_loader), metrics, batch_idx, epoch, True)
        logger.info(f"Train Accuracy: {train_accuracy.avg * 100:.2f}%")

        # Evaluate on validation set
        logger.info("Evaluating on validation set...")
        val_acc = evaluate(args, hific_model, classifier, val_loader, device, logger)
        logger.info("Evaluating on Test set...")
        for name, test_loader in fakes_test_set.items():
            test_acc = evaluate(args, hific_model, classifier, test_loader, device, logger, fake_type=name)

        train_acc_log.append(train_accuracy.avg)
        val_acc_log.append(val_acc.avg)

        # utils.log_summaries(args, writer, metrics, epoch, mode = 'lfw', use_discriminator=model.use_discriminator)
        train_writer.add_scalar('{}/Accuracy'.format(args.dataset), train_accuracy.avg, epoch)
        val_writer.add_scalar('{}/Accuracy'.format(args.dataset), val_acc.avg, epoch)
        

    # Plot Accuracy vs. Epochs
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_acc_log, marker='o', label="Train")
    plt.plot(range(1, epochs + 1), val_acc_log, marker='o', label="Validation")
    plt.title(f"Regime: {args.regime}, DC: {args.double_compression}, FakeType: {args.fake_type}" + "SVM" if args.classifier == "svm" else "CNN" + "FFT" if args.fft else "Spatial")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.xticks(range(1, epochs + 1))
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.figures_save, "plot.png"))
    np.save(os.path.join(args.storage_save, "performances.npy"), np.array(accuracies))

    return classifier, accuracies

def init_train_config():
    storage = model.storage_train

    loss_train = utils.AverageMeter() 
    accuracy_train = utils.AverageMeter() 
    ssim_rec_train = utils.AverageMeter()
    ssim_zoom_train = utils.AverageMeter()
    psnr_rec_train = utils.AverageMeter()
    psnr_zoom_train = utils.AverageMeter()
    cosine_ffx_train = utils.AverageMeter()
    compression_loss_sans_G = utils.AverageMeter()
    weighted_rate = utils.AverageMeter()
    perceptual = utils.AverageMeter()
    rate_penalty = utils.AverageMeter()
    gamma1 = utils.AverageMeter()
    gamma2 = utils.AverageMeter()
    bpp = utils.AverageMeter()
    n_rate = utils.AverageMeter()
    q_rate = utils.AverageMeter()
    n_rate_latent = utils.AverageMeter()
    q_rate_latent = utils.AverageMeter()
    n_rate_hyperlatent = utils.AverageMeter()
    q_rate_hyperlatent = utils.AverageMeter()


    metrics = {
        'loss': loss_train,
        'accuracy': accuracy_train,
        'ssim_rec': ssim_rec_train,
        'ssim_zoom': ssim_zoom_train,
        'psnr_rec': psnr_rec_train,
        'psnr_zoom': psnr_zoom_train,
        'cosine_ffx': cosine_ffx_train,
        'compression_loss_sans_G': compression_loss_sans_G,
        'weighted_rate': weighted_rate,
        'perceptual': perceptual,
        'rate_penalty': rate_penalty,
        'gamma1': gamma1,
        'gamma2': gamma2,
        'bpp': bpp,
        'n_rate': n_rate,
        'q_rate': q_rate,
        'n_rate_latent': n_rate_latent,
        'q_rate_latent': q_rate_latent,
        'n_rate_hyperlatent': n_rate_hyperlatent,
        'q_rate_hyperlatent': q_rate_hyperlatent,
        }
    
    return storage,metrics

def evaluate(args, hific_model, classifier, val_loader, device, logger, fake_type = None):
    
    val_acc = utils.AverageMeter()
    storage, metrics = init_val_config()
    
    for (images, labels, *_) in val_loader:
        residuals, _ = process_batch_hific(args, hific_model, images, device)
        if args.fft:
            residuals = torch.abs(torch.fft.fft2(residuals, norm="ortho")).detach()

        if args.classifier == "svm" or args.classifier == "kmeans":
            batch_residuals = residuals.view(residuals.size(0), -1).cpu().numpy()  # Flatten residuals
            batch_labels = labels.cpu().numpy()
            batch_predictions = classifier.predict(batch_residuals)

        elif args.classifier == "cnn":
            classifier.eval()
            batch_residuals = residuals.to(device)
            batch_labels = labels.cpu().numpy()
            with torch.no_grad():
                outputs = classifier(batch_residuals)
                batch_predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # Calculate batch accuracy
        batch_accuracy = accuracy_score(batch_labels, batch_predictions)
        # batch_accuracies.append(batch_accuracy)

        # update_performance(args, metrics, storage)
        val_acc.update(batch_accuracy, len(images))

        # Compute overall accuracy as the mean of batch accuracies
        # accuracy = np.mean(batch_accuracies)

    logger.info(f"{fake_type + " " if fake_type else " "}Validation Accuracy: {val_acc.avg * 100:.2f}%")

    return val_acc

def init_val_config():
    val_loss = utils.AverageMeter()
    val_acc = utils.AverageMeter()
    ssim_rec_val = utils.AverageMeter()
    ssim_zoom_val = utils.AverageMeter()
    psnr_rec_val = utils.AverageMeter()
    psnr_zoom_val = utils.AverageMeter()
    cosine_ffx_val = utils.AverageMeter()

    compression_loss_sans_G = utils.AverageMeter()
    weighted_rate = utils.AverageMeter()
    perceptual = utils.AverageMeter()
    rate_penalty = utils.AverageMeter()
    gamma1 = utils.AverageMeter()
    gamma2 = utils.AverageMeter()
    bpp = utils.AverageMeter()
    n_rate = utils.AverageMeter()
    q_rate = utils.AverageMeter()
    n_rate_latent = utils.AverageMeter()
    q_rate_latent = utils.AverageMeter()
    n_rate_hyperlatent = utils.AverageMeter()
    q_rate_hyperlatent = utils.AverageMeter()


    metrics = {
    'loss': val_loss,
    'val_acc': val_acc,
    'ssim_rec': ssim_rec_val,
    'ssim_zoom': ssim_zoom_val,
    'psnr_rec': psnr_rec_val,
    'psnr_zoom': psnr_zoom_val,
    'cosine_ffx': cosine_ffx_val,
    'compression_loss_sans_G' : compression_loss_sans_G,
    'weighted_rate' : weighted_rate,
    'perceptual' : perceptual,
    'rate_penalty' : rate_penalty,
    'gamma1' : gamma1,
    'gamma2' : gamma2,
    'bpp' : bpp,
    'n_rate' : n_rate,
    'q_rate' : q_rate,
    'n_rate_latent' : n_rate_latent,
    'q_rate_latent' : q_rate_latent,
    'n_rate_hyperlatent' : n_rate_hyperlatent,
    'q_rate_hyperlatent' : q_rate_hyperlatent
    }
    
    model.eval()

    storage = model.storage_val
    return storage, metrics

def save_images(tensor, save_path):
    """
    Save a tensor as an image.
    
    Args:
        tensor: PyTorch tensor representing the image (C, H, W).
        save_path: Directory path to save the image.
        file_name: Name of the file to save.
    """
    # Convert tensor to numpy and scale to 0-255 if necessary
    image = tensor.detach().cpu().numpy()
    if image.ndim == 3:  # Handle (C, H, W)
        image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, C)
    image = (image * 255).clip(0, 255).astype(np.uint8)

    # Save as image
    Image.fromarray(image).save(os.path.join(save_path))


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



def run_pruning_flow(args, encoders, encoder_name="encoder_0.45_bpp", prune_rate=0.01, input_size=(1, 3, 256, 256), orig_flops = 0, orig_param = 0, logger = None):
    logger.info("pruning '{}%' of paramters during this iteration".format(prune_rate))
    pruned_filters_info = prune_filters_by_l1_norm(encoders, logger, prune_rate=prune_rate)

    if args.common_filters:
        # Analyze pruned filters across encoders to find common ones
        pruned_filters_info = analyze_pruned_filters_across_encoders(pruned_filters_info, encoder_name)


    original_encoder, pruned_encoder, flops_ratio, param_ratio = prune_and_compare(encoder_name, encoders, pruned_filters_info, input_size, orig_flops, orig_param, logger)

    verify_pruned_weights(original_encoder, pruned_encoder, pruned_filters_info[encoder_name], logger)

    # Optionally return for further inspection or saving
    return pruned_encoder, flops_ratio


def evalaute_model(args, model, val_loader, device, logger):
    model = model.to(device)

    args.norm_loss = False

    logger.info("FF++ evaluation")
    val_loss, *_ = eval_lfw_jpegai(args, 0, model, val_loader, device, None, dataset=args.dataset)

    logger.info("=" * 150)
    logger.info("\n")
    logger.info("End of Evaluation")
    return model


def update_performance(args, metrics, store, dataset = "lfw"):
# def update_performance(args, loss, ssim_rec, ssim_zoom, psnr_rec, psnr_zoom, cosine_ffx, store):

    loss = metrics['loss']
    accuracy = metrics['accuracy']
    ssim_rec = metrics['ssim_rec']
    ssim_zoom = metrics['ssim_zoom']
    psnr_rec = metrics['psnr_rec']
    psnr_zoom = metrics['psnr_zoom']
    cosine_ffx = metrics['cosine_ffx']

    batch_size = args.batch_size
    if dataset == "jpegai":
        batch_size = 1

    if dataset == "ff++":
        accuracy.update(store["acacuracy"][-1], batch_size)

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
        if args.auto_norm:
            metrics['gamma1'].update(store['gamma1'][-1], batch_size)
            metrics['gamma2'].update(store['gamma2'][-1], batch_size)
        elif not(args.target_rate_loss):
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
        if args.auto_norm:
            display += "Weighted Rate: {:.3f} ({:.3f}) | Perceptual: {:.3f} ({:.3f}) | Gamma_1: {:.3f} ({:.3f}) | Gamma_2: {:.3f} ({:.3f})".format(metrics['weighted_rate'].val, metrics['weighted_rate'].avg, 
                                                metrics['perceptual'].val, metrics['perceptual'].avg, metrics['gamma1'].val, metrics['gamma1'].avg, metrics['gamma2'].val, metrics['gamma2'].avg)
        else:
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




# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 256, 2)  # Assuming input size of 3x32x32
        # self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 32 * 256)
        x = self.fc1(x)
        # x = self.softmax(x)
        return x


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
    general.add_argument("-target_rate_loss", "--target_rate_loss", help="Track a target compression rate instead of using a control term", action="store_true")
    general.add_argument("-auto_norm", "--auto_norm", help="Automatic Normalize the global loss between tasks and compression rate", action="store_true")
    general.add_argument("-dc", "--double_compression", type=int, default=hific_args.double_compression, help="DoubleCompression : Compress the image many times ")
    general.add_argument("-adp", "--adaptative", choices=('exp','linear'), default=None, help="choose the adaptative function for the lambda for controlling the compression term")
    general.add_argument("-eval", "--evaluate", help="Evaluate the framework before training", action="store_true")
    # general.add_argument("-tckpt", "--target_ckpt", help="Operate on the model located on this ckpt path")
    general.add_argument("-tt", "--test_task", help="Test task only, train a new task on freezed latent", action="store_true")
    general.add_argument('-bs', '--batch_size', type=int, default=hific_args.batch_size, help='input batch size for training')
    general.add_argument('--save', type=str, default='experiments', help='Parent directory for stored information (checkpoints, logs, etc.)')
    general.add_argument("-lt", "--likelihood_type", choices=('gaussian', 'logistic'), default='gaussian', help="Likelihood model for latents.")
    general.add_argument("-force_gpu", "--force_set_gpu", help="Set GPU to given ID", action="store_true")
    general.add_argument("-LMM", "--use_latent_mixture_model", help="Use latent mixture model as latent entropy model.", action="store_true")

    # Pruning Settings
    general.add_argument("-prune", "--prune", help="enable pruning", action="store_true")
    general.add_argument("-pr", "--pruning_ratio", type=float, default = 0.5, help="enable pruning")
    general.add_argument("-common_filters", "--common_filters", help="whether prune the common filters among the encoders of 3 different compression level", action="store_true")

    # DeepFake Settings
    general.add_argument("-fake_type", "--fake_type", type=str, default=hific_args.fake_type, help="Choose the fake type")
    general.add_argument("-dataset_type", "--dataset_type", type=str, default=hific_args.dataset_type, help="Wether using fake or original iamges")
    general.add_argument("-fft", "--fft", help="process on frequency domain", action="store_true")
    general.add_argument("-cls", "--classifier", help="choose the classifier type", default = "svm")

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
    
    arch_args.add_argument('-t', '--tasks', choices=['HiFiC', 'Zoom','FFX'], nargs='+', default=[hific_args.default_task], help="Choose which task to add into the MTL framework")


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


    if not args.prune:
        amortization_parameters = itertools.chain.from_iterable(
            [am.parameters() for am in model.amortization_models])

        hyperlatent_likelihood_parameters = model.Hyperprior.hyperlatent_likelihood.parameters()

        amortization_opt = torch.optim.Adam(amortization_parameters,
            lr=args.learning_rate)
        
        hyperlatent_likelihood_opt = torch.optim.Adam(hyperlatent_likelihood_parameters, 
            lr=args.learning_rate)
        
        optimizers = dict(amort=amortization_opt, hyper=hyperlatent_likelihood_opt)

    else:
        optimizers = None

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

    if args.evaluate:
        ff_loader = datasets.get_dataloaders(dataset=args.dataset, 
                                                root="/home/bellelbn/DL/datasets/Faceforensics", 
                                                batch_size=args.batch_size, 
                                                dataset_type=args.dataset_type, 
                                                fake_type= args.fake_type, 
                                                frames = args.nframes,
                                                normalize=args.normalize_input_image,
                                                logger=logger)
        

        # jpeg_loader = datasets.get_dataloaders('evaluation', root=args.image_dir, batch_size=1,
        #                                        logger=logger, shuffle=False, normalize=args.normalize_input_image)

        
        args.n_data = len(ff_loader.dataset)
        logger.info("=" * 50)
        logger.info('ffloader elements: {}'.format(args.n_data))
        # logger.info('JPEGAI elements: {}'.format(len(jpeg_loader.dataset)))
        logger.info('Input Dimensions: {}'.format(args.image_dims))
        logger.info('Batch size: {}'.format(args.batch_size))
        logger.info('Optimizers: {}'.format(optimizers))
        logger.info('Using device {}'.format(device))
        logger.info("=" * 50)


        metadata = dict((n, getattr(args, n)) for n in dir(args) if not (n.startswith('__') or 'logger' in n))
        logger.info(metadata)

        evalaute_model(args, model, ff_loader, device, logger)


    else:
        from torch.utils.data import DataLoader, ConcatDataset, random_split

        # Load original data
        original_loader = datasets.get_dataloaders(
            dataset="ff++",
            root="/home/bellelbn/DL/datasets/Faceforensics",
            batch_size=args.batch_size,
            dataset_type="original",
            frames=args.nframes,
            normalize=args.normalize_input_image,
            logger=logger
        )

        # Load fake data
        fake_loader = datasets.get_dataloaders(
            dataset="ff++",
            root="/home/bellelbn/DL/datasets/Faceforensics",
            batch_size=args.batch_size,
            dataset_type="fake",
            fake_type=args.fake_type,
            frames=args.nframes,
            normalize=args.normalize_input_image,
            logger=logger
        )

        # Combine datasets
        original_dataset = original_loader.dataset
        fake_dataset = fake_loader.dataset

        logger.info("Real data size : {}".format(len(original_dataset)))
        logger.info("Fake data size : {}".format(len(fake_dataset)))

        combined_dataset = ConcatDataset([original_dataset, fake_dataset])

        # Unified DataLoader
        combined_loader = DataLoader(
            combined_dataset,
            batch_size=args.batch_size,
            shuffle=True,  # Enable shuffling
            pin_memory=torch.cuda.is_available()
        )

        # Split the combined dataset
        train_length = int(0.8 * len(combined_dataset))
        val_length = int(0.1 * len(combined_dataset))
        test_length = len(combined_dataset) - train_length - val_length

        train_dataset, val_dataset, test_dataset = random_split(
            combined_dataset,
            [train_length, val_length, test_length],
            generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
        )


        fakes_test_set = {}

        fakes_type = ["DeepFakeDetection", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
        
        for fake_type in fakes_type:
            if fake_type == args.fake_type:
                continue
            
            logger.info("Loading {} for testing".format(fake_type))

            fakes_test_set[fake_type] = datasets.get_dataloaders(dataset="ff++", 
                                                            root="/home/bellelbn/DL/datasets/Faceforensics", 
                                                            batch_size=args.batch_size, 
                                                            dataset_type="fake", 
                                                            fake_type=fake_type, 
                                                            frames=args.nframes, 
                                                            normalize=args.normalize_input_image, 
                                                            logger=logger
                                                            )
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)


        # Initialize SVM classifier
        # svm = SVC(kernel='linear', probability=True)  # Linear kernel for simplicity

        if args.classifier == "svm":
            from sklearn.svm import SVC
            classifier = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3)
            optimizer = None

        elif args.classifier == "kmeans":
            from sklearn.cluster import KMeans
            classifier = KMeans(n_clusters=2, random_state=0, n_init="auto")
            optimizer = None

        elif args.classifier == "cnn":
            import torch.optim as optim
            
            classifier = SimpleCNN().to(device)
            criterion = nn.CrossEntropyLoss().to(device)
            optimizer = optim.Adam(classifier.parameters(), lr=0.0001)

        logger.info(f"Using {args.classifier.upper()} classifier")

        epochs = 20
        # logger.info(f"Images shape: {images.shape}, Labels: {labels}, BPP: {bpp}")
        classifier, accuracies = train_svm_with_hific(args, model, classifier, train_loader, val_loader, fakes_test_set, optimizer, epochs, device, logger)


