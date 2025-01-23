import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import copy
from network.encoder import Encoder
from normalisation.channel import ChannelNorm2D
from tqdm import tqdm
import time

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os 


# def apply_physical_pruning(original_encoder, pruned_filters_info):
#     """
#     Apply physical pruning to the encoder based on the pruned filters info.
#     Cascade pruning to subsequent layers (adjust input channels of the next layer).
#     """
#     # Clone the original encoder so we don't modify it
#     pruned_encoder = copy.deepcopy(original_encoder)
#     previous_pruned_filters = None  # Track pruned output channels from the previous layer

#     for layer_name, pruned_info in pruned_filters_info.items():
#         # Access the Conv2d layer by traversing the model hierarchy
#         print(layer_name)
#         modules = layer_name.split('.')
#         conv_layer = pruned_encoder
#         for mod in modules:
#             conv_layer = getattr(conv_layer, mod)

#         # Get the pruned indices
#         total_filters = conv_layer.weight.size(1)
#         pruned_indices = (
#             pruned_info['pruned_indices'].cpu().numpy()
#             if isinstance(pruned_info['pruned_indices'], torch.Tensor)
#             else np.array(pruned_info['pruned_indices'])
#         )
#         remaining_filter_indices = np.setdiff1d(np.arange(total_filters), pruned_indices)

#         if len(remaining_filter_indices) <= 0:
#             raise ValueError(f"All filters pruned for layer {layer_name}. This is not allowed.")

#         # Prune the Conv2d layer weights
#         original_weights = conv_layer.weight.data.cpu().numpy()
#         original_bias = conv_layer.bias.data.cpu().numpy() if conv_layer.bias is not None else None

#         print(len(remaining_filter_indices))
#         new_out_channels = len(remaining_filter_indices)
#         new_weights = original_weights[:, remaining_filter_indices, :, :]
#         new_bias = original_bias[remaining_filter_indices] if original_bias is not None else None

#         # Cascade pruning to input channels
#         if previous_pruned_filters is not None:
#             new_weights = new_weights[previous_pruned_filters, :, :, :]

#         # Recreate Conv2d layer with pruned weights
#         new_layer = nn.ConvTranspose2d(
#             in_channels=new_weights.shape[0],
#             out_channels=new_out_channels,
#             kernel_size=conv_layer.kernel_size,
#             stride=conv_layer.stride,
#             padding=conv_layer.padding,
#             dilation=conv_layer.dilation,
#             groups=conv_layer.groups,
#             output_padding=conv_layer.output_padding,
#             bias=(new_bias is not None),
#             padding_mode=conv_layer.padding_mode,
#         )
#         new_layer.weight.data = torch.from_numpy(new_weights).to(conv_layer.weight.device)
#         if new_bias is not None:
#             new_layer.bias.data = torch.from_numpy(new_bias).to(conv_layer.bias.device)

#         # Replace the layer
#         parent_module = pruned_encoder
#         for mod in modules[:-1]:
#             parent_module = getattr(parent_module, mod)
#         setattr(parent_module, modules[-1], new_layer)

#         # Adjust normalization layers if applicable
#         norm_layer = parent_module[1] #if len(parent_module) > 2 else None
#         # if isinstance(norm_layer, ChannelNorm2D):
#         if hasattr(norm_layer, 'beta'):
#             norm_layer.beta = nn.Parameter(norm_layer.beta.data[:, remaining_filter_indices])
#         if hasattr(norm_layer, 'gamma'):
#             norm_layer.gamma = nn.Parameter(norm_layer.gamma.data[:, remaining_filter_indices])    

#         # Update previous_pruned_filters
#         previous_pruned_filters = remaining_filter_indices

#     # Handle `conv_block_out` separately
#     conv_block_out_layer = pruned_encoder.conv_block_out[1]  # Access Conv2d in `conv_block_out`
#     if isinstance(conv_block_out_layer, nn.Conv2d) and previous_pruned_filters is not None:
#         original_weights = conv_block_out_layer.weight.data.cpu().numpy()
#         original_bias = conv_block_out_layer.bias.data.cpu().numpy() if conv_block_out_layer.bias is not None else None

#         # Adjust input channels to match pruned filters from the previous layer
#         new_weights = original_weights[:, previous_pruned_filters, :, :]
#         # Create a new Conv2d layer with adjusted input channels but the same output channels
#         final_layer = nn.Conv2d(
#             in_channels=len(previous_pruned_filters),
#             out_channels=conv_block_out_layer.out_channels,
#             kernel_size=conv_block_out_layer.kernel_size,
#             stride=conv_block_out_layer.stride,
#             padding=conv_block_out_layer.padding,
#             dilation=conv_block_out_layer.dilation,
#             groups=conv_block_out_layer.groups,
#             bias=(original_bias is not None),
#             padding_mode=conv_block_out_layer.padding_mode,
#         )
#         final_layer.weight.data = torch.from_numpy(new_weights).to(conv_block_out_layer.weight.device)
#         if original_bias is not None:
#             final_layer.bias.data = torch.from_numpy(original_bias).to(conv_block_out_layer.bias.device)

#         # Replace the layer in `conv_block_out`
#         pruned_encoder.conv_block_out[1] = final_layer
        
#     return pruned_encoder


def validate_pruned_model(original_model, pruned_model, pruned_filters_info):
    """
    Validate that the pruned model layers are correctly initialized compared to the original model.
    """
    previous_pruned_filters = None  # Initialize the previous pruned filters

    for layer_name, pruned_info in pruned_filters_info.items():
        print(layer_name)
        # Access layers in both models
        modules = layer_name.split('.')
        original_layer = original_model
        pruned_layer = pruned_model

        for mod in modules:
            original_layer = getattr(original_layer, mod)
            pruned_layer = getattr(pruned_layer, mod)

        # Get the pruned indices
        total_filters = original_layer.weight.size(0 if isinstance(original_layer, nn.Conv2d) else 1)
        pruned_indices = (
            pruned_info['pruned_indices'].cpu().numpy()
            if isinstance(pruned_info['pruned_indices'], torch.Tensor)
            else np.array(pruned_info['pruned_indices'])
        )
        remaining_filter_indices = np.setdiff1d(np.arange(total_filters), pruned_indices)

        # Validate weights
        original_weights = original_layer.weight.data.cpu().numpy()
        pruned_weights = pruned_layer.weight.data.cpu().numpy()

        if isinstance(original_layer, nn.Conv2d):
            expected_weights = original_weights[remaining_filter_indices, :, :, :]
            if previous_pruned_filters is not None:
                # Handle input pruning cascade
                expected_weights = expected_weights[:, previous_pruned_filters, :, :]
        elif isinstance(original_layer, nn.ConvTranspose2d):
            expected_weights = original_weights[:, remaining_filter_indices, :, :]
            if previous_pruned_filters is not None:
                # Handle output pruning cascade
                expected_weights = expected_weights[previous_pruned_filters, :, :, :]

        print("pruned_weights shape :{} - expected_weights shape: {}".format(pruned_weights.shape, expected_weights.shape))   
        if not np.allclose(pruned_weights, expected_weights):
            print(f"Weight mismatch in layer {layer_name}.")
            return False

        # Validate biases
        if original_layer.bias is not None:
            original_bias = original_layer.bias.data.cpu().numpy()
            pruned_bias = pruned_layer.bias.data.cpu().numpy()
            expected_bias = original_bias[remaining_filter_indices]
            if not np.allclose(pruned_bias, expected_bias):
                print(f"Bias mismatch in layer {layer_name}.")
                return False

        # Validate normalization layers
        parent_module = pruned_model
        for mod in modules[:-1]:
            parent_module = getattr(parent_module, mod)
        sibling_modules = list(parent_module.children())

        norm_layer_index = sibling_modules.index(pruned_layer) + 1
        if norm_layer_index < len(sibling_modules):
            norm_layer = sibling_modules[norm_layer_index]
            if isinstance(norm_layer, ChannelNorm2D):
                if hasattr(norm_layer, 'gamma'):
                    original_gamma = norm_layer.gamma.data.cpu().numpy()
                    pruned_gamma = norm_layer.gamma.data.cpu().numpy()
                    expected_gamma = original_gamma[remaining_filter_indices]
                    if not np.allclose(pruned_gamma, expected_gamma):
                        print(f"Gamma mismatch in norm layer following {layer_name}.")
                        return False

                if hasattr(norm_layer, 'beta'):
                    original_beta = norm_layer.beta.data.cpu().numpy()
                    pruned_beta = norm_layer.beta.data.cpu().numpy()
                    expected_beta = original_beta[remaining_filter_indices]
                    if not np.allclose(pruned_beta, expected_beta):
                        print(f"Beta mismatch in norm layer following {layer_name}.")
                        return False

        # Update the previous pruned filters for the next layer
        previous_pruned_filters = remaining_filter_indices

        print(f"Layer {layer_name} and associated norm layer validated successfully.")

    return True




def apply_physical_pruning(original_encoder, pruned_filters_info):
    """
    Apply physical pruning to the encoder based on the pruned filters info.
    Supports both Conv2d and ConvTranspose2d layers.
    """
    
    # Clone the original encoder so we don't modify it
    pruned_encoder = copy.deepcopy(original_encoder)
    previous_pruned_filters = None  # Track pruned output channels from the previous layer

    # Initialize normalization layers in the first block (conv_block_init.0)
    print(original_encoder.conv_block_init)
    orig_norm_layer = original_encoder.conv_block_init[0]
    
    pruned_encoder.conv_block_init[0].beta = nn.Parameter(orig_norm_layer.beta.data)
    pruned_encoder.conv_block_init[0].gamma = nn.Parameter(orig_norm_layer.gamma.data)

    # orig_conv_layer = original_encoder.conv_block_init[2]
    
    # pruned_indices = np.array(pruned_filters_info["conv_block_init.2"]['pruned_indices'])
    
    # total_filters = orig_conv_layer.weight.size(0)
    # remaining_filter_indices = np.setdiff1d(np.arange(total_filters), pruned_indices)
    

    # pruned_encoder.conv_block_init[2].weight.data = nn.Parameter(orig_conv_layer.weight.data[remaining_filter_indices])
    # pruned_encoder.conv_block_init[2].bias.data = nn.Parameter(orig_conv_layer.bias.data[remaining_filter_indices])

    
    for layer_name, pruned_info in pruned_filters_info.items():
        
        # Access the layer by traversing the model hierarchy
        modules = layer_name.split('.')
        print(layer_name)
        target_layer = pruned_encoder
        for mod in modules:
            target_layer = getattr(target_layer, mod)

        # Get the pruned indices
        # total_filters = target_layer.weight.size(0 if isinstance(target_layer, nn.Conv2d) else 1)
        # pruned_indices = (
        #     pruned_info['pruned_indices'].cpu().numpy()
        #     if isinstance(pruned_info['pruned_indices'], torch.Tensor)
        #     else np.array(pruned_info['pruned_indices'])
        # )
        # remaining_filter_indices = np.setdiff1d(np.arange(total_filters), pruned_indices)
        
        remaining_filter_indices = (
            pruned_info['remaining_indices'].cpu().numpy()
            if isinstance(pruned_info['remaining_indices'], torch.Tensor)
            else np.array(pruned_info['remaining_indices'])
        )

        if len(remaining_filter_indices) <= 0:
            raise ValueError(f"All filters pruned for layer {layer_name}. This is not allowed.")

        # Prune the layer weights
        original_weights = target_layer.weight.data.cpu().numpy()
        original_bias = target_layer.bias.data.cpu().numpy() if target_layer.bias is not None else None

        if isinstance(target_layer, nn.Conv2d):
            new_weights = original_weights[remaining_filter_indices, :, :, :]
            if previous_pruned_filters is not None:
                new_weights = new_weights[:, previous_pruned_filters, :, :]
        elif isinstance(target_layer, nn.ConvTranspose2d):
            new_weights = original_weights[:, remaining_filter_indices, :, :]
            if previous_pruned_filters is not None:
                new_weights = new_weights[previous_pruned_filters, :, :, :]

        new_bias = (
            original_bias[remaining_filter_indices]
            if original_bias is not None and isinstance(target_layer, nn.Conv2d) or isinstance(target_layer, nn.ConvTranspose2d)
            else None
        )

        # Create a new layer with pruned weights
        if isinstance(target_layer, nn.Conv2d):
            new_layer = nn.Conv2d(
                in_channels=new_weights.shape[1],
                out_channels=new_weights.shape[0],
                kernel_size=target_layer.kernel_size,
                stride=target_layer.stride,
                padding=target_layer.padding,
                dilation=target_layer.dilation,
                groups=target_layer.groups,
                bias=(new_bias is not None),
                padding_mode=target_layer.padding_mode,
            )
        elif isinstance(target_layer, nn.ConvTranspose2d):
            new_layer = nn.ConvTranspose2d(
                in_channels=new_weights.shape[0],
                out_channels=new_weights.shape[1],
                kernel_size=target_layer.kernel_size,
                stride=target_layer.stride,
                padding=target_layer.padding,
                dilation=target_layer.dilation,
                groups=target_layer.groups,
                bias=(new_bias is not None),
                padding_mode=target_layer.padding_mode,
                output_padding=target_layer.output_padding,
            )
        
        print("weights shape :{}".format(new_weights.shape))

        new_layer.weight.data = torch.from_numpy(new_weights).to(target_layer.weight.device)
        if new_bias is not None:
            new_layer.bias.data = torch.from_numpy(new_bias).to(target_layer.bias.device)

        # Replace the layer in the encoder
        parent_module = pruned_encoder
        for mod in modules[:-1]:
            parent_module = getattr(parent_module, mod)
        setattr(parent_module, modules[-1], new_layer)

        # Adjust normalization layers if applicable
        # Check for normalization layers that follow the pruned layer
        sibling_modules = list(parent_module.children())

        if "conv_block_out" not in layer_name:
            norm_layer_index = sibling_modules.index(new_layer) + 1 if not(isinstance(sibling_modules[sibling_modules.index(new_layer) + 1], nn.Conv2d)) and not(isinstance(sibling_modules[sibling_modules.index(new_layer) - 1], nn.Conv2d)) else sibling_modules.index(new_layer) + 2  # Start looking after the new_layer
        
        print(sibling_modules)
        
        norm_layer = sibling_modules[norm_layer_index]

            # Update normalization parameters
        if hasattr(norm_layer, 'beta'):
            norm_layer.beta = nn.Parameter(norm_layer.beta.data[:, remaining_filter_indices])
        if hasattr(norm_layer, 'gamma'):
            norm_layer.gamma = nn.Parameter(norm_layer.gamma.data[:, remaining_filter_indices]) 

        # Update previous_pruned_filters
        previous_pruned_filters = remaining_filter_indices

    # Handle `conv_block_out` separately
    conv_block_out_layer = pruned_encoder.conv_block_out[1]  # Access Conv2d in `conv_block_out`
    if isinstance(conv_block_out_layer, nn.Conv2d) and previous_pruned_filters is not None:
        original_weights = conv_block_out_layer.weight.data.cpu().numpy()
        original_bias = conv_block_out_layer.bias.data.cpu().numpy() if conv_block_out_layer.bias is not None else None

        # Adjust input channels to match pruned filters from the previous layer
        new_weights = original_weights[:, previous_pruned_filters, :, :]
        # Create a new Conv2d layer with adjusted input channels but the same output channels
        final_layer = nn.Conv2d(
            in_channels=len(previous_pruned_filters),
            out_channels=conv_block_out_layer.out_channels,
            kernel_size=conv_block_out_layer.kernel_size,
            stride=conv_block_out_layer.stride,
            padding=conv_block_out_layer.padding,
            dilation=conv_block_out_layer.dilation,
            groups=conv_block_out_layer.groups,
            bias=(original_bias is not None),
            padding_mode=conv_block_out_layer.padding_mode,
        )
        final_layer.weight.data = torch.from_numpy(new_weights).to(conv_block_out_layer.weight.device)
        if original_bias is not None:
            final_layer.bias.data = torch.from_numpy(original_bias).to(conv_block_out_layer.bias.device)

        # Replace the layer in `conv_block_out`
        pruned_encoder.conv_block_out[1] = final_layer


    is_valid = validate_pruned_model(original_encoder, pruned_encoder, pruned_filters_info)
    if is_valid:
        print("Pruned model validated successfully.")
    else:
        print("Validation failed: Some layers are not correctly initialized.")


    return pruned_encoder




def prune_and_compare(encoder_name, pruned_encoder, pruned_results, input_size, orig_flops, orig_param, logger):
    pruned_filters_info = pruned_results[encoder_name]

    # Original Encoder FLOPs and Parameters (used for comparison)
    logger.info(f"Original Model FLOPs: {orig_flops}, Parameters: {orig_param}")

    # Apply physical pruning to the current pruned model
    logger.info("\nApplying physical pruning...")
    pruned_encoder = apply_physical_pruning(pruned_encoder, pruned_filters_info)

    # Validate pruned model FLOPs and Parameters
    pruned_params = count_parameters(pruned_encoder)
    hooks, pruned_flops = register_hooks(pruned_encoder)

    print(torch.randn(input_size).shape)
    logger.info('Trainable parameters:')
    for n, p in pruned_encoder.named_parameters():
        logger.info('{} - {}'.format(n, p.shape))
    output = pruned_encoder(torch.randn(input_size))
    pruned_flops = print_flops(pruned_flops, logger)  # Get pruned FLOPs

    logger.info(f"\nUpdated Pruned Encoder '{encoder_name}'")
    logger.info(f"Number of Parameters: {pruned_params}")
    logger.info(f"FLOPs: {pruned_flops}")

    # Compute pruning ratios based on the original model
    param_ratio = 1 - pruned_params / orig_param
    flops_ratio = 1 - pruned_flops / orig_flops

    logger.info("Pruning param ratio: {:.3f}".format(param_ratio))
    logger.info("Pruning flops ratio: {:.3f}".format(flops_ratio))

    # Debugging Weight Shapes
    for name, param in pruned_encoder.named_parameters():
        logger.info(f"Layer: {name}, Shape: {param.shape}")
    
    return None, pruned_encoder, flops_ratio, param_ratio




def compute_conv_flops(input_shape, output_shape, layer):
    """
    Compute the FLOPs for a given Conv2D layer.
    
    Arguments:
    - input_shape: Tuple (Batch size, Input channels, Input height, Input width).
    - output_shape: Tuple (Batch size, Output channels, Output height, Output width).
    - layer: The Conv2D layer.
    
    Returns:
    - flops: The number of FLOPs for this layer.
    """
    batch_size, input_channels, input_height, input_width = input_shape
    batch_size, output_channels, output_height, output_width = output_shape
    
    kernel_size = layer.kernel_size
    flops_per_instance = (
        2 * output_channels * output_height * output_width *
        kernel_size[0] * kernel_size[1] * input_channels
    )
    
    total_flops = batch_size * flops_per_instance
    return total_flops



def register_hooks(model):
    """
    Registers hooks on each Conv2D layer to capture input and output shapes.
    Also calculates and sums up FLOPs for each layer.
    
    Arguments:
    - model: The PyTorch model.
    
    Returns:
    - total_flops: The total number of FLOPs for the model.
    """
    conv_flops = {}

    def hook(module, input, output):
        if isinstance(module, nn.Conv2d):
            input_shape = input[0].shape  # Shape of input tensor (Batch, Channels, Height, Width)
            output_shape = output.shape  # Shape of output tensor
            flops = compute_conv_flops(input_shape, output_shape, module)
            conv_flops[module] = flops

    hooks = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(hook))
    
    return hooks, conv_flops


def print_flops(conv_flops, logger):
    """
    Prints the FLOPs for each Conv2D layer and the total FLOPs.
    
    Arguments:
    - conv_flops: Dictionary of Conv2D layers and their corresponding FLOPs.
    """
    total_flops = 0
    for layer, flops in conv_flops.items():
        total_flops += flops
    logger.info(f"Total FLOPs: {total_flops / 1e6:.4f} MFLOPs")

    return total_flops

    



def print_encoder_pruning_details(original_encoder, pruned_encoder, logger):
    """
    Print the details of each encoder side by side.
    Show the size of filters (output channels) within each layer (original and pruned).
    
    Arguments:
    - original_encoder: The original encoder model.
    - pruned_encoder: The pruned encoder model.
    """
    logger.info("{:<25} {:<25} {:<25}".format("Layer", "Original Filters", "Pruned Filters"))
    logger.info("="*75)

    for (layer_name, layer), (_, pruned_layer) in zip(original_encoder.named_modules(), pruned_encoder.named_modules()):
        if isinstance(layer, nn.Conv2d):
            # Get original and pruned number of filters (output channels)
            original_filters = layer.weight.size(0)
            pruned_filters = pruned_layer.weight.size(0)

            # Print side by side comparison
            logger.info("{:<25} {:<25} {:<25}".format(
                layer_name,
                original_filters,
                pruned_filters
            ))

    logger.info("\nPruning details printed for all encoders.")


def count_parameters(model):
    total_params = 0
    for layer in model.modules():
        if isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d):
            layer_params = sum(np.prod(p.shape) for p in layer.weight.data.cpu().numpy())
            total_params += layer_params
    
    return total_params



def prune_filters_by_l1_norm(encoders, logger, prune_rate=0.2):
    """
    Prune filters using the L1 norm criterion.
    """
    pruned_results = {}
    for encoder_name, encoder in encoders.items():
        logger.info(f"Pruning {encoder_name} using L1 norm...")
        pruned_filters = encoder.prune_filters_by_l1_norm_percentage(prune_rate=prune_rate)
        pruned_results[encoder_name] = pruned_filters
    return pruned_results


def create_encoders_with_checkpoints(image_dims, batch_size, bottleneck_depth, checkpoint_paths, logger):
    """
    Function to create encoders for different bitrates and load their respective checkpoints.
    
    Arguments:
    - image_dims: Input image dimensions
    - batch_size: Number of instances per minibatch
    - bottleneck_depth: The depth of the bottleneck for the encoder
    - checkpoint_paths: Dictionary mapping encoder names to checkpoint file paths
    
    Returns:
    - encoders: Dictionary of encoders with loaded checkpoints
    """
    encoders = {}

    for encoder_name, checkpoint_path in checkpoint_paths.items():
        logger.info(f"Creating {encoder_name} and loading checkpoint from {checkpoint_path}...")
        encoder = Encoder(image_dims=image_dims, batch_size=batch_size, C=bottleneck_depth)
        encoder.to(torch.device("cpu"))
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        sd = {}
        for name, value in checkpoint["model_state_dict"].items():
            if "Encoder" in name:
                name = name.split("Encoder.")[1]
                sd[name] = value

    
        # Load the state dictionary into the encoder
        encoder.load_state_dict(sd, strict=False)
        
        encoders[encoder_name] = encoder
    
    return encoders




def analyze_pruned_filters_across_encoders(pruned_results, encoder_name):
    """
    Analyze and compare pruned filters across encoders to identify common unimportant filters.
    Now also granularly breaks it down layer by layer for each block.
    """
    common_pruned_filters = {}

    for layer_name in pruned_results[encoder_name]:
        # Collect the pruned indices from each encoder for the current layer
        pruned_0_45 = set(pruned_results["encoder_0.45_bpp"][layer_name]["pruned_indices"].tolist())
        pruned_0_30 = set(pruned_results["encoder_0.30_bpp"][layer_name]["pruned_indices"].tolist())
        pruned_0_14 = set(pruned_results["encoder_0.14_bpp"][layer_name]["pruned_indices"].tolist())

        # Find common pruned filters across all encoders
        common_pruned = pruned_0_45 & pruned_0_30 & pruned_0_14
            
        pruned_results[encoder_name][layer_name]["pruned_indices"] = torch.tensor(list(common_pruned))

    return pruned_results


def apply_pruning(conv_layer, mask):
        """
        Apply the pruning mask to the weights of the Conv2D layer.
        Filters that are pruned (mask == False) will have their weights set to zero.
        The mask will be broadcasted across the input channels and spatial dimensions.
        """
        if isinstance(conv_layer, nn.Conv2d):
            conv_layer.weight.data *= mask.view(-1, 1, 1, 1).float()  # Broadcasting the mask to match weight shape
        elif isinstance(conv_layer, nn.ConvTranspose2d):
            conv_layer.weight.data *= mask.view(1, -1, 1, 1).float()  # Broadcasting the mask to match weight shape
        
        # Optional: You can also prune the corresponding bias (if exists)
        if conv_layer.bias is not None:
            conv_layer.bias.data[~mask] = 0  # Set pruned bias to zero


def compute_layer_flops_params(module, input_shape):
    """Compute FLOPs and parameters for a Conv2d or ConvTranspose2d layer."""
    if isinstance(module, nn.Conv2d):
        out_channels, in_channels, k_h, k_w = module.weight.shape
        H_out, W_out = input_shape  # Output height and width
        flops = out_channels * in_channels * k_h * k_w * H_out * W_out
        params = out_channels * in_channels * k_h * k_w
    elif isinstance(module, nn.ConvTranspose2d):
        in_channels, out_channels, k_h, k_w = module.weight.shape
        H_out, W_out = input_shape  # Output height and width after upsampling
        flops = in_channels * out_channels * k_h * k_w * H_out * W_out
        params = in_channels * out_channels * k_h * k_w
    else:
        flops, params = 0, 0
    return flops, params


def capture_output_shapes(model, input_tensor):
    """
    Run a forward pass through the model and capture the output shapes of Conv2d and ConvTranspose2d layers.
    Returns a dictionary of output shapes for each layer.
    """
    output_shapes = {}

    def hook(module, input, output):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            output_shapes[module] = output.shape[2:]  # Capture H_out, W_out

    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            hooks.append(module.register_forward_hook(hook))

    # Perform a forward pass
    model(input_tensor)

    # Remove hooks
    for h in hooks:
        h.remove()

    return output_shapes


# def prune_filters_by_l1_norm_percentage(model, encoder_name, input_tensor, prune_rate=0.2):
#     """
#     Prune a percentage of filters based on the L1 norm of the filters' weights.
#     Compute FLOPs and parameter reduction ratios dynamically.
    
#     Arguments:
#     - input_tensor: Example input tensor to the model for shape inference.
#     - prune_rate: Percentage of filters to prune (between 0 and 1).
#     """
#     # Capture output shapes dynamically
#     output_shapes = capture_output_shapes(model, input_tensor)

#     pruned_filters = {encoder_name: {}}

#     original_flops, pruned_flops = 0, 0
#     original_params, pruned_params = 0, 0

#     for name, module in model.named_modules():
#         if isinstance(module, nn.ConvTranspose2d):
#             input_shape = output_shapes.get(module, (1, 1))  # Default to (1, 1) if shape not provided
#             layer_flops, layer_params = compute_layer_flops_params(module, input_shape)
#             original_flops += layer_flops
#             original_params += layer_params

#             # Determine the dimensions to reduce based on the layer type
#             if isinstance(module, nn.Conv2d):
#                 reduce_dims = (1, 2, 3)  # Reduce over in_channels, k_h, k_w
#             elif isinstance(module, nn.ConvTranspose2d):
#                 reduce_dims = (0, 2, 3)  # Reduce over out_channels, k_h, k_w

#             # Compute the L1 norm of each filter
#             filter_l1_norm = torch.sum(torch.abs(module.weight.data), dim=reduce_dims)

#             # Sort the filters by their L1 norm
#             sorted_indices = torch.argsort(filter_l1_norm)
#             num_filters = filter_l1_norm.size(0)
#             num_prune = int(prune_rate * num_filters)  # Number of filters to prune
            


#             if "conv_block_out" in name:
#                 print(f"Skipping pruning for {name} to preserve latent space size.")
#                 pruned_flops += layer_flops
#                 pruned_params += layer_params
#                 continue

#             # Create mask for keeping and pruning filters
#             mask = torch.ones(num_filters, dtype=torch.bool)
#             pruned_mask = sorted_indices[:num_prune]  # Indices of filters to prune
#             mask[pruned_mask] = False  # Mark the lowest L1 norm filters for pruning

#             # Compute pruned FLOPs and Params
#             pruned_layer_flops = layer_flops * mask.sum().item() / num_filters
#             pruned_layer_params = layer_params * mask.sum().item() / num_filters
#             pruned_flops += pruned_layer_flops
#             pruned_params += pruned_layer_params

#             # Apply pruning mask
#             apply_pruning(module, mask)

#             pruned_filters[encoder_name][name] = {
#                 'remaining_filters': mask.sum().item(),
#                 'pruned_filters': num_prune,
#                 'pruned_indices': np.sort(pruned_mask)
#             }

#     flops_ratio = (original_flops - pruned_flops) / original_flops
#     param_ratio = (original_params - pruned_params) / original_params

#     return pruned_filters, flops_ratio, param_ratio



# def prune_filters_by_l1_norm_percentage(model, encoder_name, input_tensor, prune_rate=0.2, logger=None):
#     output_shapes = capture_output_shapes(model, input_tensor)

#     pruned_filters = {encoder_name: {}}

#     original_params, original_flops = 0, 0

#     logger.info("Original parameters according to HiFiC code: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

#     original_weights = get_model_weights(model)
#     orig_param = np.sum([np.prod(weight.shape) for weight in original_weights])
#     logger.info("Original parameters {}".format(orig_param))
#     pruned_weights = get_model_weights(model)
    
#     model_features = [f for f in model.modules() if isinstance(f, nn.Conv2d) or isinstance(f, nn.ConvTranspose2d)]
#     feature_names = [name for name, f in model.named_modules() if isinstance(f, nn.Conv2d) or isinstance(f, nn.ConvTranspose2d)]

#     iteration = 0
#     flops_stable = 0  # Counter to track stable FLOPs

#     previous_flops_ratio = 0
#     while True:
#         pruned_flops = 0
#         pruned_params = 0
#         previous_indices = None
#         logger.info(f"Pruning iteration {iteration}")

#         for layer_idx, layer_weights in enumerate(pruned_weights):
#             input_shape = output_shapes.get(model_features[layer_idx], (1, 1))
#             layer_flops, layer_params = compute_layer_flops_params(model_features[layer_idx], input_shape)
#             layer_weights = torch.Tensor(layer_weights)

#             if iteration == 0:
#                 original_flops += layer_flops
#                 original_params += layer_params

#             logger.info(f"Processing layer {layer_idx}")

#             if "conv_block_out" in feature_names[layer_idx]:
#                 print(f"Skipping pruning for {feature_names[layer_idx]} to preserve latent space size.")
#                 pruned_flops += layer_flops
#                 pruned_params += layer_params
#                 continue

#             # Compute L1-norm for pruning criteria
#             start_time = time.time()
#             if isinstance(model_features[layer_idx], nn.Conv2d):
#                 l1_norms = layer_weights.abs().sum(dim=(1, 2, 3))  # Sum across spatial and input dimensions for Conv2d
#             elif isinstance(model_features[layer_idx], nn.ConvTranspose2d):
#                 l1_norms = layer_weights.abs().sum(dim=(0, 2, 3))  # Sum across spatial and output dimensions for ConvTranspose2d
#             else:
#                 continue

#             prune_threshold = np.percentile(l1_norms.cpu().numpy(), prune_rate * 100)  # Compute threshold
#             prune_indices = [i for i, norm in enumerate(l1_norms) if norm < prune_threshold]

#             if isinstance(model_features[layer_idx], nn.Conv2d):
#                 remaining_indices = [i for i in range(layer_weights.shape[0]) if i not in prune_indices]
#             elif isinstance(model_features[layer_idx], nn.ConvTranspose2d):
#                 remaining_indices = [i for i in range(layer_weights.shape[1]) if i not in prune_indices]

#             logger.info(f"Layer {layer_idx} - Current Filters: {len(l1_norms)} - Remaining Filters: {len(remaining_indices)} - Pruned Filters: {len(prune_indices)}")

#             pruned_filters[encoder_name][feature_names[layer_idx]] = {
#                 'remaining_filters': len(remaining_indices),
#                 'pruned_filters': len(prune_indices),
#                 'pruned_indices': prune_indices,
#                 'remaining_indices': remaining_indices,
#             }

#             if isinstance(model_features[layer_idx], nn.Conv2d):
#                 pruned_layer_weights = layer_weights[remaining_indices, :, :, :]
#                 out_channels, in_channels, kernel_h, kernel_w = pruned_layer_weights.shape

#                 if previous_indices is not None:
#                     out_channels, in_channels, kernel_h, kernel_w = pruned_layer_weights[:, previous_indices, :, :].shape

#             elif isinstance(model_features[layer_idx], nn.ConvTranspose2d):
#                 pruned_layer_weights = layer_weights[:, remaining_indices, :, :]
#                 in_channels, out_channels, kernel_h, kernel_w = pruned_layer_weights.shape

#                 if previous_indices is not None:
#                     in_channels, out_channels, kernel_h, kernel_w = pruned_layer_weights[previous_indices, :, :, :].shape

#             input_h, input_w = input_shape
#             layer_flops = out_channels * in_channels * kernel_h * kernel_w * input_h * input_w
#             layer_params = out_channels * in_channels * kernel_h * kernel_w

#             pruned_flops += layer_flops
#             pruned_params += layer_params

#             pruned_weights[layer_idx] = pruned_layer_weights
#             previous_indices = remaining_indices

#         iteration += 1
#         flops_ratio = 1 - pruned_flops / original_flops
#         param_ratio = 1 - pruned_params / original_params

#         logger.info(f"Iteration {iteration}: FLOP reduction ratio: {flops_ratio:.3f}, Param reduction ratio: {param_ratio:.3f}")
#         logger.info(f"Original FLOP: {original_flops}, Original Param: {original_params}")

#         if previous_flops_ratio == flops_ratio:
#             flops_stable += 1
#         else:
#             flops_stable = 0

#         if flops_stable >= 2:
#             logger.info("FLOPs stable for 3 consecutive iterations, exiting loop.")
#             break

#         previous_flops_ratio = flops_ratio
#         if flops_ratio >= prune_rate:
#             logger.info(f"Flops rate {flops_ratio * 100:.1f} - Param rate {param_ratio * 100:.1f}. Target pruning rate of {prune_rate * 100:.1f}% reached.")
#             break

#     return pruned_filters, flops_ratio, param_ratio






def prune_filters_by_l1_norm_percentage(model, encoder_name, input_tensor, prune_rate=0.2, logger=None):
    output_shapes = capture_output_shapes(model, input_tensor)

    pruned_filters = {encoder_name: {}}

    original_params, original_flops = 0, 0

    logger.info("Original parameters according to HiFiC code: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    original_weights = get_model_weights(model)
    orig_param = np.sum([np.prod(weight.shape) for weight in original_weights])
    logger.info("Original parameters {}".format(orig_param))
    pruned_weights = get_model_weights(model)

    model_features = [f for f in model.modules() if isinstance(f, nn.Conv2d) or isinstance(f, nn.ConvTranspose2d)]
    feature_names = [name for name, f in model.named_modules() if isinstance(f, nn.Conv2d) or isinstance(f, nn.ConvTranspose2d)]

    iteration = 0
    flops_stable = 0  # Counter to track stable FLOPs

    previous_flops_ratio = 0
    while True:
        pruned_flops = 0
        pruned_params = 0
        previous_indices = None
        logger.info(f"Pruning iteration {iteration}")

        for layer_idx, layer_weights in enumerate(pruned_weights):
            input_shape = output_shapes.get(model_features[layer_idx], (1, 1))
            layer_flops, layer_params = compute_layer_flops_params(model_features[layer_idx], input_shape)

            if iteration == 0:
                original_flops += layer_flops
                original_params += layer_params

            logger.info(f"Processing layer {layer_idx}")

            if "conv_block_out" in feature_names[layer_idx]:
                print(f"Skipping pruning for {feature_names[layer_idx]} to preserve latent space size.")
                pruned_flops += layer_flops
                pruned_params += layer_params
                continue

            # Compute L1-norm for pruning criteria
            if isinstance(model_features[layer_idx], nn.Conv2d):
                l1_norms = np.sum(np.abs(layer_weights)**2, axis=(1, 2, 3))  # Sum across spatial and input dimensions for Conv2d
            elif isinstance(model_features[layer_idx], nn.ConvTranspose2d):
                l1_norms = np.sum(np.abs(layer_weights)**2, axis=(0, 2, 3))  # Sum across spatial and output dimensions for ConvTranspose2d
            else:
                continue

            # Sort indices by L1-norm in descending order
            sorted_indices = np.argsort(l1_norms)[::-1]  # Sort of L1-norms
            prune_unitl = int(np.round(len(sorted_indices) * prune_rate)) if prune_rate != 0 else 0
            prune_indices = np.sort(sorted_indices[:prune_unitl])

            # Updated remaining_indices calculation
            if not (("resblock" in feature_names[layer_idx] and "conv2" in feature_names[layer_idx]) or "conv_block_init" in feature_names[layer_idx]):
                remaining_indices = [i for i in range(len(l1_norms)) if i not in prune_indices]
            else:
                remaining_indices = np.arange(layer_weights.shape[0] if isinstance(model_features[layer_idx], nn.Conv2d) else layer_weights.shape[1])
                prune_indices = []

            logger.info(f"Layer {layer_idx} - Current Filters: {len(l1_norms)} - Remaining Filters: {len(remaining_indices)} - Pruned Filters: {len(prune_indices)}")

            pruned_filters[encoder_name][feature_names[layer_idx]] = {
                'remaining_filters': len(remaining_indices),
                'pruned_filters': len(prune_indices),
                'pruned_indices': prune_indices,
                'remaining_indices': remaining_indices,
            }

            # Update layer weights based on remaining indices
            if isinstance(model_features[layer_idx], nn.Conv2d):
                pruned_layer_weights = layer_weights[remaining_indices, :, :, :]
                out_channels, in_channels, kernel_h, kernel_w = pruned_layer_weights.shape

                if previous_indices is not None:
                    out_channels, in_channels, kernel_h, kernel_w = pruned_layer_weights[:, previous_indices, :, :].shape

            elif isinstance(model_features[layer_idx], nn.ConvTranspose2d):
                pruned_layer_weights = layer_weights[:, remaining_indices, :, :]
                in_channels, out_channels, kernel_h, kernel_w = pruned_layer_weights.shape

                if previous_indices is not None:
                    in_channels, out_channels, kernel_h, kernel_w = pruned_layer_weights[previous_indices, :, :, :].shape

            input_h, input_w = input_shape
            layer_flops = out_channels * in_channels * kernel_h * kernel_w * input_h * input_w
            layer_params = out_channels * in_channels * kernel_h * kernel_w

            pruned_flops += layer_flops
            pruned_params += layer_params

            pruned_weights[layer_idx] = pruned_layer_weights
            previous_indices = remaining_indices

        iteration += 1
        flops_ratio = 1 - pruned_flops / original_flops
        param_ratio = 1 - pruned_params / original_params

        logger.info(f"Iteration {iteration}: FLOP reduction ratio: {flops_ratio:.3f}, Param reduction ratio: {param_ratio:.3f}")
        logger.info(f"Original FLOP: {original_flops}, Original Param: {original_params}")

        if previous_flops_ratio == flops_ratio:
            flops_stable += 1
        else:
            flops_stable = 0

        if flops_stable >= 2:
            logger.info("FLOPs stable for 3 consecutive iterations, exiting loop.")
            break

        previous_flops_ratio = flops_ratio
        if flops_ratio >= prune_rate:
            logger.info(f"Flops rate {flops_ratio * 100:.1f} - Param rate {param_ratio * 100:.1f}. Target pruning rate of {prune_rate * 100:.1f}% reached.")
            break

    return pruned_filters, flops_ratio, param_ratio






def prune_filters_by_ccfp(args, model, encoder_name, input_tensor, prune_rate=0.2, angle_thresh=15, logger=None):
    output_shapes = capture_output_shapes(model, input_tensor)

    pruned_filters = {encoder_name: {}}

    original_params, original_flops = 0, 0
    
    logger.info("Original parameters according to HiFiC code: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    original_weights = get_model_weights(model)
    orig_param = np.sum([np.prod(weight.shape) for weight in original_weights])
    logger.info("original paramters {}".format(orig_param))
    pruned_weights = get_model_weights(model)
    Weights_p = []

    model_features = [f for f in model.modules() if isinstance(f, nn.Conv2d) or isinstance(f, nn.ConvTranspose2d)]
    feature_names = [name for name, f in model.named_modules() if isinstance(f, nn.Conv2d) or isinstance(f, nn.ConvTranspose2d)]

    iteration = 0
    flops_stable = 0  # Counter to track stable FLOPs
    correlation_history = {}  # Store correlation circles for each layer
    remaining_indices_history = {}  # Track remaining indices for labeling

    previous_flops_ratio = 0
    while True:
        pruned_flops = 0
        pruned_params = 0
        previous_indices = None
        logger.info(f"Pruning iteration {iteration}")

        for layer_idx, layer_weights in enumerate(pruned_weights):
            input_shape = output_shapes.get(model_features[layer_idx], (1, 1))
            layer_flops, layer_params = compute_layer_flops_params(model_features[layer_idx], input_shape)

            if iteration == 0:
                original_flops += layer_flops
                original_params += layer_params

            logger.info(f"Processing layer {layer_idx}")
            

            if "conv_block_out" in feature_names[layer_idx]:
                print(f"Skipping pruning for {feature_names[layer_idx]} to preserve latent space size.")
                pruned_flops += layer_flops
                pruned_params += layer_params
                continue


            start_time = time.time()
            total_corr, l2_total, pca_weights = get_cc(layer_weights, model_features[layer_idx], Weights_p, layer_idx, iteration)
            get_cc_time = time.time() - start_time
            logger.info(f"get_cc for layer {layer_idx} took {get_cc_time:.3f} seconds")

            if layer_idx not in correlation_history:
                correlation_history[layer_idx] = []
                remaining_indices_history[layer_idx] = [list(range(total_corr.shape[0]))]  # Initialize indices

            correlation_history[layer_idx].append(total_corr)

            prune_indices = []
            idx = list(range(total_corr.shape[0]))

            start_time = time.time()
            cov_mat = cosine_sim(total_corr, absolute=True)
            cosine_sim_time = time.time() - start_time
            logger.info(f"cosine_sim for layer {layer_idx} took {cosine_sim_time:.3f} seconds")

            max_corr = [
                (i, np.argmax(line))
                for i, line in enumerate(cov_mat)
                if np.max(line) > np.cos(angle_thresh * np.pi / 180)
            ]

            selected = mutual_corr(max_corr)

            prune_indices = [
                sel[0] if l2_total[sel[0]] < l2_total[sel[1]] else sel[1]
                for sel in selected
            ]

            prune_unitl = int(np.round(len(prune_indices) * 0.25)) if prune_rate != 0 else 0
            prune_indices = prune_indices[:prune_unitl]

            if not (("resblock" in feature_names[layer_idx] and "conv2" in feature_names[layer_idx]) or "conv_block_init" in feature_names[layer_idx]):
                remaining_indices = [i for i in idx if i not in prune_indices]
            else:
                remaining_indices = np.arange(layer_weights.shape[(0 if isinstance(model_features[layer_idx], nn.Conv2d) else 1)])
                prune_indices = []

            # Map the remaining indices to the original index list
            prev_indices = remaining_indices_history[layer_idx][-1]
            new_indices = [prev_indices[i] for i in remaining_indices]
            remaining_indices_history[layer_idx].append(new_indices)
            

            logger.info(f"Layer {layer_idx} - Current Filters: {len(idx)} - Remaining Filters: {len(remaining_indices)} - Pruned Filters: {len(prune_indices)}")

            update_pruned = np.setdiff1d(np.arange(original_weights[layer_idx].shape[(0 if isinstance(model_features[layer_idx], nn.Conv2d) else 1)]), new_indices)

            pruned_filters[encoder_name][feature_names[layer_idx]] = {
                'remaining_filters': len(new_indices),
                'pruned_filters': len(update_pruned),
                'pruned_indices': update_pruned,
                'remaining_indices': new_indices,
            }

            
            if isinstance(model_features[layer_idx], nn.Conv2d):
                pruned_layer_weights = layer_weights[remaining_indices, :, :, :]
                out_channels, in_channels, kernel_h, kernel_w = pruned_layer_weights.shape

                if previous_indices is not None: #and pruned_layer_weights.shape[1] > 0:
                    out_channels, in_channels, kernel_h, kernel_w = pruned_layer_weights[:, previous_indices, :, :].shape
                
            elif isinstance(model_features[layer_idx], nn.ConvTranspose2d):
                pruned_layer_weights = layer_weights[:, remaining_indices, :, :]
                in_channels, out_channels, kernel_h, kernel_w = pruned_layer_weights.shape
            
                if previous_indices is not None: #and pruned_layer_weights.shape[1] > 0:
                    in_channels, out_channels, kernel_h, kernel_w = pruned_layer_weights[previous_indices, :, :, :].shape

            input_h, input_w = input_shape
            layer_flops = out_channels * in_channels * kernel_h * kernel_w * input_h * input_w
            layer_params = out_channels * in_channels * kernel_h * kernel_w

            pruned_flops += layer_flops
            pruned_params += layer_params

            pruned_weights[layer_idx] = pruned_layer_weights
            previous_indices = remaining_indices

        iteration += 1
        flops_ratio = 1 - pruned_flops / original_flops
        param_ratio = 1 - pruned_params / original_params

        logger.info(f"Iteration {iteration}: FLOP reduction ratio: {flops_ratio:.3f}, Param reduction ratio: {param_ratio:.3f}")
        logger.info(f"Original FLOP: {original_flops}, Original Param: {original_params}")

        if previous_flops_ratio == flops_ratio:
            flops_stable += 1
        else:
            flops_stable = 0

        if flops_stable >= 2:
            logger.info("FLOPs stable for 3 consecutive iterations, exiting loop.")
            break

        previous_flops_ratio = flops_ratio
        if flops_ratio >= prune_rate:
            logger.info(f"Flops rate {flops_ratio * 100:.1f} - Param rate {param_ratio * 100:.1f}. Target pruning rate of {prune_rate * 100:.1f}% reached.")
            break


    plot_correlation_evolution(args, correlation_history, remaining_indices_history)
    return pruned_filters, flops_ratio, param_ratio



def plot_correlation_evolution(args, correlation_history, remaining_indices_history):

    for layer_idx, correlation_circles in correlation_history.items():
        num_iterations = len(correlation_circles)
        fig, axes = plt.subplots(1, num_iterations, figsize=(30, 10))
        fig.suptitle(f'Correlation Circle Evolution for Layer {layer_idx}', fontsize=20)

        original_labels = [j for j in range(correlation_circles[0].shape[0])]

        for i, circle_corr in enumerate(correlation_circles):
            ax = axes[i] if num_iterations > 1 else axes
            ax.set_xlim([-1.1, 1.1])
            ax.set_ylim([-1.1, 1.1])
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(f'Iteration {i + 1}')
            ax.grid(True)

            remaining_indices = np.sort(remaining_indices_history[layer_idx][i])

            for j, (x, y) in enumerate(circle_corr):
                if j == original_labels[remaining_indices[j]]:
                    color = 'b'
                else:
                    color = 'r'
                    
                ax.quiver(0, 0, x, y, angles='xy', scale_units='xy', scale=1, color=color)
                ax.text(x, y, f'F{original_labels[remaining_indices[j]]}', fontsize=20, ha='right', color = color)

            ax.add_artist(plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='dashed'))

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save the figure to a file
        save_file = os.path.join(args.figures_save, f'layer_{layer_idx}_correlation_evolution.png')
        plt.savefig(save_file, dpi=300)
        plt.close(fig)  # Close the figure to free up memory



# def corr_circle(original_model, pruned_model, encoder_name, angle_thresh, pruning_ratio, input_size, logger):
#     """
#     Perform pruning iteratively based on correlation between filters until the target FLOPS ratio is reached.

#     Args:
#         original_model: The original, unpruned model for baseline calculations.
#         pruned_model: The model to prune progressively.
#         encoder_name: Name of the encoder being pruned.
#         angle_thresh: Threshold for angle between filters in degrees.
#         pruning_ratio: Target pruning ratio for FLOPs.
#         input_size: Input tensor size (batch size, channels, height, width).
#         logger: Logger instance for logging progress and results.

#     Returns:
#         pruned_filters: Information about the pruning performed for each layer.
#         flops_ratio: Final FLOPS reduction ratio relative to the original model.
#         param_ratio: Final parameter reduction ratio relative to the original model.
#     """
#     # Get weights for the original model
#     original_weights = get_model_weights(original_model[encoder_name])  # Original model weights
#     orig_param = np.sum([np.prod(weight.shape) for weight in original_weights])

#     # Calculate original FLOPS
#     orig_flops = 0
#     input_channels, input_h, input_w = input_size[1:]
#     for weight in original_weights:
#         if weight.ndim == 4:  # Convolutional layer
#             out_channels, in_channels, kernel_h, kernel_w = weight.shape
#             orig_flops += (
#                 out_channels * in_channels * kernel_h * kernel_w * input_h * input_w
#             )
#             input_channels = out_channels  # Update for the next layer

#     logger.info("=" * 50)
#     logger.info(f"Baseline Model Parameters: {orig_param}")
#     logger.info(f"Baseline Model FLOPS: {orig_flops}")
#     logger.info("=" * 50)

#     # Initialize pruning data
#     pruned_filters = {encoder_name: {}}
#     iteration = 0
#     pruned_weights = get_model_weights(pruned_model)
#     flops_ratio = 0

    
#     Weights_p = []
#     # Iteratively prune until target FLOPS ratio is reached
#     while flops_ratio < pruning_ratio:
#         logger.info(f"Pruning Iteration {iteration}")
#         input_channels = input_size[1]  # Reset input channels to original
#         pruned_flops = 0

        # # Compute PCA and correlations once for the pruned model
        # total_corr, l2_total = get_cc(pruned_model, Weights_p, pruned_weights, iteration)

        # # Prune each layer based on correlations
        # for layer_idx, (name, circle_corr) in enumerate(total_corr):
        #     idx = list(range(circle_corr.shape[0]))
        #     cov_mat = cosine_sim(circle_corr, True)
        #     max_corr = [
        #         (i, np.argmax(line))
        #         for i, line in enumerate(cov_mat)
        #         if np.max(line) > np.cos(angle_thresh * np.pi / 180)
        #     ]
        #     selected = mutual_corr(max_corr)
        #     pruned_indices = [
        #         sel[0] if l2_total[layer_idx][sel[0]] < l2_total[layer_idx][sel[1]] else sel[1]
        #         for sel in selected
        #     ]

        #     remaining_indices = [i for i in idx if i not in pruned_indices]
            
        #     logger.info(len(remaining_indices))
        #     if "conv_block_out" in name:
        #         logger.info(f"Skipping pruning for {name} to preserve latent space size.")
        #         remaining_indices = idx  # Retain all filters if pruning is skipped

        #     # Update pruning information
        #     pruned_filters[encoder_name][name] = {
        #         "remaining_filters": len(remaining_indices),
        #         "pruned_filters": len(pruned_indices),
        #         "pruned_indices": pruned_indices,
        #         "remaining_indices": remaining_indices,
        #     }

        #     # Recompute weights and FLOPS for the pruned layer
        #     original_layer_weights = original_weights[layer_idx]
        #     pruned_layer_weights = original_layer_weights[remaining_indices, :, :, :]
        #     if in_channels is not None:
        #         pruned_layer_weights = pruned_layer_weights[:, :in_channels, :, :]

        #     # Update FLOPS
        #     out_channels, in_channels, kernel_h, kernel_w = pruned_layer_weights.shape
        #     pruned_flops += (
        #         out_channels * in_channels * kernel_h * kernel_w * input_h * input_w
        #     )
        #     in_channels = out_channels  # Update for next layer

        #     # Store the pruned weights
        #     pruned_weights[layer_idx] = pruned_layer_weights

#         # Calculate the FLOPS and parameter ratios
#         pruned_params = np.sum([np.prod(weight.shape) for weight in pruned_weights])
#         param_ratio = 1 - pruned_params / orig_param
#         flops_ratio = 1 - pruned_flops / orig_flops

#         logger.info(f"Pruning param ratio: {param_ratio:.3f}")
#         logger.info(f"Pruning flops ratio: {flops_ratio:.3f}")

#         # Check if target FLOPS ratio is reached
#         if flops_ratio >= pruning_ratio:
#             logger.info(f"Target pruning ratio of {pruning_ratio * 100:.1f}% reached.")
#             break

#         iteration += 1

#     logger.info("=" * 50)
#     logger.info(f"Final Pruned Model Parameters: {pruned_params}")
#     logger.info(f"Final Pruned Model FLOPS: {pruned_flops}")
#     logger.info("=" * 50)

#     return pruned_filters, flops_ratio, param_ratio



def get_model_weights(model):
    model_weights = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ConvTranspose2d):
            weights = module.weight.data.detach().cpu().numpy()
            model_weights.append(weights)

        elif isinstance(module, nn.Conv2d):
            weights = module.weight.data.detach().cpu().numpy()
            model_weights.append(weights)

    return model_weights

    



def get_cc(weight, module, Weights_p, layer_idx, iteration):

    # Reshape weights for PCA based on layer type
    if isinstance(module, nn.Conv2d):
        weights = weight.reshape(weight.shape[0], -1).transpose()  # (elements, filters)
    if isinstance(module, nn.ConvTranspose2d):
        weights = weight.reshape(weight.shape[1], -1).transpose()  # (elements, filters)
        

    # Perform PCA for the first iteration
    if iteration == 0:
        pca = PCA().fit(weights)
        weights_p = pca.transform(weights)
        Weights_p.append(weights_p)
    else:
        weights_p = np.array(Weights_p[layer_idx])  # Ensure it's a NumPy array

    

    # Calculate correlation using broadcasting and vectorization
    # weights_transposed = weights.transpose()  # Transpose once instead of per loop
    print(weights_p.shape)
    print(weights.shape)
    circle_corr = np.corrcoef(weights, weights_p[:, :2], rowvar=False)[:weights.shape[1], -2:]

    # Calculate L2 norms efficiently
    l2 = np.sum(weights**2, axis=0)  # Sum across dimensions for each vector

    return circle_corr, l2, weights_p

    # print(weights_p.shape)
    # circle_corr = []
    # l2 = []
    # for w in weights.transpose():
    #     corrx = [np.corrcoef(w, weights_p[:, dim])[0, 1] for dim in range(2)]
    #     circle_corr.append(np.array(corrx))
    #     l2.append(np.sum(w**2))

    # return np.array(circle_corr), np.array(l2), weights_p



def mutual_corr(args):
    selected = []
    for e in args:
      if (e[::-1] in args) and (e[::-1] not in selected):
          selected.append(e)
    return selected


# def cosine_sim(circle_corr, absolute=True):
#     cov_mat = np.zeros((len(circle_corr), len(circle_corr)))
#     for i in range(len(circle_corr)):
#         for j in range(i, len(circle_corr)):
#             if i == j:
#                 cov_mat[i, j] = 0
#                 continue
#             scalar = np.sum(circle_corr[i] * circle_corr[j])
#             norm = np.sqrt(np.sum(circle_corr[i]**2)) * np.sqrt(np.sum(circle_corr[j]**2))
#             angle = scalar / norm
#             cov_mat[i, j] = abs(angle) if absolute else angle
#             cov_mat[j, i] = cov_mat[i, j]
#     return cov_mat.astype(np.float32)


def cosine_sim(circle_corr, absolute=True):
    circle_corr = np.array(circle_corr)  # Ensure input is a NumPy array
    norms = np.linalg.norm(circle_corr, axis=1)  # Compute norms for each row
    dot_products = np.dot(circle_corr, circle_corr.T)  # Compute pairwise dot products
    norm_matrix = np.outer(norms, norms)  # Compute outer product of norms
    
    # Compute cosine similarity
    cov_mat = dot_products / norm_matrix
    np.fill_diagonal(cov_mat, 0)  # Set diagonal to 0
    
    # Apply absolute if needed
    if absolute:
        cov_mat = np.abs(cov_mat)
    
    return cov_mat.astype(np.float32)

