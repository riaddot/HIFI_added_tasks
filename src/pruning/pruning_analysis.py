import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def verify_pruned_weights(original_encoder, pruned_encoder, pruned_filters_info, logger):
    """
    Verifies that the pruned encoder only contains the remaining weights as specified
    in the pruned_filters_info. This includes checking both filters and input channels.

    Arguments:
    - original_encoder: The original encoder model.
    - pruned_encoder: The pruned encoder model.
    - pruned_filters_info: A dictionary containing pruned filters information for each layer.
    - logger: Logger instance for logging messages.
    """
    logger.info("Verifying pruned weights and channels...")

    previous_remaining_indices = None  # Track remaining output channels from the previous layer

    for layer_name, pruned_info in pruned_filters_info.items():
        logger.info(f"Verifying layer: {layer_name}")

        # Retrieve the original and pruned layers
        modules = layer_name.split('.')
        original_layer = original_encoder
        pruned_layer = pruned_encoder

        # Access the original and pruned Conv2D layers
        try:
            for mod in modules:
                original_layer = getattr(original_layer, mod)
                pruned_layer = getattr(pruned_layer, mod)
        except AttributeError:
            logger.info(f"Layer {layer_name} not found in model. Please check layer names.")
            continue

        # Get remaining indices for the current layer
        if isinstance(pruned_info['pruned_indices'], torch.Tensor):
            pruned_indices = pruned_info['pruned_indices'].cpu().numpy()
        else:
            pruned_indices = pruned_info['pruned_indices']

        remaining_indices = np.setdiff1d(np.arange(original_layer.weight.size(0)), pruned_indices)

        # Filter weights to match the pruned layer's dimensions
        original_weights = original_layer.weight.data[remaining_indices].cpu()

        # Channel pruning verification: Check that the pruned layer's input channels match previous layer's output
        if previous_remaining_indices is not None:
            # Verify that the pruned layer's input channels match previous layer's output
            original_weights = original_weights[:, previous_remaining_indices, :, :].cpu()

        pruned_weights = pruned_layer.weight.data.cpu()

        # Ensure output channel shapes match
        if original_weights.shape != pruned_weights.shape or not torch.allclose(original_weights, pruned_weights, atol=1e-6):
            logger.info(f"Weight mismatch detected in layer {layer_name}!")
            return False

        if original_layer.bias is not None:
            original_bias = original_layer.bias.data[remaining_indices].cpu()
            pruned_bias = pruned_layer.bias.data.cpu()
            if original_bias.shape != pruned_bias.shape or not torch.allclose(original_bias, pruned_bias, atol=1e-6):
                logger.info(f"Bias mismatch detected in layer {layer_name}!")
                return False

        
            # # Ensure input channel shapes match
            # if original_input_channels.shape != pruned_input_channels.shape or not torch.allclose(original_input_channels, pruned_input_channels, atol=1e-6):
            #     logger.info(f"Channel mismatch detected in input channels of layer {layer_name}!")
            #     return False

        logger.info(f"Layer {layer_name} passed verification.")

        # Update previous remaining indices for channel verification in the next layer
        previous_remaining_indices = remaining_indices

    # Special handling for the final layer
    if hasattr(pruned_encoder, 'conv_block_out'):
        final_layer = pruned_encoder.conv_block_out[1]
        original_final_layer = original_encoder.conv_block_out[1]
        if previous_remaining_indices is not None:

            print(final_layer.weight.data.shape)
            final_layer_weights = final_layer.weight.data#[:, previous_remaining_indices, :, :].cpu()
            original_final_weights = original_final_layer.weight.data[:, previous_remaining_indices, :, :].cpu()

            if original_final_weights.shape != final_layer_weights.shape or not torch.allclose(original_final_weights, final_layer_weights, atol=1e-6):
                logger.info("Mismatch detected in final layer input channels after pruning.")
                return False

    logger.info("All pruned layers and channels have been verified successfully.")
    return True

