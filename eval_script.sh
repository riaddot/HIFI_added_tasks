# Train additional tasks on standard latent

# python train.py -mt compression --tasks HiFiC Zoom FFX -norm -bs 20 --os_gpu 1 --regime high --evaluate

# python train.py -mt compression --tasks HiFiC Zoom FFX -norm -bs 20 --os_gpu 1 --regime high --evaluate --eval_ckpt /home/bellelbn/DL/datapart/jpegai_experiments/experiments/lfw_compression_high_ln_adapt_linear_Pruning_0.60_dc_1_2024_11_30_16_28/HiFiC_Zoom_FFX/best_checkpoint.pt

# python train.py -mt compression --tasks HiFiC  Zoom FFX -bs 20 --os_gpu 1 --regime high --evaluate --eval_ckpt /home/bellelbn/DL/datapart/jpegai_experiments/experiments/lfw_compression_high_ln_adapt_linear_Pruning_0.40_dc_1_2024_12_01_20_39/HiFiC_Zoom_FFX/best_checkpoint.pt

# python train.py -mt compression --tasks HiFiC Zoom FFX -norm -bs 20 --os_gpu 1 --regime high --evaluate --eval_ckpt /home/bellelbn/DL/datapart/jpegai_experiments/experiments/lfw_compression_high_ln_adapt_linear_Pruning_0.30_dc_1_angle-th5_2024_11_29_00_21/HiFiC_Zoom_FFX/best_checkpoint.pt

# python train.py -mt compression --tasks HiFiC Zoom FFX -norm -bs 20 --os_gpu 1 --regime high --evaluate --eval_ckpt /home/bellelbn/DL/datapart/jpegai_experiments/experiments/lfw_compression_high_ln_adapt_linear_Pruning_0.40_dc_1_2024_12_01_20_39/HiFiC_Zoom_FFX/best_checkpoint.pt



python train.py -mt compression --tasks HiFiC --prune -pr 0.2 -norm -bs 20 --os_gpu 1 --regime high --evaluate --eval_ckpt "/home/bellelbn/DL/datapart/jpegai_experiments/experiments/openimages_compression_high_Pruning_0.20_angle-th 2_Eval_dc_1_2024_12_25_17_43/HiFiC/best_checkpoint.pt"
python train.py -mt compression --tasks HiFiC --prune -pr 0.4 -norm -bs 20 --os_gpu 1 --regime high --evaluate --eval_ckpt "/home/bellelbn/DL/datapart/jpegai_experiments/experiments/openimages_compression_high_Pruning_0.40_angle-th 2_Eval_dc_1_2024_12_26_07_32/HiFiC/best_checkpoint.pt"
python train.py -mt compression --tasks HiFiC --prune -pr 0.6 -norm -bs 20 --os_gpu 1 --regime high --evaluate --eval_ckpt "/home/bellelbn/DL/datapart/jpegai_experiments/experiments/openimages_compression_high_Pruning_0.60_angle-th 2_Eval_dc_1_2024_12_26_21_21/HiFiC/best_checkpoint.pt"