# Train additional tasks on standard latent
# python train.py -mt compression --tasks Zoom FFX -w 1 -norm -bs 10 --os_gpu 1
# python train.py -mt compression --tasks HiFiC Zoom FFX -w 1 --norm_loss --adaptative linear -norm -bs 10 --os_gpu 1 --regime med
# python train.py -mt compression --tasks HiFiC Zoom FFX -w 1 --norm_loss --adaptative linear -norm -bs 10 --os_gpu 1 --regime med

# python train.py -mt compression --tasks Zoom FFX -w 1 --norm_loss --adaptative linear -norm -bs 10 --os_gpu 1 --regime low


# python train.py -mt compression --tasks HiFiC Zoom FFX -w 1 --norm_loss --adaptative linear -norm -bs 10 --os_gpu 1 --regime low




python train.py --evaluate /home/bellelbn/DL/datapart/models/hific_hi.pt -mt compression --tasks HiFiC -norm -bs 10 --double_compression 3 --os_gpu 0 --regime low
# python train.py --evaluate /home/bellelbn/DL/datapart/models/hific_med.pt -mt compression --tasks HiFiC -bs 10 --double_compression 5 --os_gpu 0 --regime med
# python train.py --evaluate /home/bellelbn/DL/datapart/models/hific_low.pt -mt compression --tasks HiFiC -norm -bs 10 --double_compression 5 --os_gpu 0 --regime low



# python train.py -mt compression --tasks HiFiC -norm --adaptative exp -bs 10 --os_gpu 1 --regime high

# python train.py -mt compression --tasks HiFiC -w 1 -norm --adaptative linear -bs 10 --os_gpu 1 --regime high

# python train.py -mt compression --tasks HiFiC Zoom -w 1 -norm --adaptative exp -bs 10 --os_gpu 1 --regime high
# python train.py -mt compression --tasks HiFiC Zoom -w 1 -norm --adaptative linear -bs 10 --os_gpu 1 --regime high

# python train.py -mt compression --tasks HiFiC FFX -w 1 -norm --adaptative exp -bs 10 --os_gpu 1 --regime high
# python train.py -mt compression --tasks HiFiC FFX -w 1 -norm --adaptative linear -bs 10 --os_gpu 1 --regime high

# python train.py -mt compression --tasks HiFiC Zoom -w 1 -norm --test_task -bs 10 --os_gpu 0 --regime high
# python train.py -mt compression --tasks HiFiC FFX -w 1 -norm --test_task -bs 10 --os_gpu 1 --regime high

# python train.py -mt compression --tasks Zoom -w 1 -norm --test_task -bs 10 --os_gpu 0 --regime high
# python train.py -mt compression --tasks FFX -w 1 -norm --test_task -bs 10 --os_gpu 1 --regime high

# python train.py -mt compression --tasks HiFiC Zoom FFX -w 1 -norm -bs 10 --os_gpu 1 --regime med
# python train.py -mt compression --tasks HiFiC Zoom FFX -w 1 -norm -bs 10 --os_gpu 1 --regime low


# python train.py --evaluate /home/bellelbn/DL/datapart/jpegai_experiments/experiments/lfw_compression_high_adapt_linear_origW_2024_08_21_15_25/HiFiC/best_checkpoint.pt -mt compression --tasks HiFiC -w 1 -norm --adaptative linear -bs 10 --os_gpu 1 --regime high
# python train.py --evaluate /home/bellelbn/DL/datapart/jpegai_experiments/experiments/lfw_compression_high_adapt_exp_origW_2024_08_21_12_42/HiFiC/best_checkpoint.pt -mt compression --tasks HiFiC -w 1 -norm --adaptative exp -bs 10 --os_gpu 1 --regime high

# python train.py -mt compression --tasks Zoom FFX -w 1 -bs 10 --os_gpu 1 --regime med
# python train.py -mt compression --tasks HiFiC Zoom FFX -w 1 --norm_loss --adaptative exp -bs 10 --os_gpu 1 --regime med

