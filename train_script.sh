
# Train additional tasks on standard latent
# python train.py -mt compression --tasks Zoom FFX -norm -bs 10


# Train multiple task without loss normalization
# python train.py -mt compression --tasks HiFiC Zoom FFX -norm -bs 10
# python train.py -mt compression --tasks HiFiC Zoom -norm -bs 10
# python train.py -mt compression --tasks HiFiC FFX -norm -bs 10


# Train with loss normalization
python train.py -mt compression --tasks HiFiC Zoom FFX -norm -bs 10 --norm_loss
python train.py -mt compression --tasks HiFiC Zoom -norm -bs 10 --norm_loss
python train.py -mt compression --tasks HiFiC FFX -norm -bs 10 --norm_loss


# Train with loss normalization using test-task only
# python train.py -mt compression --tasks HiFiC Zoom -norm -bs 10 --norm_loss --test_task
# python train.py -mt compression --tasks HiFiC FFX -norm -bs 10 --norm_loss --test_task