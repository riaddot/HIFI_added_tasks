
# Train additional tasks on standard latent
python train.py -mt compression --tasks Zoom FFX -norm -bs 10 #--n_epochs 80


# Train multiple task without loss normalization
python train.py -mt compression --tasks HiFiC Zoom FFX -norm -bs 10 #--n_epochs 80
python train.py -mt compression --tasks HiFiC Zoom -norm -bs 10 #--n_epochs 80
python train.py -mt compression --tasks HiFiC FFX -norm -bs 10 #--n_epochs 80


# Train with loss normalization
python train.py -mt compression --tasks HiFiC Zoom FFX -norm -bs 10 --norm_loss #--n_epochs 80
python train.py -mt compression --tasks HiFiC Zoom -norm -bs 10 --norm_loss #--n_epochs 80
python train.py -mt compression --tasks HiFiC FFX -norm -bs 10 --norm_loss #--n_epochs 80


# Train with loss normalization using test-task only
python train.py -mt compression --tasks HiFiC Zoom --test_task -bs 10 -norm #--n_epochs 80
python train.py -mt compression --tasks HiFiC FFX --test_task -bs 10 -norm #--n_epochs 80