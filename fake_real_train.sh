# Train additional tasks on standard latent

python train.py -mt compression --tasks HiFiC -norm -bs 64 --os_gpu 0 --regime low --nframes 32 --dataset_type fake --fake_type Face2Face FaceShifter FaceSwap NeuralTextures Deepfakes DeepFakeDetection
python train.py -mt compression --tasks HiFiC -norm -bs 64 --os_gpu 0 --regime low --nframes 192 --dataset_type original

python train.py -mt compression --tasks HiFiC -norm -bs 64 --os_gpu 0 --regime high --nframes 32 --dataset_type fake --fake_type Face2Face FaceShifter FaceSwap NeuralTextures Deepfakes DeepFakeDetection
python train.py -mt compression --tasks HiFiC -norm -bs 64 --os_gpu 0 --regime high --nframes 192 --dataset_type original
