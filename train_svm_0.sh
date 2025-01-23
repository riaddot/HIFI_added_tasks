python code_images.py -mt compression --tasks HiFiC -norm -bs 32 --os_gpu 0 --regime high --fake_type FaceSwap -dc 1 -cls cnn
python code_images.py -mt compression --tasks HiFiC -norm -bs 32 --os_gpu 0 --regime high --fake_type FaceSwap -dc 1 -cls cnn --fft
python code_images.py -mt compression --tasks HiFiC -norm -bs 32 --os_gpu 0 --regime high --fake_type FaceSwap -dc 2 -cls cnn
python code_images.py -mt compression --tasks HiFiC -norm -bs 32 --os_gpu 0 --regime high --fake_type FaceSwap -dc 2 -cls cnn --fft

python code_images.py -mt compression --tasks HiFiC -norm -bs 32 --os_gpu 0 --regime high --fake_type Deepfakes -dc 1 -cls cnn
python code_images.py -mt compression --tasks HiFiC -norm -bs 32 --os_gpu 0 --regime high --fake_type Deepfakes -dc 1 -cls cnn --fft
python code_images.py -mt compression --tasks HiFiC -norm -bs 32 --os_gpu 0 --regime high --fake_type Deepfakes -dc 2 -cls cnn
python code_images.py -mt compression --tasks HiFiC -norm -bs 32 --os_gpu 0 --regime high --fake_type Deepfakes -dc 2 -cls cnn --fft