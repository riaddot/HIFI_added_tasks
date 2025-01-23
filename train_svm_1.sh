python code_images.py -mt compression --tasks HiFiC -norm -bs 32 --os_gpu 1 --regime high --fake_type FaceShifter -dc 1 -cls cnn
python code_images.py -mt compression --tasks HiFiC -norm -bs 32 --os_gpu 1 --regime high --fake_type FaceShifter -dc 1 -cls cnn --fft
python code_images.py -mt compression --tasks HiFiC -norm -bs 32 --os_gpu 1 --regime high --fake_type FaceShifter -dc 2 -cls cnn
python code_images.py -mt compression --tasks HiFiC -norm -bs 32 --os_gpu 1 --regime high --fake_type FaceShifter -dc 2 -cls cnn --fft

python code_images.py -mt compression --tasks HiFiC -norm -bs 32 --os_gpu 1 --regime high --fake_type Face2Face -dc 1 -cls cnn
python code_images.py -mt compression --tasks HiFiC -norm -bs 32 --os_gpu 1 --regime high --fake_type Face2Face -dc 1 -cls cnn --fft
python code_images.py -mt compression --tasks HiFiC -norm -bs 32 --os_gpu 1 --regime high --fake_type Face2Face -dc 2 -cls cnn
python code_images.py -mt compression --tasks HiFiC -norm -bs 32 --os_gpu 1 --regime high --fake_type Face2Face -dc 2 -cls cnn --fft

