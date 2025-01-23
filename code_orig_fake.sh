# python code_images.py -mt compression --tasks HiFiC -norm -bs 10 --os_gpu 0 --regime high &
# python code_images.py -mt compression --tasks HiFiC -norm -bs 10 --os_gpu 0 --regime high -dc 2 &

# python code_images.py -mt compression --tasks HiFiC -norm -bs 10 --os_gpu 1 --regime high --dataset_type fake --fake_type Face2Face &
# python code_images.py -mt compression --tasks HiFiC -norm -bs 10 --os_gpu 1 --regime high --dataset_type fake --fake_type Face2Face -dc 2 &

python code_images.py -mt compression --tasks HiFiC -norm -bs 20 --os_gpu 0 --regime low --classifier kmeans --dataset_type fake --fake_type FaceShifter &
python code_images.py -mt compression --tasks HiFiC -norm -bs 20 --os_gpu 1 --regime low --classifier kmeans --dataset_type fake --fake_type FaceShifter -dc 2

python code_images.py -mt compression --tasks HiFiC -norm -bs 20 --os_gpu 0 --regime low --classifier kmeans --fft --dataset_type fake --fake_type FaceShifter &
python code_images.py -mt compression --tasks HiFiC -norm -bs 20 --os_gpu 1 --regime low --classifier kmeans --fft --dataset_type fake --fake_type FaceShifter -dc 2

# python code_images.py -mt compression --tasks HiFiC -norm -bs 10 --os_gpu 1 --regime high --dataset_type fake --fake_type FaceSwap &
# python code_images.py -mt compression --tasks HiFiC -norm -bs 10 --os_gpu 1 --regime high --dataset_type fake --fake_type FaceSwap -dc 2