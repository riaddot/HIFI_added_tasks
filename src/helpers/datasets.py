import os
import abc
import glob
import math
import logging
import numpy as np

from skimage.io import imread
import PIL
from PIL import Image
from tqdm import tqdm

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg
from torchvision.datasets.vision import VisionDataset

DIR = os.path.abspath(os.path.dirname(__file__))
COLOUR_BLACK = 0
COLOUR_WHITE = 1
NUM_DATASET_WORKERS = 4
SCALE_MIN = 0.75
SCALE_MAX = 0.95
DATASETS_DICT = {"lfw": "LFWPeople", "openimages": "OpenImages", "cityscapes": "CityScapes", 
                 "jetimages": "JetImages", "evaluation": "Evaluation"}
DATASETS = list(DATASETS_DICT.keys())

def get_dataset(dataset):
    """Return the correct dataset."""
    dataset = dataset.lower()
    try:
        # eval because stores name as string in order to put it at top of file
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        raise ValueError("Unknown dataset: {}".format(dataset))

def get_img_size(dataset):
    """Return the correct image size."""
    return get_dataset(dataset).img_size

def get_background(dataset):
    """Return the image background color."""
    return get_dataset(dataset).background_color

def exception_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def get_dataloaders(dataset, split='train', root=None, shuffle=True, pin_memory=True, 
                    batch_size=8, logger=logging.getLogger(__name__), normalize=False, **kwargs):
    """A generic data loader

    Parameters
    ----------
    dataset : {"openimages", "jetimages", "evaluation"}
        Name of the dataset to load

    root : str
        Path to the dataset root. If `None` uses the default one.

    kwargs :
        Additional arguments to `DataLoader`. Default values are modified.
    """
    pin_memory = pin_memory and torch.cuda.is_available  # only pin if GPU available
    Dataset = get_dataset(dataset)

    if root is None:
        dataset = Dataset(logger=logger, split=split, normalize=normalize, **kwargs)
    else:
        dataset = Dataset(root=root, logger=logger, split=split, normalize=normalize, **kwargs)

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=NUM_DATASET_WORKERS,
                      collate_fn=exception_collate_fn,
                      pin_memory=pin_memory)


class BaseDataset(Dataset, abc.ABC):
    """Base Class for datasets.

    Parameters
    ----------
    root : string
        Root directory of dataset.

    transforms_list : list
        List of `torch.vision.transforms` to apply to the data when loading it.
    """

    def __init__(self, root, transforms_list=[], mode='train', logger=logging.getLogger(__name__),
         **kwargs):
        self.root = root
        
        try:
            self.train_data = os.path.join(root, self.files["train"])
            self.test_data = os.path.join(root, self.files["test"])
            self.val_data = os.path.join(root, self.files["val"])
        except AttributeError:
            pass

        self.transforms = transforms.Compose(transforms_list)
        self.logger = logger


        if not os.path.isdir(root):
            raise ValueError('Files not found in specified directory: {}'.format(root))

    def __len__(self):
        return len(self.imgs)

    def __ndim__(self):
        return tuple(self.imgs.size())

    @abc.abstractmethod
    def __getitem__(self, idx):
        """Get the image of `idx`.

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.
        """
        pass

class Evaluation(BaseDataset):
    """
    Parameters
    ----------
    root : string
        Root directory of dataset.

    """

    def __init__(self, root=os.path.join(DIR, 'data'), normalize=False, **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        self.imgs = glob.glob(os.path.join(root, '*.jpg'))
        self.imgs += glob.glob(os.path.join(root, '*.png'))

        self.normalize = normalize

    def _transforms(self):
        """
        Up(down)scale and randomly crop to `crop_size` x `crop_size`
        """
        transforms_list = [transforms.ToTensor()]

        if self.normalize is True:
            transforms_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        """ TODO: This definitely needs to be optimized.
        Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        """
        # img values already between 0 and 255
        img_path = self.imgs[idx]
        filename = os.path.splitext(os.path.basename(img_path))[0]
        filesize = os.path.getsize(img_path)
        try:
            img = PIL.Image.open(img_path)
            img = img.convert('RGB') 
            W, H = img.size  # slightly confusing
            bpp = filesize * 8. / (H * W)

            test_transform = self._transforms()
            transformed = test_transform(img)
        except:
            print('Error reading input images!')
            return None

        return transformed, bpp, filename

class OpenImages(BaseDataset):
    """OpenImages dataset from [1].

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] https://storage.googleapis.com/openimages/web/factsfigures.html

    """
    files = {"train": "train", "test": "test", "val": "validation"}

    def __init__(self, root=os.path.join(DIR, 'data/openimages'), mode='train', crop_size=256, 
        normalize=False, **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        if mode == 'train':
            data_dir = self.train_data
        elif mode == 'validation':
            data_dir = self.val_data
        else:
            raise ValueError('Unknown mode!')

        self.imgs = glob.glob(os.path.join(data_dir, '*.jpg'))
        self.imgs += glob.glob(os.path.join(data_dir, '*.png'))

        self.crop_size = crop_size
        self.image_dims = (3, self.crop_size, self.crop_size)
        self.scale_min = SCALE_MIN
        self.scale_max = SCALE_MAX
        self.normalize = normalize

    def _transforms(self, scale, H, W):
        """
        Up(down)scale and randomly crop to `crop_size` x `crop_size`
        """
        transforms_list = [# transforms.ToPILImage(),
                           transforms.RandomHorizontalFlip(),
                           transforms.Resize((math.ceil(scale * H), math.ceil(scale * W))),
                           transforms.RandomCrop(self.crop_size),
                           transforms.ToTensor()]

        if self.normalize is True:
            transforms_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        """ TODO: This definitely needs to be optimized.
        Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        """
        # img values already between 0 and 255
        img_path = self.imgs[idx]
        filesize = os.path.getsize(img_path)
        try:
            # This is faster but less convenient
            # H X W X C `ndarray`
            # img = imread(img_path)
            # img_dims = img.shape
            # H, W = img_dims[0], img_dims[1]
            # PIL
            img = PIL.Image.open(img_path)
            img = img.convert('RGB') 
            W, H = img.size  # slightly confusing
            bpp = filesize * 8. / (H * W)

            shortest_side_length = min(H,W)

            minimum_scale_factor = float(self.crop_size) / float(shortest_side_length)
            scale_low = max(minimum_scale_factor, self.scale_min)
            scale_high = max(scale_low, self.scale_max)
            scale = np.random.uniform(scale_low, scale_high)

            dynamic_transform = self._transforms(scale, H, W)
            transformed = dynamic_transform(img)
        except:
            return None

        # apply random scaling + crop, put each pixel 
        # in [0.,1.] and reshape to (C x H x W)
        return transformed, bpp

class CityScapes(datasets.Cityscapes):
    """CityScapes wrapper. Docs: `datasets.Cityscapes.`"""
    img_size = (1, 32, 32)

    def _transforms(self, scale, H, W):
        """
        Up(down)scale and randomly crop to `crop_size` x `crop_size`
        """
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((math.ceil(scale * H), 
                               math.ceil(scale * W))),
            transforms.RandomCrop(self.crop_size),
            transforms.ToTensor(),
            ])

    def __init__(self, mode, root=os.path.join(DIR, 'data/cityscapes'), **kwargs):
        super().__init__(root,
                         split=mode,
                         transform=self._transforms(scale=np.random.uniform(0.5,1.0), 
                            H=512, W=1024))

def preprocess(root, size=(64, 64), img_format='JPEG', center_crop=None):
    """Preprocess a folder of images.

    Parameters
    ----------
    root : string
        Root directory of all images.

    size : tuple of int
        Size (width, height) to rescale the images. If `None` don't rescale.

    img_format : string
        Format to save the image in. Possible formats:
        https://pillow.readthedocs.io/en/3.1.x/handbook/image-file-formats.html.

    center_crop : tuple of int
        Size (width, height) to center-crop the images. If `None` don't center-crop.
    """
    imgs = []
    for ext in [".png", ".jpg", ".jpeg"]:
        imgs += glob.glob(os.path.join(root, '*' + ext))

    for img_path in tqdm(imgs):
        img = PIL.Image.open(img_path)
        width, height = img.size

        if size is not None and width != size[1] or height != size[0]:
            img = img.resize(size, PIL.Image.ANTIALIAS)

        if center_crop is not None:
            new_width, new_height = center_crop
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = (width + new_width) // 2
            bottom = (height + new_height) // 2

            img.crop((left, top, right, bottom))

        img.save(img_path, img_format)


        
class _LFW(VisionDataset):

    base_folder = "lfw-py"
    download_url_prefix = "http://vis-www.cs.umass.edu/lfw/"

    file_dict = {
        "original": ("lfw", "lfw.tgz", "a17d05bd522c52d84eca14327a23d494"),
        "funneled": ("lfw_funneled", "lfw-funneled.tgz", "1b42dfed7d15c9b2dd63d5e5840c86ad"),
        "deepfunneled": ("lfw-deepfunneled", "lfw-deepfunneled.tgz", "68331da3eb755a505a502b5aacb3c201"),
    }
    checksums = {
        "pairs.txt": "9f1ba174e4e1c508ff7cdf10ac338a7d",
        "pairsDevTest.txt": "5132f7440eb68cf58910c8a45a2ac10b",
        "pairsDevTrain.txt": "4f27cbf15b2da4a85c1907eb4181ad21",
        "people.txt": "450f0863dd89e85e73936a6d71a3474b",
        "peopleDevTest.txt": "e4bf5be0a43b5dcd9dc5ccfcb8fb19c5",
        "peopleDevTrain.txt": "54eaac34beb6d042ed3a7d883e247a21",
        "lfw-names.txt": "a6d0a479bd074669f656265a6e693f6d",
    }
    annot_file = {"10fold": "", "train": "DevTrain", "test": "DevTest"}
    names = "lfw-names.txt"

    def __init__(
        self,
        root: str,
        split: str,
        image_set: str,
        view: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
    ) -> None:
        super().__init__(os.path.join(root, self.base_folder), transform=transform, target_transform=target_transform)

        self.image_set = verify_str_arg(image_set.lower(), "image_set", self.file_dict.keys())
        images_dir, self.filename, self.md5 = self.file_dict[self.image_set]

        self.view = verify_str_arg(view.lower(), "view", ["people", "pairs"])
        self.split = verify_str_arg(split.lower(), "split", ["10fold", "train", "test"])
        self.labels_file = f"{self.view}{self.annot_file[self.split]}.txt"
        self.data: List[Any] = []

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self.images_dir = os.path.join(self.root, images_dir)

    def _loader(self, path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def _check_integrity(self) -> bool:
        st1 = check_integrity(os.path.join(self.root, self.filename), self.md5)
        st2 = check_integrity(os.path.join(self.root, self.labels_file), self.checksums[self.labels_file])
        if not st1 or not st2:
            return False
        if self.view == "people":
            return check_integrity(os.path.join(self.root, self.names), self.checksums[self.names])
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        url = f"{self.download_url_prefix}{self.filename}"
        download_and_extract_archive(url, self.root, filename=self.filename, md5=self.md5)
        download_url(f"{self.download_url_prefix}{self.labels_file}", self.root)
        if self.view == "people":
            download_url(f"{self.download_url_prefix}{self.names}", self.root)

    def _get_path(self, identity: str, no: Union[int, str]) -> str:
        return os.path.join(self.images_dir, identity, f"{identity}_{int(no):04d}.jpg")

    def extra_repr(self) -> str:
        return f"Alignment: {self.image_set}\nSplit: {self.split}"

    def __len__(self) -> int:
        return len(self.data)


class LFWPeople(_LFW):
    """`LFW <http://vis-www.cs.umass.edu/lfw/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``lfw-py`` exists or will be saved to if download is set to True.
        split (string, optional): The image split to use. Can be one of ``train``, ``test``,
            ``10fold`` (default).
        image_set (str, optional): Type of image funneling to use, ``original``, ``funneled`` or
            ``deepfunneled``. Defaults to ``funneled``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomRotation``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    def __init__(
        self,
        root: str,
        split: str = "10fold",
        image_set: str = "funneled",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        normalize = True,
        logger=None,
    ) -> None:
        super().__init__(root, split, image_set, "people", transform, target_transform, download)

        self.normalize = normalize
        self.logger = logger

        self.image_dims = (3, 256, 256)
        self.scale_min = SCALE_MIN
        self.scale_max = SCALE_MAX


        self.class_to_idx = self._get_classes()
        self.data, self.targets = self._get_people()

    def _get_people(self) -> Tuple[List[str], List[int]]:
        data, targets = [], []
        with open(os.path.join(self.root, self.labels_file)) as f:
            lines = f.readlines()
            n_folds, s = (int(lines[0]), 1) if self.split == "10fold" else (1, 0)

            for fold in range(n_folds):
                n_lines = int(lines[s])
                people = [line.strip().split("\t") for line in lines[s + 1 : s + n_lines + 1]]
                s += n_lines + 1
                for i, (identity, num_imgs) in enumerate(people):
                    for num in range(1, int(num_imgs) + 1):
                        img = self._get_path(identity, num)
                        data.append(img)
                        targets.append(self.class_to_idx[identity])

        return data, targets

    def _get_classes(self) -> Dict[str, int]:
        with open(os.path.join(self.root, self.names)) as f:
            lines = f.readlines()
            names = [line.strip().split()[0] for line in lines]
        class_to_idx = {name: i for i, name in enumerate(names)}
        return class_to_idx

    def _transforms(self):

        transforms_list = [# transforms.ToPILImage(),
                           transforms.RandomHorizontalFlip(),
                           transforms.Resize((256, 256)),
                           transforms.ToTensor()]

        if self.normalize is True:
            transforms_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        """ TODO: This definitely needs to be optimized.
        Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        """
        # img values already between 0 and 255
        # This is faster but less convenient
        # H X W X C `ndarray`
        # img = imread(img_path)
        # img_dims = img.shape
        # H, W = img_dims[0], img_dims[1]
        # PIL
        # img = PIL.Image.open(img_path)
        # img = img.convert('RGB') 
        img = self._loader(self.data[idx])
        target = self.targets[idx]
        W, H = img.size  # slightly confusing
        filesize = os.path.getsize(self.data[idx])
        bpp = filesize * 8. / (H * W)

        dynamic_transform = self._transforms()
        transformed = dynamic_transform(img)

        # apply random scaling + crop, put each pixel 
        # in [0.,1.] and reshape to (C x H x W)
        return transformed, bpp


class LFWPairs(_LFW):
    """`LFW <http://vis-www.cs.umass.edu/lfw/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``lfw-py`` exists or will be saved to if download is set to True.
        split (string, optional): The image split to use. Can be one of ``train``, ``test``,
            ``10fold``. Defaults to ``10fold``.
        image_set (str, optional): Type of image funneling to use, ``original``, ``funneled`` or
            ``deepfunneled``. Defaults to ``funneled``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomRotation``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    def __init__(
        self,
        root: str,
        split: str = "10fold",
        image_set: str = "funneled",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        logger=None,
    ) -> None:
        super().__init__(root, split, image_set, "pairs", transform, target_transform, download)

        self.logger = logger
        self.pair_names, self.data, self.targets = self._get_pairs(self.images_dir)

    def _get_pairs(self, images_dir: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[int]]:
        pair_names, data, targets = [], [], []
        with open(os.path.join(self.root, self.labels_file)) as f:
            lines = f.readlines()
            if self.split == "10fold":
                n_folds, n_pairs = lines[0].split("\t")
                n_folds, n_pairs = int(n_folds), int(n_pairs)
            else:
                n_folds, n_pairs = 1, int(lines[0])
            s = 1

            for fold in range(n_folds):
                matched_pairs = [line.strip().split("\t") for line in lines[s : s + n_pairs]]
                unmatched_pairs = [line.strip().split("\t") for line in lines[s + n_pairs : s + (2 * n_pairs)]]
                s += 2 * n_pairs
                for pair in matched_pairs:
                    img1, img2, same = self._get_path(pair[0], pair[1]), self._get_path(pair[0], pair[2]), 1
                    pair_names.append((pair[0], pair[0]))
                    data.append((img1, img2))
                    targets.append(same)
                for pair in unmatched_pairs:
                    img1, img2, same = self._get_path(pair[0], pair[1]), self._get_path(pair[2], pair[3]), 0
                    pair_names.append((pair[0], pair[2]))
                    data.append((img1, img2))
                    targets.append(same)

        return pair_names, data, targets

    def _transforms(self):

        transforms_list = [# transforms.ToPILImage(),
                           transforms.RandomHorizontalFlip(),
                           transforms.Resize((256, 256)),
                           transforms.ToTensor()]

        if self.normalize is True:
            transforms_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        """ TODO: This definitely needs to be optimized.
        Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        """
        # img values already between 0 and 255
        img_path = self.imgs[idx]
        filesize = os.path.getsize(img_path)
        try:
            # This is faster but less convenient
            # H X W X C `ndarray`
            # img = imread(img_path)
            # img_dims = img.shape
            # H, W = img_dims[0], img_dims[1]
            # PIL
            img = PIL.Image.open(img_path)
            img = img.convert('RGB') 
            W, H = img.size  # slightly confusing
            bpp = filesize * 8. / (H * W)

            shortest_side_length = min(H,W)

            minimum_scale_factor = float(self.crop_size) / float(shortest_side_length)
            scale_low = max(minimum_scale_factor, self.scale_min)
            scale_high = max(scale_low, self.scale_max)
            scale = np.random.uniform(scale_low, scale_high)

            dynamic_transform = self._transforms()
            transformed = dynamic_transform(img)
        except:
            return None

        # apply random scaling + crop, put each pixel 
        # in [0.,1.] and reshape to (C x H x W)
        return transformed, bpp