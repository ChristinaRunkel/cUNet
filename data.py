import os
import cv2
import PIL
import torch
import random
import numpy as np
import torchvision
import torchvision.transforms.functional as TF
import scipy.ndimage
from typing import Tuple
from augmentations import ElasticTransformations, RandomRotationWithMask


cv2.setNumThreads(0)
random.seed(0)
torch.manual_seed(0)


class GLaSDataLoader(object):
    """
    Data loader for the GlaS histology image segmentation dataset.
    Handles loading, patch extraction, augmentation, and train/validation/test splits.

    Args:
        data_path (str): Path to dataset directory.
        patch_size (int or tuple): Size of patches to extract.
        dataset_repeat (int): Number of times to repeat dataset.
        images (array-like): Indices of images to use.
        validation (bool): Whether to use validation mode.
        test (bool): Whether to use test mode.
    """
    def __init__(
        self,
        data_path,
        patch_size,
        dataset_repeat=1,
        images=np.arange(0, 70),
        validation=False,
        test=False,
    ):
        if test:
            self.image_fname = os.path.join(data_path, "testA_")
        else:
            self.image_fname = os.path.join(data_path, "train_")
        self.images = images

        self.patch_size = patch_size
        self.repeat = dataset_repeat
        self.validation = validation
        self.image_mask_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                RandomRotationWithMask(45, resample=False, expand=False, center=None),
                ElasticTransformations(2000, 60),
                torchvision.transforms.ToTensor(),
            ]
        )
        self.image_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ColorJitter(
                    brightness=0.3, contrast=0.2, saturation=0.1, hue=0.1
                ),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __getitem__(self, index):
        image, mask = self.index_to_filename(index)
        image, mask = self.open_and_resize(image, mask)
        image, mask = self.pad_image(image, mask)
        label, patch = self.apply_data_augmentation(image, mask)
        label = self.create_eroded_mask(label, mask)
        patch, label = self.extract_random_region(image, patch, label)
        return patch, label.float()

    def index_to_filename(self, index):
        """Helper function to retrieve filenames from index"""
        index_img = index // self.repeat
        index_img = self.images[index_img]
        index_str = str(index_img.item() + 1)

        image = self.image_fname + index_str + ".bmp"
        mask = self.image_fname + index_str + "_anno.bmp"
        return image, mask

    def open_and_resize(self, image, mask):
        """Helper function to pad smaller image to the correct size"""
        image = PIL.Image.open(image)
        mask = PIL.Image.open(mask)

        ratio = 775 / 512
        new_size = (
            int(round(image.size[0] / ratio)),
            int(round(image.size[1] / ratio)),
        )

        image = image.resize(new_size)
        mask = mask.resize(new_size)

        image = np.array(image)
        mask = np.array(mask)
        return image, mask

    def pad_image(self, image, mask):
        """Helper function to pad smaller image to the correct size"""
        if not self.validation:
            pad_h = max(self.patch_size[0] - image.shape[0], 128)
            pad_w = max(self.patch_size[1] - image.shape[1], 128)
        else:
            # we pad more than needed to later do translation augmentation
            pad_h = max((self.patch_size[0] - image.shape[0]) // 2 + 1, 0)
            pad_w = max((self.patch_size[1] - image.shape[1]) // 2 + 1, 0)

        padded_image = np.pad(
            image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="reflect"
        )
        mask = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
        return padded_image, mask

    def apply_data_augmentation(self, image, mask):
        """Helper function to apply all configured data augmentations on both mask and image"""
        patch = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255
        n_glands = mask.max()
        label = torch.from_numpy(mask).float() / n_glands

        if not self.validation:
            patch_label_concat = torch.cat((patch, label[None, :, :].float()))
            patch_label_concat = self.image_mask_transforms(patch_label_concat)
            patch, label = patch_label_concat[0:3], np.round(
                patch_label_concat[3] * n_glands
            )
            patch = self.image_transforms(patch)
        else:
            label *= n_glands
        return label, patch

    def create_eroded_mask(self, label, mask):
        """Helper function to create a mask where every gland is eroded"""
        boundaries = torch.zeros(label.shape)
        for i in np.unique(mask):
            if i == 0:
                continue  # the first label is background
            gland_mask = (label == i).float()
            binarized_mask_border = scipy.ndimage.morphology.binary_erosion(
                gland_mask, structure=np.ones((13, 13)), border_value=1
            )

            binarized_mask_border = torch.from_numpy(
                binarized_mask_border.astype(np.float32)
            )
            boundaries[label == i] = binarized_mask_border[label == i]

        label = (label > 0).float()
        label = torch.stack((boundaries, label))
        return label

    def extract_random_region(self, image, patch, label):
        """Helper function to perform translation data augmentation"""
        if not self.validation:
            loc_y = random.randint(0, image.shape[0] - self.patch_size[0])
            loc_x = random.randint(0, image.shape[1] - self.patch_size[1])
        else:
            loc_y, loc_x = 0, 0

        patch = patch[
            :, loc_y : loc_y + self.patch_size[0], loc_x : loc_x + self.patch_size[1]
        ]
        label = label[
            :, loc_y : loc_y + self.patch_size[0], loc_x : loc_x + self.patch_size[1]
        ]
        return patch, label

    def __len__(self):
        return len(self.images) * self.repeat


class STAREDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for the STARE retinal vessel segmentation dataset.
    Handles loading, train/validation splits, and optional transforms.

    Args:
        inp_path (str): Path to input images.
        gt_path (str): Path to ground truth masks.
        train_set_idx (list): Indices for training set.
        val_set_idx (list): Indices for validation set.
        train (bool): Whether to use training set.
        transforms (torchvision.transforms): Optional transforms to apply.
    """
    def __init__(
        self,
        inp_path: str,
        gt_path: str,
        train_set_idx: list,
        val_set_idx: list,
        train: bool,
        transforms: torchvision.transforms,
        normalise: torchvision.transforms,
    ) -> None:
        super(STAREDataset, self).__init__()
        self.transforms = transforms
        self.normalise = normalise
        self.img_tensor, self.lbl_tensor = self.load_data(
            inp_path=inp_path, gt_path=gt_path
        )
        (
            self.trainset,
            self.valset,
            self.trainset_lbl,
            self.valset_lbl,
        ) = self.train_val_split(train_set_idx=train_set_idx, val_set_idx=val_set_idx)
        if train:
            self.data = self.trainset
            self.lbl = self.trainset_lbl
        else:
            self.data = self.valset
            self.lbl = self.valset_lbl

    def load_data(self, inp_path: str, gt_path: str) -> torch.Tensor:
        img_arr = []
        lbl_arr = []
        assert os.path.isdir(inp_path) and os.path.isdir(gt_path)
        for img_name in os.listdir(inp_path):
            img_path = os.path.join(inp_path, img_name)
            lbl_path = os.path.join(gt_path, img_name[:-3] + "ah.ppm")
            if os.path.isfile(img_path):
                img = cv2.imread(img_path)
                img = np.transpose(np.array(img, dtype=np.float32), (2, 0, 1))
                img = torch.tensor(img / 255)
                img = TF.resize(img=img, size=(512, 512))
                img_cp = torch.zeros_like(img)
                for i in range(2):
                    img_cp[i] = img[2 - i]
                img_arr.append(img_cp)

            if os.path.isfile(lbl_path):
                lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
                lbl = np.array(lbl, dtype=np.float32)[np.newaxis, :, :]
                lbl_new = (lbl >= 127.5).astype(np.float16)

                lbl_new = torch.tensor(lbl_new)
                lbl_new = TF.resize(
                    img=lbl_new,
                    size=(512, 512),
                    interpolation=TF.InterpolationMode.NEAREST,
                )

                lbl_arr.append(lbl_new)

        img_tensor = torch.stack(img_arr)
        lbl_tensor = torch.stack(lbl_arr)

        return img_tensor, lbl_tensor

    # Crop random region for training
    def crop_random_region(
        self, img: torch.Tensor, lbl: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_start = random.randint(0, 92)
        y_start = random.randint(0, 11)

        x_end = 93 - x_start
        y_end = 12 - y_start

        img_cropped, lbl_cropped = (
            img[:, x_start:-x_end, y_start:-y_end],
            lbl[:, x_start:-x_end, y_start:-y_end],
        )

        return img_cropped, lbl_cropped

    # Crop centre region for validation and testing
    def crop_centre_region(
        self, img: torch.Tensor, lbl: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_cropped, lbl_cropped = img[:, 46:-47, 6:-6], lbl[:, 46:-47, 6:-6]
        return img_cropped, lbl_cropped

    def train_val_split_cropped(
        self, train_set_idx: list, val_set_idx: list
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        train_img_arr = []
        train_lbl_arr = []
        val_img_arr = []
        val_lbl_arr = []

        for i in train_set_idx:
            img_cropped, lbl_cropped = self.crop_random_region(
                img=self.img_tensor[i], lbl=self.lbl_tensor[i]
            )
            train_img_arr.append(img_cropped.numpy())
            train_lbl_arr.append(lbl_cropped.numpy())

        for j in val_set_idx:
            img_cropped, lbl_cropped = self.crop_centre_region(
                img=self.img_tensor[j], lbl=self.lbl_tensor[j]
            )
            val_img_arr.append(img_cropped.numpy())
            val_lbl_arr.append(lbl_cropped.numpy())

        trainset, valset = torch.tensor(
            np.array(train_img_arr, dtype=np.float32)
        ), torch.tensor(np.array(val_img_arr, dtype=np.float32))
        trainset_lbl, valset_lbl = torch.tensor(
            np.array(train_lbl_arr, dtype=np.float32)
        ), torch.tensor(np.array(val_lbl_arr, dtype=np.float32))
        # trainset, valset = trainset.permute(1, 0, 2, 3, 4), valset.permute(1, 0, 2, 3, 4)
        return trainset, valset, trainset_lbl, valset_lbl

    def train_val_split(
        self, train_set_idx: list, val_set_idx: list
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Splits image and label tensors into training and validation sets based on provided indices.

        Args:
            train_set_idx (list): Indices for training set.
            val_set_idx (list): Indices for validation set.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Training and validation image and label tensors.
        """
        train_img_arr = []
        train_lbl_arr = []
        val_img_arr = []
        val_lbl_arr = []

        for i in train_set_idx:
            train_img_arr.append(self.img_tensor[i].numpy())
            train_lbl_arr.append(self.lbl_tensor[i].numpy())

        for j in val_set_idx:
            val_img_arr.append(self.img_tensor[j].numpy())
            val_lbl_arr.append(self.lbl_tensor[j].numpy())

        trainset, valset = torch.tensor(
            np.array(train_img_arr, dtype=np.float32)
        ), torch.tensor(np.array(val_img_arr, dtype=np.float32))
        trainset_lbl, valset_lbl = torch.tensor(
            np.array(train_lbl_arr, dtype=np.float32)
        ), torch.tensor(np.array(val_lbl_arr, dtype=np.float32))
        return trainset, valset, trainset_lbl, valset_lbl

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = {"data": self.data[idx], "lbl": self.lbl[idx]}
        if self.normalise is not None:
            sample["data"] = self.normalise(sample["data"])
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample["data"], sample["lbl"]

    def __len__(self) -> int:
        return len(self.data)


class PolypDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for polyp segmentation tasks.
    Handles loading, normalization, and optional transforms for polyp images and masks.

    Args:
        ... (see class __init__ for details)
    """
    def __init__(
        self,
        inp_path: str,
        gt_path: str,
        train_set_idx: list,
        val_set_idx: list,
        train: bool,
        transforms: torchvision.transforms,
        normalise: torchvision.transforms,
    ) -> None:
        super().__init__()
        self.train = train
        self.transforms = transforms
        self.normalise = normalise
        self.img_tensor, self.lbl_tensor = self.load_data(
            inp_path=inp_path, gt_path=gt_path
        )
        (
            self.trainset,
            self.valset,
            self.trainset_lbl,
            self.valset_lbl,
        ) = self.train_val_split(train_set_idx=train_set_idx, val_set_idx=val_set_idx)
        if train:
            self.data = self.trainset
            self.lbl = self.trainset_lbl
        else:
            self.data = self.valset
            self.lbl = self.valset_lbl

        # Crop random region for training

    def random_crop(
        self, img: torch.Tensor, lbl: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_start = random.randint(0, img.shape[-2] - 224)
        y_start = random.randint(0, img.shape[-1] - 224)

        x_end = x_start + 224
        y_end = y_start + 224

        img_cropped, lbl_cropped = (
            img[:, x_start:x_end, y_start:y_end],
            lbl[:, x_start:x_end, y_start:y_end],
        )

        return img_cropped, lbl_cropped

    # Crop centre region for validation and testing
    def centre_crop(
        self, img: torch.Tensor, lbl: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = img.shape[-2]
        y = img.shape[-1]
        img_cropped, lbl_cropped = (
            img[:, x // 2 - 128 : x // 2 + 128, y // 2 - 128 : y // 2 + 128],
            lbl[:, x // 2 - 128 : x // 2 + 128, y // 2 - 128 : y // 2 + 128],
        )
        return img_cropped, lbl_cropped

    def load_data(self, inp_path: str, gt_path: str) -> torch.Tensor:
        img_arr = []
        lbl_arr = []

        assert os.path.isdir(inp_path) and os.path.isdir(gt_path)
        for img_name in os.listdir(inp_path):
            img_path = os.path.join(inp_path, img_name)
            lbl_path = os.path.join(gt_path, img_name)
            if os.path.isfile(img_path) and os.path.isfile(lbl_path):
                img = cv2.imread(img_path)
                img = np.transpose(np.array(img / 255, dtype=np.float32), (2, 0, 1))
                img = torch.tensor(img)
                img = TF.resize(img=img, size=(256, 256))
                img_cp = torch.zeros_like(img)
                for i in range(2):
                    img_cp[i] = img[2 - i]

                lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
                lbl = np.array(lbl, dtype=np.float32)[np.newaxis, :, :]
                lbl_new = (lbl >= 127.5).astype(np.float16)

                lbl_new = torch.tensor(lbl_new)
                lbl_new = TF.resize(
                    img=lbl_new,
                    size=(256, 256),
                    interpolation=TF.InterpolationMode.NEAREST,
                )
                # crop to 224x224
                # if self.train:
                #    img_cp, lbl_new = self.random_crop(img=img_cp, lbl=lbl_new)
                img_arr.append(img_cp)
                lbl_arr.append(lbl_new)
        img_tensor = torch.stack(img_arr)
        lbl_tensor = torch.stack(lbl_arr)

        return img_tensor, lbl_tensor

    def train_val_split(
        self, train_set_idx: list, val_set_idx: list
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        train_img_arr = []
        train_lbl_arr = []
        val_img_arr = []
        val_lbl_arr = []

        for i in train_set_idx:
            train_img_arr.append(self.img_tensor[i].numpy())
            train_lbl_arr.append(self.lbl_tensor[i].numpy())

        for j in val_set_idx:
            val_img_arr.append(self.img_tensor[j].numpy())
            val_lbl_arr.append(self.lbl_tensor[j].numpy())

        trainset, valset = torch.tensor(
            np.array(train_img_arr, dtype=np.float32)
        ), torch.tensor(np.array(val_img_arr, dtype=np.float32))
        trainset_lbl, valset_lbl = torch.tensor(
            np.array(train_lbl_arr, dtype=np.float32)
        ), torch.tensor(np.array(val_lbl_arr, dtype=np.float32))
        return trainset, valset, trainset_lbl, valset_lbl

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = {"data": self.data[idx], "lbl": self.lbl[idx]}

        if self.transforms is not None:
            sample = self.transforms(sample)
            sample["lbl"] = (sample["lbl"] >= 0.5).to(torch.float)
        if self.normalise is not None:
            sample["data"] = self.normalise(sample["data"])

        return sample["data"], sample["lbl"]

    def __len__(self) -> int:
        return len(self.data)


class NucleiDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for nuclei segmentation tasks.
    Handles loading, train/validation splits, and optional transforms for nuclei images and masks.

    Args:
        ... (see class __init__ for details)
    """
    def __init__(
        self,
        inp_path: str,
        train_set_idx: list,
        val_set_idx: list,
        train: bool,
        transforms: torchvision.transforms,
        normalise: torchvision.transforms,
    ) -> None:
        super().__init__()
        self.transforms = transforms
        self.normalise = normalise

        self.img_tensor, self.lbl_tensor = self.load_data(inp_path=inp_path)

        (
            self.trainset,
            self.valset,
            self.trainset_lbl,
            self.valset_lbl,
        ) = self.train_val_split(train_set_idx=train_set_idx, val_set_idx=val_set_idx)

        if train:
            self.data = self.trainset
            self.lbl = self.trainset_lbl
        else:
            self.data = self.valset
            self.lbl = self.valset_lbl

    def load_data(self, inp_path: str) -> torch.Tensor:
        img_arr = []
        lbl_arr = []

        assert os.path.isdir(inp_path)
        for img_name in os.listdir(inp_path):
            img_path = os.path.join(inp_path, img_name, "images", img_name + ".png")
            lbl_path = os.path.join(inp_path, img_name, "masks")
            if os.path.isfile(img_path):
                img = cv2.imread(img_path)
                img = cv2.resize(img, (256, 256))
                img = np.transpose(np.array(img / 255, dtype=np.float32), (2, 0, 1))
                img = torch.tensor(img)
                img_arr.append(img)  # changed to range [0,1]

            mask_full = torch.zeros((1, 256, 256))
            if os.path.isdir(lbl_path):
                for f in os.listdir(lbl_path):
                    lbl_img_path = os.path.join(lbl_path, f)
                    lbl = cv2.imread(lbl_img_path, cv2.IMREAD_GRAYSCALE)
                    lbl = np.array(lbl, dtype=np.float32)[np.newaxis, :, :]
                    lbl_new = (lbl >= 127.5).astype(np.float16)
                    lbl_new = torch.tensor(lbl_new)
                    lbl_new = TF.resize(
                        img=lbl_new,
                        size=(256, 256),
                        interpolation=TF.InterpolationMode.NEAREST,
                    )
                    mask_full = torch.maximum(mask_full, lbl_new)
                lbl_arr.append(mask_full)
        img_tensor = torch.stack(img_arr)
        lbl_tensor = torch.stack(lbl_arr)

        return img_tensor, lbl_tensor

    def train_val_split(
        self, train_set_idx: list, val_set_idx: list
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        train_img_arr = []
        train_lbl_arr = []
        val_img_arr = []
        val_lbl_arr = []

        for i in train_set_idx:
            train_img_arr.append(self.img_tensor[i].numpy())
            train_lbl_arr.append(self.lbl_tensor[i].numpy())

        for j in val_set_idx:
            val_img_arr.append(self.img_tensor[j].numpy())
            val_lbl_arr.append(self.lbl_tensor[j].numpy())

        trainset, valset = torch.tensor(
            np.array(train_img_arr, dtype=np.float32)
        ), torch.tensor(np.array(val_img_arr, dtype=np.float32))
        trainset_lbl, valset_lbl = torch.tensor(
            np.array(train_lbl_arr, dtype=np.float32)
        ), torch.tensor(np.array(val_lbl_arr, dtype=np.float32))
        return trainset, valset, trainset_lbl, valset_lbl

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = {"data": self.data[idx], "lbl": self.lbl[idx]}
        if self.normalise is not None:
            sample["data"] = self.normalise(sample["data"])
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample["data"], sample["lbl"]

    def __len__(self) -> int:
        return len(self.data)


class BreastCancerDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for breast cancer segmentation tasks.
    Handles loading, train/validation splits, and optional transforms for breast cancer images and masks.

    Args:
        ... (see class __init__ for details)
    """
    def __init__(
        self,
        inp_path: str,
        train_set_idx: list,
        val_set_idx: list,
        train: bool,
        transforms: torchvision.transforms,
        normalise: torchvision.transforms,
    ) -> None:
        super().__init__()
        self.train = train
        self.transforms = transforms
        self.normalise = normalise
        self.img_tensor, self.lbl_tensor = self.load_data(inp_path=inp_path)
        (
            self.trainset,
            self.valset,
            self.trainset_lbl,
            self.valset_lbl,
        ) = self.train_val_split(train_set_idx=train_set_idx, val_set_idx=val_set_idx)
        if train:
            self.data = self.trainset
            self.lbl = self.trainset_lbl
        else:
            self.data = self.valset
            self.lbl = self.valset_lbl

    def random_crop(
        self, img: torch.Tensor, lbl: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_start = random.randint(0, img.shape[-2] - 224)
        y_start = random.randint(0, img.shape[-1] - 224)

        x_end = x_start + 224
        y_end = y_start + 224

        img_cropped, lbl_cropped = (
            img[:, x_start:x_end, y_start:y_end],
            lbl[:, x_start:x_end, y_start:y_end],
        )

        return img_cropped, lbl_cropped

    def load_data(self, inp_path: str) -> torch.Tensor:
        img_arr = []
        lbl_arr = []

        assert os.path.isdir(inp_path)
        for subdir in os.listdir(inp_path):
            if subdir != "normal":
                subdir = os.path.join(inp_path, subdir)
                for img_name in os.listdir(subdir):
                    if "mask" not in img_name:
                        img_path = os.path.join(inp_path, subdir, img_name)
                        lbl_path = os.path.join(
                            inp_path, subdir, img_name[:-4] + "_mask.png"
                        )
                        lbl_path_add_mask1 = os.path.join(
                            inp_path, subdir, img_name[:-4] + "_mask_1.png"
                        )
                        lbl_path_add_mask2 = os.path.join(
                            inp_path, subdir, img_name[:-4] + "_mask_2.png"
                        )

                        if os.path.isfile(img_path):
                            img = cv2.imread(img_path)
                            img = cv2.resize(img, (256, 256))
                            img = np.transpose(
                                np.array(img / 255, dtype=np.float32), (2, 0, 1)
                            )
                            img = torch.tensor(img)
                            img_arr.append(img)  # changed to range [0,1]

                        mask_full = torch.zeros((1, 256, 256))
                        if os.path.isfile(lbl_path):
                            lbl_img_path = os.path.join(lbl_path)
                            lbl = cv2.imread(lbl_img_path, cv2.IMREAD_GRAYSCALE)
                            lbl = np.array(lbl, dtype=np.float32)[np.newaxis, :, :]
                            lbl_new = (lbl >= 127.5).astype(np.float16)
                            lbl_new = torch.tensor(lbl_new)
                            lbl_new = TF.resize(
                                img=lbl_new,
                                size=(256, 256),
                                interpolation=TF.InterpolationMode.NEAREST,
                            )
                            mask_full = torch.maximum(mask_full, lbl_new)

                        if os.path.isfile(lbl_path_add_mask1):
                            lbl_img_path = os.path.join(lbl_path_add_mask1)
                            lbl = cv2.imread(lbl_img_path, cv2.IMREAD_GRAYSCALE)
                            lbl = np.array(lbl, dtype=np.float32)[np.newaxis, :, :]
                            lbl_new = (lbl >= 127.5).astype(np.float16)
                            lbl_new = torch.tensor(lbl_new)
                            lbl_new = TF.resize(
                                img=lbl_new,
                                size=(256, 256),
                                interpolation=TF.InterpolationMode.NEAREST,
                            )
                            mask_full = torch.maximum(mask_full, lbl_new)
                        if os.path.isfile(lbl_path_add_mask2):
                            lbl_img_path = os.path.join(lbl_path_add_mask2)
                            lbl = cv2.imread(lbl_img_path, cv2.IMREAD_GRAYSCALE)
                            lbl = np.array(lbl, dtype=np.float32)[np.newaxis, :, :]
                            lbl_new = (lbl >= 127.5).astype(np.float16)
                            lbl_new = torch.tensor(lbl_new)
                            lbl_new = TF.resize(
                                img=lbl_new,
                                size=(256, 256),
                                interpolation=TF.InterpolationMode.NEAREST,
                            )
                            mask_full = torch.maximum(mask_full, lbl_new)

                        lbl_arr.append(mask_full)
                    else:
                        pass

        img_tensor = torch.stack(img_arr)
        lbl_tensor = torch.stack(lbl_arr)

        return img_tensor, lbl_tensor

    def train_val_split(
        self, train_set_idx: list, val_set_idx: list
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        train_img_arr = []
        train_lbl_arr = []
        val_img_arr = []
        val_lbl_arr = []

        for i in train_set_idx:
            train_img_arr.append(self.img_tensor[i])
            train_lbl_arr.append(self.lbl_tensor[i])

        for j in val_set_idx:
            val_img_arr.append(self.img_tensor[j])
            val_lbl_arr.append(self.lbl_tensor[j])

        trainset, valset = torch.stack(train_img_arr), torch.stack(val_img_arr)
        trainset_lbl, valset_lbl = torch.stack(train_lbl_arr), torch.stack(val_lbl_arr)

        return trainset, valset, trainset_lbl, valset_lbl

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = {"data": self.data[idx], "lbl": self.lbl[idx]}
        if self.normalise is not None:
            sample["data"] = self.normalise(sample["data"])
        # if self.train:
        #    sample['data'], sample['lbl'] = self.random_crop(img=sample['data'], lbl=sample['lbl'])
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample["data"], sample["lbl"]

    def __len__(self) -> int:
        return len(self.data)


class MelanomaDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for melanoma segmentation tasks.
    Handles loading, train/validation splits, and optional transforms for melanoma images and masks.

    Args:
        ... (see class __init__ for details)
    """
    def __init__(
        self,
        inp_path: str,
        gt_path: str,
        train_set_idx: list,
        val_set_idx: list,
        train: bool,
        transforms: torchvision.transforms,
        normalise: torchvision.transforms,
        test: bool = False,
    ) -> None:
        super().__init__()
        self.transforms = transforms
        self.normalise = normalise
        self.img_tensor, self.lbl_tensor = self.load_data(
            inp_path=inp_path, gt_path=gt_path
        )
        if not test:
            (
                self.trainset,
                self.valset,
                self.trainset_lbl,
                self.valset_lbl,
            ) = self.train_val_split(
                train_set_idx=train_set_idx, val_set_idx=val_set_idx
            )
            if train:
                self.data = self.trainset
                self.lbl = self.trainset_lbl
            else:
                self.data = self.valset
                self.lbl = self.valset_lbl
        else:
            self.data = self.img_tensor
            self.lbl = self.lbl_tensor

    def load_data(self, inp_path: str, gt_path: str) -> torch.Tensor:
        img_arr = []
        lbl_arr = []

        assert os.path.isdir(inp_path) and os.path.isdir(gt_path)
        for img_name in os.listdir(inp_path):
            img_path = os.path.join(inp_path, img_name)
            lbl_path = os.path.join(gt_path, img_name[:-4] + "_Segmentation.png")
            if os.path.isfile(img_path):
                img = cv2.imread(img_path)
                img = cv2.resize(img, (512, 512))
                img = np.transpose(np.array(img / 255, dtype=np.float32), (2, 0, 1))
                img = torch.tensor(img)
                img_arr.append(img)

            if os.path.isfile(lbl_path):
                lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
                lbl = np.array(lbl, dtype=np.float32)[np.newaxis, :, :]
                lbl_new = (lbl >= 127.5).astype(np.float16)
                lbl_new = torch.tensor(lbl_new)
                lbl_new = TF.resize(
                    img=lbl_new,
                    size=(512, 512),
                    interpolation=TF.InterpolationMode.NEAREST,
                )
                lbl_arr.append(lbl_new)
        img_tensor = torch.stack(img_arr)
        lbl_tensor = torch.stack(lbl_arr)

        return img_tensor, lbl_tensor

    def train_val_split(
        self, train_set_idx: list, val_set_idx: list
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        train_img_arr = []
        train_lbl_arr = []
        val_img_arr = []
        val_lbl_arr = []

        for i in train_set_idx:
            train_img_arr.append(self.img_tensor[i].numpy())
            train_lbl_arr.append(self.lbl_tensor[i].numpy())

        for j in val_set_idx:
            val_img_arr.append(self.img_tensor[j].numpy())
            val_lbl_arr.append(self.lbl_tensor[j].numpy())

        trainset, valset = torch.tensor(
            np.array(train_img_arr, dtype=np.float32)
        ), torch.tensor(np.array(val_img_arr, dtype=np.float32))
        trainset_lbl, valset_lbl = torch.tensor(
            np.array(train_lbl_arr, dtype=np.float32)
        ), torch.tensor(np.array(val_lbl_arr, dtype=np.float32))
        return trainset, valset, trainset_lbl, valset_lbl

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = {"data": self.data[idx], "lbl": self.lbl[idx]}
        if self.normalise is not None:
            sample["data"] = self.normalise(sample["data"])
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample["data"], sample["lbl"]

    def __len__(self) -> int:
        return len(self.data)
