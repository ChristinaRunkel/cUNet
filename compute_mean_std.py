import os
import argparse
import torch
import torchvision
from transforms import *
from data import *

parser = argparse.ArgumentParser(description="USODE compute mean, std")
parser.add_argument(
    "--dataset",
    type=str,
    help="Dataset to use.",
)
parser.add_argument("--batch-size", type=int, default=1)


def compute_mean_std(
    data: torchvision.datasets, img_size: int, batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the mean and standard deviation per channel for a dataset.

    Args:
        data (torchvision.datasets): The dataset to compute statistics for.
        img_size (int): The size of the images (assumes square images).
        batch_size (int): Batch size for loading the data.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Mean and standard deviation tensors for each channel.
    """
    num_pixels = len(data) * img_size**2

    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, num_workers=1)
    total_sum0 = 0.0
    total_sum1 = 0.0
    total_sum2 = 0.0

    sum_squared_error0 = 0.0
    sum_squared_error1 = 0.0
    sum_squared_error2 = 0.0

    for batch in dataloader:
        total_sum0 += batch[0][:, 0].sum()
        total_sum1 += batch[0][:, 1].sum()
        total_sum2 += batch[0][:, 2].sum()

    mean0 = total_sum0 / num_pixels
    mean1 = total_sum1 / num_pixels
    mean2 = total_sum2 / num_pixels

    mean_tensor = torch.tensor([mean0, mean1, mean2])

    for batch in dataloader:
        sum_squared_error0 += ((batch[0][:, 0] - mean0).pow(2)).sum()
        sum_squared_error1 += ((batch[0][:, 1] - mean1).pow(2)).sum()
        sum_squared_error2 += ((batch[0][:, 2] - mean2).pow(2)).sum()

    std0 = torch.sqrt(sum_squared_error0 / num_pixels)
    std1 = torch.sqrt(sum_squared_error1 / num_pixels)
    std2 = torch.sqrt(sum_squared_error2 / num_pixels)

    std_tensor = torch.tensor([std0, std1, std2])

    return mean_tensor, std_tensor


if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(0)
    if args.dataset == "cells":
        data_path = "<PATH_TO_CELLS_DATASET>"
        # val_set_idx = torch.LongTensor(10).random_(0, 85)
        train_set_idx = torch.arange(0, 85)
        # overlapping = (train_set_idx[..., None] == val_set_idx).any(-1)
        # train_set_idx = torch.masked_select(train_set_idx, ~overlapping)

        trainset = GLaSDataLoader(
            data_path=data_path,
            patch_size=(352, 352),
            dataset_repeat=1,
            images=train_set_idx,
        )
        img_size = 352
    elif args.dataset == "STARE":
        data_path = "<PATH_TO_STARE_DATASET>"
        inp_path = os.path.join(data_path, "inputs")
        gt_path = os.path.join(data_path, "GT")
        perm_idx = torch.randperm(20)
        val_set_idx = []
        train_set_idx = perm_idx
        transforms = None
        trainset = STAREDataset(
            inp_path=inp_path,
            gt_path=gt_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=True,
            transforms=transforms,
        )
        img_size = 512
    elif args.dataset == "polyp":
        data_path = "<PATH_TO_POLYP_DATASET>"
        inp_path = os.path.join(data_path, "images")
        gt_path = os.path.join(data_path, "masks")
        perm_idx = torch.randperm(1000)
        val_set_idx = []
        train_set_idx = perm_idx
        transforms = None
        trainset = PolypDataset(
            inp_path=inp_path,
            gt_path=gt_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=True,
            transforms=transforms,
        )
        img_size = 256
    elif args.dataset == "nuclei":
        data_path = "<PATH_TO_NUCLEI_DATASET>"
        perm_idx = torch.randperm(670)
        train_set_idx = perm_idx
        transforms = None
        trainset = NucleiDataset(
            inp_path=data_path,
            train_set_idx=train_set_idx,
            val_set_idx=[],
            train=True,
            transforms=transforms,
        )
        img_size = 256

    elif args.dataset == "breast_cancer":
        data_path = "<PATH_TO_BREAST_CANCER_DATASET>"
        perm_idx = torch.randperm(647)
        val_set_idx = []
        train_set_idx = perm_idx
        transforms = None
        trainset = BreastCancerDataset(
            inp_path=data_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=True,
            transforms=transforms,
        )
        img_size = 256
    elif args.dataset == "melanoma_segmentation":
        inp_path = "<PATH_TO_MELANOMA_TRAIN_DATA>"
        gt_path = "<PATH_TO_MELANOMA_TRAIN_GROUNDTRUTH>"
        perm_idx = torch.randperm(900)
        val_set_idx = []
        train_set_idx = perm_idx
        transforms = None
        trainset = MelanomaDataset(
            inp_path=inp_path,
            gt_path=gt_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=True,
            transforms=transforms,
        )
        img_size = 512
    else:
        raise ValueError("Invalid value for dataset.")
    mean_tensor, std_tensor = compute_mean_std(
        data=trainset, img_size=img_size, batch_size=args.batch_size
    )
    print(mean_tensor, std_tensor)
