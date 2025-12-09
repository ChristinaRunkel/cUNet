import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torchmetrics
import torchvision
import ml_collections
from typing import Tuple
from metrics import averaged_hausdorff_distance, pixelwise_acc

from models import (
    ConvSODEUNet,
    ConvODEUNet,
    ConvResUNet,
    Unet,
    UNext,
    DenseUNet,
    VisionTransformer,
)
from data import (
    STAREDataset,
    GLaSDataLoader,
    PolypDataset,
    BreastCancerDataset,
    MelanomaDataset,
    NucleiDataset,
)

parser = argparse.ArgumentParser(description="USODE compute metrics")

parser.add_argument(
    "--dataset",
    type=str,
    help="Dataset to use. Please choose one of the following options: STARE, cells",
)
parser.add_argument(
    "--net",
    nargs="+",
    type=str,
    help="Network architecture. Possible network architectures are USODE, UNODE, UNET, URESNET",
)
parser.add_argument(
    "--identifier",
    nargs="+",
    type=str,
    help="Training identifier",
)
parser.add_argument(
    "--path",
    type=str,
    help="Path",
)
parser.add_argument(
    "--pretrained",
    type=bool,
    help="Use pretrained UNet model",
)
parser.add_argument("--noise", type=float, nargs="+")
parser.add_argument(
    "--block",
    nargs="+",
    type=str,
    default="PLN",
    help="Block types: PLN, RSE, DSE, INC, PSP",
)


def load_dataset(dataset: str) -> Tuple[torch.utils.data.Dataset, float]:
    """
    Loads the specified dataset and returns the test set and image size.

    Args:
        dataset (str): Name of the dataset to load.

    Returns:
        Tuple[torch.utils.data.Dataset, float]: Test set and image size.
    """
    if dataset == "cells":
        data_path = "<PATH_TO_CELLS_DATASET>"
        test_set_idx = np.arange(0, 60)
        testset = GLaSDataLoader(
            data_path=data_path,
            patch_size=(352, 352),
            dataset_repeat=1,
            images=test_set_idx,
            validation=True,
            test=True,
        )
        img_size = 352
    elif dataset == "STARE":
        data_path = "<PATH_TO_STARE_DATASET>"
        inp_path = os.path.join(data_path, "inputs")
        gt_path = os.path.join(data_path, "GT")
        perm_idx = torch.randperm(20)
        val_set_idx = perm_idx[:4]
        train_set_idx = perm_idx[4:]
        testset = STAREDataset(
            inp_path=inp_path,
            gt_path=gt_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=False,
            transforms=None,
            normalise=None,
        )
        img_size = 512
    elif dataset == "polyp":
        data_path = "<PATH_TO_POLYP_DATASET>"
        inp_path = os.path.join(data_path, "images")
        gt_path = os.path.join(data_path, "masks")
        perm_idx = torch.randperm(1000)
        val_set_idx = perm_idx[:200]
        train_set_idx = perm_idx[200:]
        testset = PolypDataset(
            inp_path=inp_path,
            gt_path=gt_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=False,
            transforms=None,
            normalise=None,
        )
        img_size = 256
    elif dataset == "nuclei":
        data_path = "<PATH_TO_NUCLEI_DATASET>"
        perm_idx = torch.randperm(670)
        val_set_idx = perm_idx[:134]
        train_set_idx = perm_idx[134:]
        testset = NucleiDataset(
            inp_path=data_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=False,
            transforms=None,
            normalise=None,
        )
        img_size = 256

    elif dataset == "breast_cancer":
        data_path = "<PATH_TO_BREAST_CANCER_DATASET>"
        perm_idx = torch.randperm(647)
        val_set_idx = perm_idx[:129]
        train_set_idx = perm_idx[129:]
        testset = BreastCancerDataset(
            inp_path=data_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=False,
            transforms=None,
            normalise=None,
        )
        img_size = 256
    elif dataset == "melanoma_segmentation":
        inp_path = "<PATH_TO_MELANOMA_TEST_DATA>"
        gt_path = "<PATH_TO_MELANOMA_TEST_GROUNDTRUTH>"
        perm_idx = torch.randperm(900)
        val_set_idx = perm_idx[:180]
        train_set_idx = perm_idx[180:]
        testset = MelanomaDataset(
            inp_path=inp_path,
            gt_path=gt_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=False,
            transforms=None,
            normalise=None,
            test=True,
        )
        img_size = 512
    else:
        raise ValueError("Invalid value for dataset.")

    return testset, img_size


def compute_metrics(
    net_name: str,
    dataset: str,
    valset: torchvision.datasets,
    img_size: float,
    identifier: str,
    path: str,
    noise: float,
    block: str,
    pretrained: bool = False,
) -> Tuple[float, float, float]:
    """
    Computes segmentation metrics for a given model and dataset with added noise.

    Args:
        net_name (str): Name of the network/model.
        dataset (str): Name of the dataset.
        valset: Validation set.
        img_size (float): Image size.
        identifier (str): Model identifier.
        path (str): Path to checkpoints.
        noise (float): Noise level to add.
        block (str): Block type.
        pretrained (bool): Whether to use pretrained weights.

    Returns:
        Tuple[float, float, float]: Computed metrics (e.g., Dice, Hausdorff, etc.).
    """
    torch.manual_seed(0)
    checkpoint = os.path.join(path, dataset, "checkpoints", identifier + "_best.pt")
    output_dim = 2 if args.dataset == "cells" else 1
    if net_name == "USODE":
        net = ConvSODEUNet(
            num_filters=3,
            output_dim=output_dim,
            time_dependent=False,
            non_linearity="lrelu",
            adjoint=True,
            tol=1e-6,
        )
    elif net_name == "UNODE":
        net = ConvODEUNet(
            num_filters=3,
            output_dim=output_dim,
            time_dependent=True,
            non_linearity="lrelu",
            adjoint=True,
            tol=1e-3,
        )
    elif net_name == "UNET":
        net = Unet(depth=4, num_filters=3, output_dim=output_dim, block=block)

    elif net_name == "URESNET":
        net = ConvResUNet(num_filters=3, output_dim=output_dim, non_linearity="lrelu")
    elif net_name == "UNEXT":
        net = UNext(num_classes=output_dim, input_channels=3, img_size=img_size)
    elif net_name == "DENSEUNET":
        net = DenseUNet(n_classes=output_dim)
    elif net_name == "TRANSUNET":
        config = ml_collections.ConfigDict()
        config.patches = ml_collections.ConfigDict({"size": (32, 32)})
        config.hidden_size = 384  # adapt (value for img_size=512: 768)
        config.transformer = ml_collections.ConfigDict()
        config.transformer.mlp_dim = 1536  # adapt (value for img_size=512: 3072)
        config.transformer.num_heads = 12
        config.transformer.num_layers = 12
        config.transformer.attention_dropout_rate = 0.0
        config.transformer.dropout_rate = 0.1

        config.classifier = "seg"
        config.representation_size = None
        config.resnet_pretrained_path = None
        config.pretrained_path = (
            None  #'../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
        )
        config.patch_size = 16  # 16 adapt (value for img_size=512: 32)

        config.decoder_channels = (256, 128, 64, 32)
        config.n_classes = output_dim
        config.activation = "softmax"
        config.patches.grid = (16, 16)  # adapt (values for img_size=512: (32, 32))
        config.resnet = ml_collections.ConfigDict()
        config.resnet.num_layers = (3, 4, 9)
        config.resnet.width_factor = 1

        config.skip_channels = [512, 256, 64, 32]
        config.n_skip = 3
        net = VisionTransformer(
            config=config,
            img_size=img_size,
            num_classes=output_dim,
            sode_decoder=False,
        )
    elif net_name == "REGUNET":
        ########## New Model from Lihao
        from monai.networks.nets import RegUNet

        net = RegUNet(
            spatial_dims=2,
            num_channel_initial=16,
            depth=3,
            in_channels=3,
            out_channels=output_dim,
        )
    elif net_name == "ATTENTIONUNET":
        ########## New Model from Lihao
        from other_models import AttentionUnet

        net = AttentionUnet(
            spatial_dims=2,
            channels=[16, 32, 64],
            strides=[2, 2, 2],
            in_channels=3,
            out_channels=output_dim,
        )
    elif net_name == "DYNUNET":
        ########## New Model from Lihao
        from monai.networks.nets import DynUNet

        net = DynUNet(
            spatial_dims=2,
            kernel_size=[3, 3, 3],
            strides=[(1, 1), 2, 2],
            upsample_kernel_size=[2, 2, 2],
            in_channels=3,
            out_channels=output_dim,
        )
    elif net_name == "UNETPLUSPLUS":
        ########## New Model from Lihao
        from other_models import BasicUNetPlusPlus

        net = BasicUNetPlusPlus(spatial_dims=2, in_channels=3, out_channels=output_dim)
    elif net_name == "U2NET":
        ########## New Model from Lihao
        from other_models_old import U2NET

        net = U2NET(in_ch=3, out_ch=output_dim)
    elif net_name == "DEEPLABV3PLUS":
        ########## New Model from Lihao
        import segmentation_models_pytorch as smp

        net = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=None,
            classes=output_dim,
            activation=None,
        )
    else:
        raise ValueError(
            "Please pick one of the following network architectures: USODE, UNODE, UNET, URESNET, UNEXT, DENSEUNET"
        )

    model = load_checkpoint(
        checkpoint=checkpoint,
        model=net,
        device=torch.device("cpu"),
        new_checkpoint=True,
    )

    testloader = torch.utils.data.DataLoader(
        valset, batch_size=1, shuffle=False, num_workers=1
    )
    dice = 0.0
    avg_hausdorff = 0.0
    acc = 0.0
    counter = 0
    dice_op = torchmetrics.Dice(average="macro", num_classes=2)
    f1_op = torchmetrics.classification.BinaryF1Score()
    for img, lbl in testloader:
        counter += 1
        img_noisy = img + torch.normal(mean=0, std=noise, size=img.shape)
        out = model(img_noisy)
        out_sig = torch.sigmoid(out)
        out_thres = (out_sig.detach() > 0.5).to(int)
        lbl_detach = lbl.detach().to(int)
        dice += dice_op(out_thres, lbl_detach)
        avg_hausdorff += averaged_hausdorff_distance(
            set1=out_thres[:, 0].squeeze(), set2=lbl_detach[:, 0].squeeze()
        )
        acc += pixelwise_acc(pred=out_thres, lbl=lbl_detach)
    dice /= counter
    avg_hausdorff /= counter
    acc /= counter

    print(
        f"Dice: {dice:.4f} - Avg Hausdorff: {avg_hausdorff:.2f} - Accuracy: {acc:.4f}"
    )

    return dice, avg_hausdorff, acc


def load_checkpoint(
    checkpoint: str, model: nn.Module, device: str, new_checkpoint: bool = True
):
    """
    Loads model weights from a checkpoint file and sets the model to evaluation mode.

    Args:
        checkpoint (str): Path to the checkpoint file.
        model (nn.Module): Model to load weights into.
        device (str): Device to map the model to ('cpu' or 'cuda').
        new_checkpoint (bool): Whether to load state dict or not.

    Returns:
        nn.Module: Model with loaded weights in eval mode.
    """
    model_n = model
    if new_checkpoint:
        model_n.load_state_dict(torch.load(checkpoint, map_location=device))
    else:
        torch.load(checkpoint, map_location=device)
    model_n.eval()
    return model_n


def write_to_log_file(
    log_file, metric_name, metric_value, net_name, block, noise_level
):
    """
    Writes a metric value to a log file with a formatted line including noise level and block type.

    Args:
        log_file (str): Path to log file.
        metric_name (str): Name of the metric.
        metric_value (float): Value of the metric.
        net_name (str): Network/model name.
        block (str): Block type.
        noise_level (float): Noise level used in experiment.
    """
    fw = open(log_file, "a+")
    line = f"\n{noise_level} - {net_name} - {block} - {metric_name}: {metric_value:.4f}"
    fw.write(line)
    fw.close()


if __name__ == "__main__":
    args = parser.parse_args()
    valset, img_size = load_dataset(dataset=args.dataset)
    log_file = os.path.join(args.path, args.dataset, "log_files", "METRICS_BLOCKS.txt")
    for noise in args.noise:
        for identifier, net_name, b in zip(args.identifier, args.net, args.block):
            print(identifier, net_name, b, str(noise))
            dice, avg_hausdorff, acc = compute_metrics(
                net_name=net_name,
                dataset=args.dataset,
                valset=valset,
                img_size=img_size,
                identifier=identifier,
                path=args.path,
                noise=noise,
                block=b,
                pretrained=args.pretrained,
            )
            if net_name == "UNET":
                write_to_log_file(
                    log_file=log_file,
                    metric_name="Dice",
                    metric_value=dice,
                    net_name=net_name,
                    block=b,
                    noise_level=noise,
                )
                write_to_log_file(
                    log_file=log_file,
                    metric_name="Accuracy",
                    metric_value=acc,
                    net_name=net_name,
                    block=b,
                    noise_level=noise,
                )
                write_to_log_file(
                    log_file=log_file,
                    metric_name="Avg Hausdorff",
                    metric_value=avg_hausdorff,
                    net_name=net_name,
                    block=b,
                    noise_level=noise,
                )
            else:
                write_to_log_file(
                    log_file=log_file,
                    metric_name="Dice",
                    metric_value=dice,
                    net_name=net_name,
                    block=None,
                    noise_level=noise,
                )
                write_to_log_file(
                    log_file=log_file,
                    metric_name="Accuracy",
                    metric_value=acc,
                    net_name=net_name,
                    block=None,
                    noise_level=noise,
                )
                write_to_log_file(
                    log_file=log_file,
                    metric_name="Avg Hausdorff",
                    metric_value=avg_hausdorff,
                    net_name=net_name,
                    block=None,
                    noise_level=noise,
                )
