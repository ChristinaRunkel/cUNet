import os
import torch
import torch.nn as nn
import torchvision
import datetime
import argparse
import ml_collections
from tqdm import tqdm
from data import (
    GLaSDataLoader,
    STAREDataset,
    PolypDataset,
    NucleiDataset,
    BreastCancerDataset,
    MelanomaDataset,
)
from models import (
    ConvSODEUNet,
    ConvResUNet,
    Unet,
    ConvODEUNet,
    UNext,
    DenseUNet,
    VisionTransformer,
)
from transforms import RandomVerticalFlip, RandomHorizontalFlip
from train_utils import save_loss_to_file


parser = argparse.ArgumentParser(description="cUNet experiments")
parser.add_argument(
    "--epochs",
    default=100,
    type=int,
    metavar="N",
    help="number of total epochs to run",
)
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
parser.add_argument(
    "--dataset",
    type=str,
    help="Dataset to use. Please choose one of the following options: STARE, cells",
)
parser.add_argument(
    "--net",
    type=str,
    help="Network architecture. Possible network architectures are USODE, UNODE, UNET, URESNET",
)
parser.add_argument(
    "--gradient-accumulation",
    default=1,
    type=int,
    help="mini-batch size for gradient accumulation",
)
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--seed", default=0, type=int, help="Random seed")
parser.add_argument(
    "--pretrained", default=False, type=bool, help="Load pretrained model"
)
parser.add_argument("--checkpoint", type=str, help="Checkpoint ID for pretrained model")
parser.add_argument("--noise", type=float, nargs="+", help="Amoint of noise to add.")


def load_checkpoint(
    checkpoint: str,
    model: nn.Module,
    device: str,
):
    """
    Loads model weights from a checkpoint file and sets the model to evaluation mode.
    Args:
        checkpoint (str): Path to the checkpoint file.
        model (nn.Module): Model to load weights into.
        device (str): Device to map the model to ('cpu' or 'cuda').
    Returns:
        nn.Module: Model with loaded weights in eval mode.
    """
    model = model
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def main():
    """
    Main function to run ablation studies for different datasets and models.
    Parses arguments, loads datasets, and runs training/evaluation.
    """
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.dataset == "cells":
        data_path = "<PATH_TO_CELLS_DATASET>"
        val_set_idx = torch.LongTensor(10).random_(0, 85)
        train_set_idx = torch.arange(0, 85)
        overlapping = (train_set_idx[..., None] == val_set_idx).any(-1)
        train_set_idx = torch.masked_select(train_set_idx, ~overlapping)

        trainset = GLaSDataLoader(
            data_path=data_path,
            patch_size=(352, 352),
            dataset_repeat=1,
            images=train_set_idx,
        )
        valset = GLaSDataLoader(
            data_path=data_path,
            patch_size=(352, 352),
            dataset_repeat=1,
            images=val_set_idx,
            validation=True,
        )
        img_size = 352
    elif args.dataset == "STARE":
        data_path = "<PATH_TO_STARE_DATASET>"
        inp_path = os.path.join(data_path, "inputs")
        gt_path = os.path.join(data_path, "GT")
        perm_idx = torch.randperm(20)
        val_set_idx = perm_idx[:4]
        train_set_idx = perm_idx[4:]
        mean = [0.5889, 0.3338, 0.0000]
        std = [0.3528, 0.1919, 0.0001]
        transforms = torchvision.transforms.Compose(
            [RandomVerticalFlip(p=0.5), RandomHorizontalFlip(p=0.5)],
        )
        normalise = torchvision.transforms.Compose(
            [torchvision.transforms.Normalize(mean=mean, std=std)]
        )
        trainset = STAREDataset(
            inp_path=inp_path,
            gt_path=gt_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=True,
            transforms=transforms,
            normalise=None,
        )
        valset = STAREDataset(
            inp_path=inp_path,
            gt_path=gt_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=False,
            transforms=None,
            normalise=None,
        )
        img_size = 512
    elif args.dataset == "polyp":
        data_path = "<PATH_TO_POLYP_DATASET>"
        inp_path = os.path.join(data_path, "images")
        gt_path = os.path.join(data_path, "masks")
        perm_idx = torch.randperm(1000)
        val_set_idx = perm_idx[:200]
        train_set_idx = perm_idx[200:]
        mean = [0.4726, 0.2711, 0.0000]
        std = [0.2796, 0.1945, 0.0001]
        transforms = torchvision.transforms.Compose(
            [RandomVerticalFlip(p=0.5), RandomHorizontalFlip(p=0.5)],
        )
        normalise = torchvision.transforms.Compose(
            [torchvision.transforms.Normalize(mean=mean, std=std)]
        )
        trainset = PolypDataset(
            inp_path=inp_path,
            gt_path=gt_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=True,
            transforms=transforms,
            normalise=None,
        )
        valset = PolypDataset(
            inp_path=inp_path,
            gt_path=gt_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=False,
            transforms=None,
            normalise=None,
        )
        img_size = 256
    elif args.dataset == "nuclei":
        data_path = "<PATH_TO_NUCLEI_DATASET>"
        perm_idx = torch.randperm(670)
        val_set_idx = perm_idx[:134]
        train_set_idx = perm_idx[134:]
        mean = [0.1890, 0.1551, 0.1707]
        std = [0.2960, 0.2433, 0.2637]
        transforms = torchvision.transforms.Compose(
            [RandomVerticalFlip(p=0.5), RandomHorizontalFlip(p=0.5)],
        )
        normalise = torchvision.transforms.Compose(
            [torchvision.transforms.Normalize(mean=mean, std=std)]
        )
        trainset = NucleiDataset(
            inp_path=data_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=True,
            transforms=transforms,
            normalise=None,
        )
        valset = NucleiDataset(
            inp_path=data_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=False,
            transforms=None,
            normalise=None,
        )
        img_size = 256

    elif args.dataset == "breast_cancer":
        data_path = "<PATH_TO_BREAST_CANCER_DATASET>"
        perm_idx = torch.randperm(647)
        val_set_idx = perm_idx[:129]
        train_set_idx = perm_idx[129:]
        mean = [0.2524, 0.2525, 0.2525]
        std = [0.2019, 0.2019, 0.2019]
        transforms = torchvision.transforms.Compose(
            [RandomVerticalFlip(p=0.5), RandomHorizontalFlip(p=0.5)],
        )
        normalise = torchvision.transforms.Compose(
            [torchvision.transforms.Normalize(mean=mean, std=std)]
        )
        trainset = BreastCancerDataset(
            inp_path=data_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=True,
            transforms=transforms,
            normalise=None,
        )
        valset = BreastCancerDataset(
            inp_path=data_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=False,
            transforms=None,
            normalise=None,
        )
        img_size = 256
    elif args.dataset == "melanoma_segmentation":
        inp_path = "<PATH_TO_MELANOMA_TRAIN_DATA>"
        gt_path = "<PATH_TO_MELANOMA_TRAIN_GROUNDTRUTH>"
        perm_idx = torch.randperm(900)
        val_set_idx = perm_idx[:180]
        train_set_idx = perm_idx[180:]
        mean = [0.5670, 0.6184, 0.7236]
        std = [0.1930, 0.1723, 0.1650]
        transforms = torchvision.transforms.Compose(
            [RandomVerticalFlip(p=0.5), RandomHorizontalFlip(p=0.5)],
        )
        normalise = torchvision.transforms.Compose(
            [torchvision.transforms.Normalize(mean=mean, std=std)]
        )
        trainset = MelanomaDataset(
            inp_path=inp_path,
            gt_path=gt_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=True,
            transforms=transforms,
            normalise=None,
        )
        valset = MelanomaDataset(
            inp_path=inp_path,
            gt_path=gt_path,
            train_set_idx=train_set_idx,
            val_set_idx=val_set_idx,
            train=False,
            transforms=None,
            normalise=None,
        )
        img_size = 512
    else:
        raise ValueError("Invalid value for dataset.")

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=1, shuffle=False, num_workers=4
    )

    # Model
    device = torch.device("cuda")

    if args.net == "USODE":
        net = ConvSODEUNet(
            num_filters=3,
            output_dim=2,
            time_dependent=False,
            non_linearity="lrelu",
            adjoint=True,
            tol=1e-3,
        )
    elif args.net == "UNODE":
        net = ConvODEUNet(
            num_filters=3,
            output_dim=1,
            time_dependent=True,
            non_linearity="lrelu",
            adjoint=True,
            tol=1e-3,
        )
    elif args.net == "UNET":
        net = Unet(depth=4, num_filters=3, output_dim=2)

    elif args.net == "URESNET":
        net = ConvResUNet(num_filters=3, output_dim=1, non_linearity="lrelu")
    elif args.net == "UNEXT":
        net = UNext(num_classes=1, input_channels=3, img_size=img_size)
    elif args.net == "DENSEUNET":
        net = DenseUNet(n_classes=1)
    elif args.net == "TRANSUNET":
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
        config.patch_size = 16  # adapt (value for img_size=512: 32)

        config.decoder_channels = (256, 128, 64, 32)
        config.n_classes = 1
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
            num_classes=2,
            sode_decoder=args.sode_decoder,
        )
    else:
        raise ValueError(
            "Please pick one of the following network architectures: USODE, UNODE, UNET, URESNET, UNEXT, DENSEUNET"
        )
    if args.pretrained:
        if args.net == "UNET":
            net = torch.hub.load(
                "mateuszbuda/brain-segmentation-pytorch",
                "unet",
                in_channels=3,
                out_channels=1,
                init_features=32,
                pretrained=True,
            )
            net.conv = nn.Conv2d(32, 2, kernel_size=(1, 1), stride=(1, 1))
        else:
            net = load_checkpoint(checkpoint=args.checkpoint, model=net, device=device)
        print("Using pretrained model.")
    net.to(device)

    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=1, gamma=0.999
    )
    torch.backends.cudnn.benchmark = True
    for n in args.noise:
        date_str = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")

        filename = os.path.join(
            "<PATH_TO_OUTPUTS>",
            args.dataset,
            "checkpoints",
            "Noise" + str(n) + args.net + date_str + ".pt",
        )
        filename_best = os.path.join(
            "<PATH_TO_OUTPUTS>",
            args.dataset,
            "checkpoints",
            "Noise" + str(n) + args.net + date_str + "_best.pt",
        )
        logfile = os.path.join(
            "<PATH_TO_OUTPUTS>",
            args.dataset,
            "log_files",
            "Noise" + str(n) + args.net + date_str + ".txt",
        )
        run_with_noise(
            args=args,
            logfile=logfile,
            optimizer=optimizer,
            trainloader=trainloader,
            valloader=valloader,
            net=net,
            criterion=criterion,
            scheduler=scheduler,
            filename=filename,
            filename_best=filename_best,
            device=device,
            noise=n,
        )


def run_with_noise(
    args,
    logfile,
    optimizer,
    trainloader,
    valloader,
    net,
    criterion,
    scheduler,
    filename,
    filename_best,
    device,
    noise,
):
    """
    Runs training and validation with added Gaussian noise to the inputs.
    Args:
        args: Arguments/configuration for training.
        logfile (str): Path to log file.
        optimizer: Optimizer for training.
        trainloader: DataLoader for training data.
        valloader: DataLoader for validation data.
        net: Model to train.
        criterion: Loss function.
        scheduler: Learning rate scheduler.
        filename (str): Path to save model.
        filename_best (str): Path to save best model.
        device (str): Device to run training on.
        noise (float): Standard deviation of Gaussian noise to add.
    Returns:
        None
    """
    losses = []
    val_losses = []
    accumulated = 0
    best_val_loss = 10000

    fw = open(logfile, "a+")
    line = f"Noise: {noise} - Learning rate: {args.lr} - Batch size: {args.batch_size} - Accumulate Gradient: {args.gradient_accumulation} - Epochs: {args.epochs} - Pretrained: {args.pretrained}"
    fw.write(line)
    fw.close()
    print(line)

    accumulated = 0
    for param_group in optimizer.param_groups:
        param_group["lr"] = args.lr

    for _ in range(args.epochs):
        # training loop with gradient accumulation
        running_loss = 0.0
        optimizer.zero_grad()

        for data in tqdm(trainloader):
            inputs, labels = data[0].cuda(), data[1].cuda()
            inputs = inputs + torch.normal(
                mean=0, std=noise, size=inputs.shape, device=device
            )
            outputs = net(inputs)
            loss = criterion(outputs, labels) / args.gradient_accumulation
            loss.backward()
            accumulated += 1
            if accumulated == args.gradient_accumulation:
                optimizer.step()
                optimizer.zero_grad()
                accumulated = 0
                if scheduler is not None:
                    scheduler.step()

            running_loss += loss.item() * args.gradient_accumulation

        batched_loss = running_loss / len(trainloader)
        print("\n Training loss: " + str(batched_loss))
        save_loss_to_file(logfile=logfile, eval_type="Training", loss=batched_loss)
        losses.append(batched_loss)

        # validation loop
        with torch.no_grad():
            running_loss = 0.0
            for data in valloader:
                inputs, labels = data[0].cuda(), data[1].cuda()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

            batched_val_loss = running_loss / len(valloader)
            print(" Validation loss: " + str(batched_val_loss))
            save_loss_to_file(
                logfile=logfile, eval_type="Validation", loss=batched_val_loss
            )
            val_losses.append(batched_val_loss)
            torch.save(net.state_dict(), filename)
            if batched_val_loss < best_val_loss:
                best_val_loss = batched_val_loss
                torch.save(net.state_dict(), filename_best)


if __name__ == "__main__":
    main()
