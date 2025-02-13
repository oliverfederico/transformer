import torch
import torch.nn as nn
import time
import datetime
import os
import random
import torchvision.transforms.v2
import utils
import warnings
from vision_transformer import VisionTransformer
import torchvision
from typing import Any, Tuple
from dataload import show_image

# Set random seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class QUADMNIST(torchvision.datasets.MNIST):

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        rstate = random.getstate()
        random.seed(index)
        i_s = [0,0,0,0] 
        for i in range(4):
            i_s[i] = random.randint(0, super().__len__() - 1)
            # i_s[i] = index % super().__len__()
            # index = index // super().__len__()
        random.setstate(rstate)
        tl_img, tl_lab = super().__getitem__(i_s[0])
        tr_img, tr_lab = super().__getitem__(i_s[1])
        bl_img, bl_lab = super().__getitem__(i_s[2])
        br_img, br_lab = super().__getitem__(i_s[3])
        top_row = torch.concat([tl_img, tr_img], dim=2)
        bottom_row = torch.concat([bl_img, br_img], dim=2)
        img = torch.cat([top_row, bottom_row], dim=1)
        # print(img.shape)
        # # Convert tensor to PIL Image for visualization
        # img = torchvision.transforms.functional.to_pil_image(img)
        # show_image(img)

        return img, tl_lab * 1000 + tr_lab * 100 + bl_lab * 10 + br_lab

    # def __len__(self) -> int:
    #     return super().__len__()


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=False):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
        # gather the stats from all processes

        # if (
        #     hasattr(data_loader.dataset, "__len__")
        #     and len(data_loader.dataset) != num_processed_samples
        #     and torch.distributed.get_rank() == 0
        # ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    print(
        f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}"
    )
    return metric_logger.acc1.global_avg


def load_data(train_dir, val_dir, args):
    print("Loading data")
    print("Loading training data")
    st = time.time()

    transforms = torchvision.transforms.v2.Compose(
        [
            torchvision.transforms.v2.PILToTensor(),
            torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
            # torchvision.transforms.v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            torchvision.transforms.v2.ToPureTensor(),
        ]
    )

    target_transforms = torchvision.transforms.v2.Compose(
        [
            torchvision.transforms.v2.ToDtype(torch.int32, scale=False),
            torchvision.transforms.v2.ToPureTensor(),
        ]
    )

    dataset = QUADMNIST(
        train_dir, train=True, transform=transforms, target_transform=torch.tensor
    )
    # dataset = torchvision.datasets.wrap_dataset_for_transforms_v2(dataset)

    print("Took", time.time() - st)

    print("Loading validation data")
    dataset_test = QUADMNIST(
        train_dir,
        train=False,
        transform=transforms,
        target_transform=torch.tensor
        # target_transform=target_transforms
    )

    print("Creating data loaders")
    train_sampler =None# torch.utils.data.RandomSampler(dataset)
    test_sampler = None# torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args):
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.cuda.empty_cache()
    print(f"Using device: {device}")

    train_dir = os.path.join(args.data_path, "")
    val_dir = os.path.join(args.data_path, "")
    dataset, dataset_test, train_sampler, test_sampler = load_data(
        train_dir, val_dir, args
    )

    # data_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=args.batch_size,
    #     sampler=train_sampler,
    #     num_workers=args.workers,
    #     pin_memory=True,
    # )
    # data_loader_test = torch.utils.data.DataLoader(
    #     dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    # )

    num_classes = 10000  # len(dataset.classes)

    print("Creating model")
    model = VisionTransformer(
        image_size=56,
        patch_size=28,
        num_layers=2,  # 12,
        num_heads=2,  # 12,
        hidden_dim=64,  # 768,
        mlp_dim=128,  # 3072,
        num_classes=num_classes,
    ).to(device)

    num_epochs = args.epochs

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0008, betas=(0.9, 0.999), weight_decay=0.1
    )
    # For finetuning, use SGD with momentum
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    # scheduler = torch.optim.lr_scheduler.SequentialLR(
    #     optimizer,

    #     {0: 0.0008, 10: 0.0001},
    #     last_epoch=-1,
    # )

    print("Start training")
    start_time = time.time()
    for epoch in range(num_epochs):
        train_one_epoch(model, criterion, optimizer, dataset, device, epoch)
        lr_scheduler.step()
        evaluate(model, criterion, dataset_test, device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="PyTorch ViT Training", add_help=add_help
    )
    parser.add_argument(
        "--data-path", default="./datasets", type=str, help="dataset path"
    )
    parser.add_argument("--batch-size", default=1, type=int, help="batch size")
    parser.add_argument("--epochs", default=1, type=int, help="number of epochs")
    parser.add_argument("--workers", default=4, type=int, help="number of workers")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
