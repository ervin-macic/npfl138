#!/usr/bin/env python3
import argparse
import os

import numpy as np
import torch
import torchmetrics
import torchvision
from torchvision.transforms import v2 

import npfl138
npfl138.require_version("2526.4")
from npfl138.datasets.cifar10 import CIFAR10

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--augment", default=False, action="store_true", help="Whether to augment the data.")
parser.add_argument("--dataloader_workers", default=0, type=int, help="Number of dataloader workers.")
parser.add_argument("--label_smoothing", default=0, type=float, help="Label smoothing.")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay strength.")
parser.add_argument("--patience", default=5, type=int, help="Early stopping patience.")


class TransformedDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: CIFAR10.Dataset, augmentation_fn=None) -> None:
        super().__init__(dataset)
        self._augmentation_fn = augmentation_fn
        self.normalize = v2.Normalize(
            mean=(0.4914,0.4822,0.4465),
            std=(0.2470,0.2435,0.2616)
        )

    def transform(self, example: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        image = example["image"].to(torch.float32) / 255
        if self._augmentation_fn is not None:
            image = self._augmentation_fn(image)
        image = self.normalize(image)
        label = example["label"]
        return image, label

    def transform_batch(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        images = images.to(memory_format=torch.channels_last)  # convert images to channels-last memory format
        return images, labels
    
class ResidualBlock(torch.nn.Module):
    def __init__(self, block, shortcut=None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut or torch.nn.Identity()

    def forward(self, x):
        # post activation style
        return self.shortcut(x) + self.block(x)

class EarlyStopping:
    def __init__(self, patience: int = 4, metric: str = "dev:accuracy"):
        self.patience = patience
        self.metric = metric
        self.best_value = -float("inf")
        self.best_state = None
        self.epochs_without_improvement = 0

    def __call__(self, module, epoch, logs):
        value = logs[self.metric]
        if value > self.best_value:
            self.best_value = value
            self.best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                print(f"  Early stopping (best {self.metric}={self.best_value:.4f})")
                return getattr(module, "STOP_TRAINING", "stop_training")

def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))
    os.makedirs(logdir, exist_ok=True)

    # Load the data.
    cifar = CIFAR10()

    # Data augmentations
    augmentation_fn = None
    if args.augment:
        augmentation_fn = v2.Compose([
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(10),
        ])

    # Dataloaders
    train = TransformedDataset(cifar.train, augmentation_fn=augmentation_fn)
    dev = TransformedDataset(cifar.dev)
    train = train.dataloader(batch_size=args.batch_size, num_workers=args.dataloader_workers, shuffle=True, seed=args.seed)
    dev = dev.dataloader(batch_size=args.batch_size, num_workers=args.dataloader_workers)

    # Create model
    model = npfl138.TrainableModule(torch.nn.Sequential(
        torch.nn.LazyConv2d(64, kernel_size=3, padding=1, bias=False),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),

        ResidualBlock(
            torch.nn.Sequential(
                torch.nn.LazyConv2d(64, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.LazyConv2d(64, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(64),
            )
        ),
        ResidualBlock(
            torch.nn.Sequential(
                torch.nn.LazyConv2d(64, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.LazyConv2d(64, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(64),
            )
        ),

        ResidualBlock(
            torch.nn.Sequential(
                torch.nn.LazyConv2d(128, 3, stride=2, padding=1, bias=False),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),
                torch.nn.LazyConv2d(128, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(128),
            ),
            shortcut=torch.nn.Sequential(
                torch.nn.LazyConv2d(128, 1, stride=2, bias=False),
                torch.nn.BatchNorm2d(128),
            ),
        ),
        ResidualBlock(
            torch.nn.Sequential(
                torch.nn.LazyConv2d(128, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),
                torch.nn.LazyConv2d(128, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(128),
            )
        ),

        ResidualBlock(
            torch.nn.Sequential(
                torch.nn.LazyConv2d(256, 3, stride=2, padding=1, bias=False),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(),
                torch.nn.LazyConv2d(256, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(256),
            ),
            shortcut=torch.nn.Sequential(
                torch.nn.LazyConv2d(256, 1, stride=2, bias=False),
                torch.nn.BatchNorm2d(256),
            ),
        ),
        ResidualBlock(
            torch.nn.Sequential(
                torch.nn.LazyConv2d(256, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(),
                torch.nn.LazyConv2d(256, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(256),
            )
        ),

        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Dropout(0.1),
        torch.nn.LazyLinear(CIFAR10.LABELS),
    ))
    # Model(args)

    # Configure model
    dummy = torch.zeros(1,3,32,32)
    model(dummy)
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if (param.ndim == 1 or "bias" in name or "bn" in name.lower()):
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.AdamW(
        params=[
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0}
        ],
        lr=1e-3,
    )
    model.configure(
        optimizer=optimizer,
        loss=torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing),
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=CIFAR10.LABELS)},
        logdir=logdir,
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(train) * args.epochs,
            eta_min=1e-5
        )
    )


    # Train model
    early_stop = EarlyStopping(patience=args.patience)
    logs = model.fit(train, dev=dev, epochs=args.epochs, log_graph=True, callbacks=[early_stop])
    if early_stop.best_state is not None:
        model.load_state_dict(early_stop.best_state)
        model.eval()
    print(f"Done (best dev_accuracy={early_stop.best_value:.4f})")
    torch.save(model.state_dict(), os.path.join(logdir, f"model.pt"))

    # Test model
    test = TransformedDataset(cifar.test)
    test = test.dataloader(batch_size=args.batch_size)

    # Generate test set annotations, but in `logdir` to allow parallel execution.
    with open(os.path.join(logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(image, target)` pairs.
        for prediction in model.predict(test, data_with_labels=True):
            print(prediction.argmax().item(), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
