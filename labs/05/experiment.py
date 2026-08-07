#!/usr/bin/env python3
import argparse
import os

import timm
import torch
import torchvision.transforms.v2 as v2
import torchmetrics 

import npfl138
npfl138.require_version("2526.5.2")
from npfl138.datasets.cags import CAGS

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--augment", default=False, action="store_true", help="Whether to augment the data.")
parser.add_argument("--dataloader_workers", default=0, type=int, help="Number of dataloader workers.")
parser.add_argument("--label_smoothing", default=0, type=float, help="Label smoothing.")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay strength.")
parser.add_argument("--patience", default=5, type=int, help="Early stopping patience.")

class TransformedDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: CAGS.Dataset, normalize_fn = None, augmentation_fn=None) -> None:
        super().__init__(dataset)
        self._augmentation_fn = augmentation_fn
        self.normalize = normalize_fn

    def transform(self, example: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        image = example["image"]
        if self._augmentation_fn is not None:
            image = self._augmentation_fn(image)
        image = self.normalize(image)
        label = example["label"]
        return image, label

    def transform_batch(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        images = images.to(memory_format=torch.channels_last)  # convert images to channels-last memory format
        return images, labels

class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, pretrained_model) -> None:
        super().__init__()
        self._args = args
        self._pretrained_model = pretrained_model
        self.classifier = torch.nn.LazyLinear(CAGS.LABELS)

    def forward(self, x):
        features = self._pretrained_model(x)
        return self.classifier(features)

def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))
    os.makedirs(logdir, exist_ok=True)

    # Load the data. The individual examples are dictionaries with the keys:
    # - "image", a `[3, 224, 224]` tensor of `torch.uint8` values in [0-255] range,
    # - "mask", a `[1, 224, 224]` tensor of `torch.float32` values in [0-1] range,
    # - "label", a scalar of the correct class in `range(CAGS.LABELS)`.
    # The `decode_on_demand` argument can be set to `True` to save memory and decode
    # each image only when accessed, but it will most likely slow down training.

    cags = CAGS(decode_on_demand=False)
    # Load the EfficientNetV2-B0 model without the classification layer. For an
    # input image, the model returns a tensor of shape `[batch_size, 1280]`.
    efficientnetv2_b0 = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)

    # TODO: Create the model and train it.
    model = Model(args, efficientnetv2_b0)
    for name, module in model._pretrained_model.named_children():
        print(name)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
