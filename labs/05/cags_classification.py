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
    augmentation_fn = None 
    if args.augment:
        augmentation_fn = v2.Compose([
            v2.RandomResizedCrop(
                (224, 224),
                scale=(0.8, 1.0),
            ),
            v2.RandomHorizontalFlip(),
            v2.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
        ])

    # Load the EfficientNetV2-B0 model without the classification layer. For an
    # input image, the model returns a tensor of shape `[batch_size, 1280]`.
    efficientnetv2_b1 = timm.create_model("tf_efficientnetv2_b1.in1k", pretrained=True, num_classes=0)

    # Create a simple preprocessing performing necessary normalization.
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Normalize(mean=efficientnetv2_b1.pretrained_cfg["mean"], std=efficientnetv2_b1.pretrained_cfg["std"]),
    ])

    train = TransformedDataset(cags.train, normalize_fn=preprocessing, augmentation_fn=augmentation_fn)
    dev = TransformedDataset(cags.dev, normalize_fn=preprocessing)
    test = TransformedDataset(cags.test, normalize_fn=preprocessing)

    train = train.dataloader(batch_size=args.batch_size, num_workers=args.dataloader_workers, shuffle=True, seed=args.seed)
    dev = dev.dataloader(batch_size=args.batch_size, num_workers=args.dataloader_workers)
    test = test.dataloader(batch_size=args.batch_size)

    # TODO: Create the model and train it.
    model = Model(args, efficientnetv2_b1)

    # Freeze pretrained parameters
    for param in model._pretrained_model.parameters():
        param.requires_grad = False
    
    dummy = torch.zeros(1, 3, 224, 224)
    model(dummy)
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

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
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=CAGS.LABELS)},
        logdir=logdir,
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(train) * args.epochs,
            eta_min=1e-5
        )
    )
    # train classifier for a bit
    classifier_epochs = min(5, args.epochs)
    model.fit(train, dev=dev, epochs=classifier_epochs)

    # train classifier + last few blocks of pretrained model 
    blocks = model._pretrained_model.blocks

    fine_tune_epochs = max(0, args.epochs - classifier_epochs)
    # freeze everything
    for p in model._pretrained_model.parameters():
        p.requires_grad = False

    # unfreeze last two blocks
    for block in blocks[-2:]:
        for p in block.parameters():
            p.requires_grad = True

    # classifier is trainable
    for p in model.classifier.parameters():
        p.requires_grad = True

    new_optimizer = torch.optim.AdamW([
        {
            "params": model.classifier.parameters(),
            "lr": 3e-4,
        },
        {
            "params": blocks[-2:].parameters(),
            "lr": 1e-5,
        },
    ])
    model.configure(
        optimizer=new_optimizer,
        loss=torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing),
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=CAGS.LABELS)},
        logdir=logdir,
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
            new_optimizer,
            T_max=len(train) * fine_tune_epochs,
            eta_min=1e-5
        )
    )
    
    logs = model.fit(train, dev=dev, epochs=fine_tune_epochs, log_graph=True)
    torch.save(model.state_dict(), os.path.join(logdir, f"model.pt"))

    # Generate test set annotations, but in `logdir` to allow parallel execution.
    
    with open(os.path.join(logdir, "cags_classification.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(image, target)` pairs.
        for prediction in model.predict(test, data_with_labels=True):
            print(prediction.argmax().item(), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
