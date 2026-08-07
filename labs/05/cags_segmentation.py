#!/usr/bin/env python3
import argparse
import os

import numpy as np
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
class SigmoidBinaryJaccardIndex(torchmetrics.classification.BinaryJaccardIndex):
    def update(self, preds, target):
        preds = torch.sigmoid(preds)
        target = (target >= 0.5).long()
        super().update(preds, target)
        
class TransformedDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: CAGS.Dataset, normalize_fn = None, augmentation_fn=None, mask_augmentation_fn=None) -> None:
        super().__init__(dataset)
        self._augmentation_fn = augmentation_fn
        self._mask_augmentation_fn = mask_augmentation_fn
        self._normalize_fn = normalize_fn

    def transform(self, example: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        image = example["image"]
        mask = example["mask"]

        if self._augmentation_fn is not None:
            image, mask = self._augmentation_fn(image, mask)

        if self._normalize_fn is not None:
            image = self._normalize_fn(image)

        return image, mask

    def transform_batch(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        images = images.to(memory_format=torch.channels_last)  # convert images to channels-last memory format
        return images, labels

class DecoderBlock(npfl138.TrainableModule):
    def __init__(self, out_channels):
        super().__init__()
        self.upsample = torch.nn.LazyConvTranspose2d(
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv = torch.nn.Sequential(
            torch.nn.LazyConv2d(
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.LazyConv2d(
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x
    
class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, encoder) -> None:
        super().__init__()
        self._args = args
        self.encoder = encoder
        self.decoder1 = DecoderBlock(512)
        self.decoder2 = DecoderBlock(256)
        self.decoder3 = DecoderBlock(128)
        self.decoder4 = DecoderBlock(64)
        self.final_upsample = torch.nn.LazyConvTranspose2d(32, 2, 2)
        self.head = torch.nn.LazyConv2d(1, 1)
        self.decoder = torch.nn.ModuleList([
            self.decoder1,
            self.decoder2,
            self.decoder3,
            self.decoder4,
            self.final_upsample,
            self.head,
        ])

    def forward(self, x):
        output, features = self.encoder.forward_intermediates(x)
        x = self.decoder1(output, features[3])
        x = self.decoder2(x, features[2])
        x = self.decoder3(x, features[1])
        x = self.decoder4(x, features[0])
        x = self.final_upsample(x)
        x = self.head(x)
        return x
    
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

    # Load the EfficientNetV2-B0 model without the classification layer.
    # Apart from calling the model as in the classification task, you can call it using
    #   output, features = efficientnetv2_b0.forward_intermediates(batch_of_images)
    # obtaining (assuming the input images have 224x224 resolution):
    # - `output` is a `[N, 1280, 7, 7]` tensor with the final features before global average pooling,
    # - `features` is a list of intermediate features with resolution 112x112, 56x56, 28x28, 14x14, 7x7.
    encoder = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)

    # Create a simple preprocessing performing necessary normalization.
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Normalize(mean=encoder.pretrained_cfg["mean"], std=encoder.pretrained_cfg["std"]),
    ])

    augmentation_fn = None 
    mask_augmentation_fn = None
    if args.augment:
        augmentation_fn = v2.Compose([
            v2.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
            v2.RandomHorizontalFlip(),
        ])
        mask_augmentation_fn = v2.Compose([
            v2.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
            v2.RandomHorizontalFlip(),
        ])
    
    train = TransformedDataset(cags.train, normalize_fn=preprocessing, augmentation_fn=augmentation_fn, mask_augmentation_fn=mask_augmentation_fn)
    dev = TransformedDataset(cags.dev, normalize_fn=preprocessing)
    test = TransformedDataset(cags.test, normalize_fn=preprocessing)

    train = train.dataloader(batch_size=args.batch_size, num_workers=args.dataloader_workers, shuffle=True, seed=args.seed)
    dev = dev.dataloader(batch_size=args.batch_size, num_workers=args.dataloader_workers)
    test = test.dataloader(batch_size=args.batch_size)

    # TODO: Create the model and train it.
    model = Model(args, encoder)

    # Freeze pretrained parameters
    for param in model.encoder.parameters():
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
        loss=torch.nn.BCEWithLogitsLoss(),
        metrics={"iou": SigmoidBinaryJaccardIndex()},
        logdir=logdir,
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(train) * args.epochs,
            eta_min=1e-5
        )
    )
    ''' 
    Train decoder for a bit
    '''

    decoder_epochs = min(5, args.epochs)
    model.fit(train, dev=dev, epochs=decoder_epochs)

    ''' 
    Train decoder + last few blocks of pretrained model 
    '''

    blocks = model.encoder.blocks

    fine_tune_epochs = max(0, args.epochs - decoder_epochs)

    # freeze everything in pretrained
    for p in model.encoder.parameters():
        p.requires_grad = False

    # unfreeze last two blocks
    for block in blocks[-2:]:
        for p in block.parameters():
            p.requires_grad = True

    # decoder is trainable
    for p in model.decoder.parameters():
        p.requires_grad = True

    new_optimizer = torch.optim.AdamW([
        {
            "params": model.decoder.parameters(),
            "lr": 3e-4,
        },
        {
            "params": blocks[-2:].parameters(),
            "lr": 1e-5,
        },
    ])
    model.configure(
        optimizer=new_optimizer,
        loss=torch.nn.BCEWithLogitsLoss(),
        metrics={"iou": SigmoidBinaryJaccardIndex()},
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
    with open(os.path.join(logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(image, target)` pairs.
        
        for mask in model.predict(test, data_with_labels=True, as_numpy=True):
            zeros, ones, runs = 0, 0, []
            mask = torch.sigmoid(torch.from_numpy(mask)).numpy()
            for pixel in np.reshape(mask >= 0.5, [-1]):
                if pixel:
                    if zeros or (not zeros and not ones):
                        runs.append(zeros)
                        zeros = 0
                    ones += 1
                else:
                    if ones:
                        runs.append(ones)
                        ones = 0
                    zeros += 1
            runs.append(zeros + ones)
            print(*runs, file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
