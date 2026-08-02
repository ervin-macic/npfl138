#!/usr/bin/env python3
import argparse
import os

import torch
import torchmetrics

import npfl138
npfl138.require_version("2526.3")
from npfl138.datasets.uppercase_data import UppercaseData

# TODO: Set reasonable values for the hyperparameters, especially for
# `alphabet_size`, `batch_size`, `epochs`, and `window`.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=0, type=int, help="If given, use this many most frequent chars.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=10, type=int, help="Window size to use.")
parser.add_argument("--hidden_layer_size", default=16, type=int, help="Size of the hidden layer.")
parser.add_argument("--num_models", default=1, type=int, help="Number of models.")


class Dataset(torch.utils.data.Dataset):
    # A dataset must always implement at least `__len__` and `__getitem__`.
    def __init__(self, uppercase_dataset: UppercaseData.Dataset):
        self.windows = uppercase_dataset.windows
        self.labels = uppercase_dataset.labels

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return self.windows[index], self.labels[index]

    # When a dataset implements `__getitems__`, this method is used to generate whole batches in a single call.
    # However, `__getitems__` is expected to return a list of items that are later collated together.
    # For maximum speedup, we already return a whole batch from `__getitems__` and implement a trivial `collate`.
    def __getitems__(self, indices):
        indices = torch.as_tensor(indices)
        return self.windows[indices], self.labels[indices]

    @staticmethod
    def collate(batch):
        return batch


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data and create windows of integral character indices and integral labels.
    uppercase_data = UppercaseData(args.window, args.alphabet_size)
    # train_data = Dataset(uppercase_data.train)
    # print(train_data.__getitem__(0))
    # print(uppercase_data.train.windows.__getitem__(0))
    # print(uppercase_data.train.text[0:20])
    train = torch.utils.data.DataLoader(Dataset(uppercase_data.train), args.batch_size, collate_fn=Dataset.collate, shuffle=True)
    dev = torch.utils.data.DataLoader(Dataset(uppercase_data.dev), args.batch_size, collate_fn=Dataset.collate)
    test = torch.utils.data.DataLoader(Dataset(uppercase_data.test), args.batch_size, collate_fn=Dataset.collate)
    print(uppercase_data.test.text[:10])
    print(uppercase_data.test.labels[:50])
    print(uppercase_data.test.labels.float().mean()) 
    dev_windows = uppercase_data.dev.windows
    test_windows = uppercase_data.test.windows
    print("Dev % index-0:", (dev_windows == 0).float().mean().item())
    print("Test % index-0:", (test_windows == 0).float().mean().item())
    print(print(len(train)))
if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
