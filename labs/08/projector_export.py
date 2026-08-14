#!/usr/bin/env python
import argparse
import os

import numpy as np
import torch
import torch.utils.tensorboard


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Directory containing the embeddings.")
    parser.add_argument("--elements", default=None, type=int, help="Words to export.")
    parser.add_argument("--output_dir", default="embeddings", type=str, help="Output directory.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Locate the TensorBoard Projector files
    tensors_file = os.path.join(args.input_dir, "tensors.tsv")
    metadata_file = os.path.join(args.input_dir, "metadata.tsv")

    # Load the embeddings
    embeddings = np.loadtxt(tensors_file, dtype=np.float32, delimiter="\t")

    # Load the words
    with open(metadata_file, "r", encoding="utf-8") as metadata:
        words = [line.rstrip("\n") for line in metadata]

    elements = len(words)
    dim = embeddings.shape[1]

    if args.elements is not None:
        elements = min(args.elements, elements)

    embeddings = embeddings[:elements]
    words = words[:elements]

    # Save the embeddings
    torch.utils.tensorboard.SummaryWriter(args.output_dir).add_embedding(
        torch.tensor(embeddings),
        metadata=words,
        tag="embeddings",
    )