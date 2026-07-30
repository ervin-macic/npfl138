#!/usr/bin/env python3
import argparse
from sortedcontainers import SortedDict

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_path", default="numpy_entropy_data.txt", type=str, help="Data distribution path.")
parser.add_argument("--model_path", default="numpy_entropy_model.txt", type=str, help="Model distribution path.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float, float]:
    # TODO: Load data distribution, each line containing a datapoint -- a string.
    with open(args.data_path, "r") as data:
        data_dict = SortedDict()
        for line in data:
            line = line.rstrip("\n")
            # TODO: Process the line, aggregating data with built-in Python
            # data structures (not NumPy, which is not suitable for incremental
            # addition and string mapping).
            if not line in data_dict:
                data_dict[line] = 1
            else: 
                data_dict[line] += 1

    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. Alternatively,
    # the NumPy array might be created after loading the model distribution.
    data_keys = np.array(list(data_dict.keys()))
    data_values = np.array(list(data_dict.values()))
    data_dist = data_values / data_values.sum()

    # TODO: Load model distribution, each line `string \t probability`.
    with open(args.model_path, "r") as model:
        model_dict = SortedDict()
        for line in model:
            line = line.rstrip("\n")
            # TODO: Process the line, aggregating using Python data structures.
            key, value = line.split("\t", 1)
            model_dict[key] = float(value)

    model_dist = np.array([model_dict.get(k, 0.0) for k in data_keys])
    
    # TODO: Compute the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    # numeric-stable entropy: ignore zero probabilities
    mask = data_dist > 0
    entropy = (-data_dist[mask] * np.log(data_dist[mask])).sum()

    # TODO: Compute cross-entropy H(data distribution, model distribution).
    # When some data distribution elements are missing in the model distribution,
    # the resulting crossentropy should be `np.inf`.
    # If any data-supported key has zero probability in the model,
    # cross-entropy / KL should be infinite.
    zero_in_model = (model_dist == 0) & (data_dist > 0)
    if zero_in_model.any():
        crossentropy = np.inf
        kl_divergence = np.inf
    else:
        crossentropy = (-data_dist[mask] * np.log(model_dist[mask])).sum()
        kl_divergence = crossentropy - entropy

    # Return the computed values for ReCodEx to validate.
    return entropy, crossentropy, kl_divergence


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(main_args)
    print(f"Entropy: {entropy:.2f} nats")
    print(f"Crossentropy: {crossentropy:.2f} nats")
    print(f"KL divergence: {kl_divergence:.2f} nats")
