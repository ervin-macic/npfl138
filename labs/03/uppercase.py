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
parser.add_argument("--alphabet_size", default=50, type=int, help="If given, use this many most frequent chars.")
parser.add_argument("--batch_size", default=2048, type=int, help="Batch size.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=24, type=int, help="Window size to use.")
parser.add_argument("--hidden_layer_size", default=128, type=int, help="Size of the hidden layer.")
parser.add_argument("--num_models", default=4, type=int, help="Number of models.")
parser.add_argument("--weight_decay", default=0.001, type=float, help="Weight decay strength.")
parser.add_argument("--dropout", default=0.2, type=float, help="Dropout regularization.")
parser.add_argument("--label_smoothing", default=0, type=float, help="Label smoothing.")
parser.add_argument("--embedding_dim", default=16, type=int, help="Input embedding dimension.")
parser.add_argument("--num_residual_blocks", default=4, type=int, help="Number of residual FC blocks.")
parser.add_argument("--patience", default=4, type=int, help="Early stopping patience.")

# ok so window = 10 means print(uppercase_data.train.windows.__getitem__(0)) gives 
# tensor([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 12,  2,  3, 24, 14,  3, 24, 19, 2, 17, 11]) for example. so 10 + c + 10 
# when doing uppercase_data = UppercaseData(args.window, args.alphabet_size)
# train_data = Dataset(uppercase_data.train)
# print(train_data.__getitem__(0))
# (tensor([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 12,  2,  3, 24, 14,  3, 24, 19, 2, 17, 11]), tensor(1))
# 1 here means it's uppercase which makes sense since the sentence is "V období prohibice b"
# this is just a wrapper for the dataset for the dataloader
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

class ResidualBlock(torch.nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim, dim),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        return torch.relu(x + self.block(x))
    
class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, vocab_size: int):
        super().__init__()
        self._args = args

        # TODO: Implement a suitable model. The inputs are _windows_ of fixed size
        # (`args.window` characters on the left, the character in question, and
        # `args.window` characters on the right), where each character is
        # represented by a `torch.int64` index. To suitably represent the
        # characters, you can:
        # - Convert the character indices into _one-hot encoding_, which you can
        #   achieve by using `torch.nn.functional.one_hot` on the characters,
        #   and then concatenate the one-hot encodings of the window characters.
        # - Alternatively, you can experiment with `torch.nn.Embedding`s (an
        #   efficient implementation of one-hot encoding followed by a linear layer)
        #   and flattening afterwards.
        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=args.embedding_dim,
            padding_idx=0,
        )
        input_dim = (2 * args.window + 1) * args.embedding_dim
        self.input_proj = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_dim, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(args.dropout),
        )
        self.res_blocks = torch.nn.Sequential(*[
            ResidualBlock(args.hidden_layer_size, args.dropout)
            for _ in range(args.num_residual_blocks)
        ])
        self.output = torch.nn.Linear(args.hidden_layer_size, 2)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass.
        x = self.embedding(windows)
        x = self.input_proj(x)
        x = self.res_blocks(x)
        return self.output(x)
    
class EnsembleModel(npfl138.TrainableModule):
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, windows, labels=None):
        logits = torch.stack([m(windows) for m in self.models])
        probs = torch.softmax(logits, dim=-1).mean(dim=0)
        return torch.log(probs)
  
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

    # Load the data and create windows of integral character indices and integral labels.
    uppercase_data = UppercaseData(args.window, args.alphabet_size)

    train = torch.utils.data.DataLoader(
        Dataset(uppercase_data.train), args.batch_size, collate_fn=Dataset.collate, shuffle=True)
    dev = torch.utils.data.DataLoader(Dataset(uppercase_data.dev), args.batch_size, collate_fn=Dataset.collate)
    test = torch.utils.data.DataLoader(Dataset(uppercase_data.test), args.batch_size, collate_fn=Dataset.collate)

    print(f"Train dataset size: {len(train)}")
    print(f"Dev dataset size: {len(dev)}")
    print(f"Test dataset size: {len(test)}")

    vocab_size = len(uppercase_data.train.alphabet)
    
    # TODO: Implement a suitable model, optionally including regularization, select
    # good hyperparameters, and train the model.
    models = []
    for i in range(args.num_models):
        model = Model(args, vocab_size)
        models.append(model)

        non_bias_parameters = [p for n, p in models[-1].named_parameters() if "bias" not in n]
        bias_parameters = [p for n, p in models[-1].named_parameters() if "bias" in n]
        optimizer = torch.optim.AdamW(
            params=[
                {"params": non_bias_parameters, "weight_decay": args.weight_decay},
                {"params": bias_parameters, "weight_decay": 0}
            ],
            lr=1e-3,
        )
        models[-1].configure(
            optimizer=optimizer,
            loss=torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing),
            metrics={"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=2)},
            logdir=logdir,
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=len(train) * args.epochs,
                eta_min=3e-4,
            )
        )
        print(f"Training model {i + 1}: ", end="", flush=True)
        early_stop = EarlyStopping(patience=args.patience)
        models[-1].fit(train, epochs=args.epochs, dev=dev, log_config=vars(args), log_graph=(i == 0), callbacks=[early_stop])
        if early_stop.best_state is not None:
            models[-1].load_state_dict(early_stop.best_state)
        print(f"Done (best dev_accuracy={early_stop.best_value:.4f})")
        torch.save(models[-1].state_dict(), os.path.join(logdir, f"model_{i}.pt"))
    
    ensemble = EnsembleModel(models)
    ensemble.configure(
        loss=torch.nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=2)}
    )

    print(ensemble.evaluate(dev))
    print(ensemble.evaluate(test))

    # Evalute for test
    test_logits = torch.stack(list(ensemble.predict(test, as_numpy=False)))
    test_preds = test_logits.argmax(dim=-1).tolist()

    # TODO: Generate correctly capitalized test set and write the result to `predictions_file`,
    # which is by default `uppercase_test.txt` in the `logdir` directory).
    with open(os.path.join(logdir, "uppercase_test.txt"), "w", encoding="utf-8") as predictions_file:
        # We start by generating the network test set predictions; if you modified the `test` dataloader
        # or your model does not process the dataset windows, you might need to adjust the following line.
        # Now you need to utilize the network predictions and the unannotated test data (lowercased text)
        # available in `uppercase_data.test.text` to produce capitalized text and print it to the `predictions_file`.
        # my idea: wherever the model predicts 1 use capital letter, otherwise 0
        text = list(uppercase_data.test.text)
        for i, pred_class in enumerate(test_preds):
            if pred_class == 1:
                text[i] = text[i].upper()
        predictions_file.write("".join(text))

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
