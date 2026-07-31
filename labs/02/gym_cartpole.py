#!/usr/bin/env python3
import argparse

import numpy as np
import torch
import torchmetrics

import npfl138

npfl138.require_version("2526.2")
from npfl138.datasets.gym_cartpole_dataset import GymCartpoleDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument(
    "--evaluate",
    default=False,
    action="store_true",
    help="Evaluate the given model.",
)
parser.add_argument(
    "--recodex", default=False, action="store_true", help="Evaluation in ReCodEx."
)
parser.add_argument(
    "--render",
    default=False,
    action="store_true",
    help="Render during evaluation.",
)
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument(
    "--threads", default=1, type=int, help="Maximum number of threads to use."
)
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--batch_size", default=..., type=int, help="Batch size.")
parser.add_argument("--epochs", default=..., type=int, help="Number of epochs.")
parser.add_argument(
    "--hidden_layer_size",
    default=128,
    type=int,
    help="Size of the hidden layer.",
)
parser.add_argument(
    "--learning_rate", default=0.01, type=float, help="Initial learning rate."
)
parser.add_argument(
    "--learning_rate_final",
    default=None,
    type=float,
    help="Final learning rate.",
)
parser.add_argument(
    "--momentum",
    default=None,
    type=float,
    help="Nesterov momentum to use in SGD.",
)
parser.add_argument(
    "--optimizer",
    default="SGD",
    choices=["SGD", "Adam"],
    help="Optimizer to use.",
)
parser.add_argument(
    "--decay",
    default=None,
    choices=["linear", "exponential", "cosine"],
    help="Decay type.",
)
parser.add_argument(
    "--model",
    default="gym_cartpole_model.pt",
    type=str,
    help="Output model path.",
)


def evaluate_model(
    model: torch.nn.Module,
    seed: int = 42,
    episodes: int = 100,
    render: bool = False,
    report_per_episode: bool = False,
) -> float:
    """Evaluate the given model on CartPole-v1 environment.

    Returns the average score achieved on the given number of episodes.
    """
    import gymnasium as gym

    # Create the environment.
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    env.reset(seed=seed)

    # Evaluate the episodes.
    total_score = 0
    for episode in range(episodes):
        observation, score, done = env.reset()[0], 0, False
        while not done:
            prediction = (
                model.predict_batch(torch.from_numpy(observation).unsqueeze(0))
                .squeeze(0)
                .numpy(force=True)
            )
            assert len(prediction) == 2, "The model must output two values."
            action = np.argmax(prediction)

            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            done = terminated or truncated

        total_score += score
        if report_per_episode:
            print(f"The episode {episode + 1} finished with score {score}.")
    return total_score / episodes


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()

        # TODO: Create the model layers, with the last layer having 2 outputs.
        # To store a list of layers, you can use either torch.nn.Sequential
        # or torch.nn.ModuleList; you should *not* use a Python list.
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(GymCartpoleDataset.FEATURES, args.hidden_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_layer_size, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # TODO: Run your model and return its output.
        return self.model(inputs)


def main(args: argparse.Namespace) -> torch.nn.Module | None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads, args.recodex)
    npfl138.global_keras_initializers()

    if not args.evaluate:
        if args.batch_size is ...:
            raise ValueError(
                "You must specify the batch size, either in the defaults or on the command line."
            )
        if args.epochs is ...:
            raise ValueError(
                "You must specify the number of epochs, either in the defaults or on the command line."
            )

        # Load the provided dataset. The dataset.train is a collection of 100 examples,
        # each being a pair of (inputs, label), where:
        # - inputs is a vector with GymCartpoleDataset.FEATURES floating point values,
        # - label is a gold 0/1 class index.
        dataset = GymCartpoleDataset()

        train = torch.utils.data.DataLoader(
            dataset.train, args.batch_size, shuffle=True
        )

        model = Model(args)

        # TODO: Configure the model for training.
        # Parse arguments for optimizer and lr_scheduler
        start_lr = args.learning_rate
        final_lr = args.learning_rate_final
        total_steps = len(train) * args.epochs

        match args.optimizer:
            case "SGD":
                if args.momentum is not None:
                    optimizer = torch.optim.SGD(
                        model.parameters(),
                        lr=start_lr,
                        momentum=args.momentum,
                        nesterov=True,
                    )
                else:
                    optimizer = torch.optim.SGD(model.parameters(), lr=start_lr)
            case "Adam":
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=start_lr,
                )

        scheduler = None
        match args.decay:
            case "linear":
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=(final_lr / start_lr),
                    total_iters=total_steps,
                )
            case "exponential":
                k = total_steps
                gamma = np.float64(final_lr / start_lr) ** (
                    np.float64(1) / np.float64(k)
                )
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=gamma
                )
            case "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=total_steps,
                    eta_min=final_lr,
                )

        model.configure(
            optimizer=optimizer,
            scheduler=scheduler,
            loss=torch.nn.CrossEntropyLoss(),
            metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=2)},
            logdir=npfl138.format_logdir(
                "logs/{file-}{timestamp}{-config}", **vars(args)
            ),
        )

        # TODO: Train the model.
        #
        # Note that the fit method accepts a callbacks argument, which is a list
        # of callables that are called at the end of each epoch, each being called
        # with the model, epoch, and logs (a dictionary with logged losses and metrics).
        def callback(
            model: Model, epoch: int, logs: dict[str, float]
        ) -> None | npfl138.StopTraining:
            # When you add items to the logs dictionary, they will be logged
            # both to the console and to TensorBoard.
            pass

        model.fit(train, epochs=args.epochs, callbacks=[callback])

        # Save the model, both the hyperparameters and the parameters. If you
        # added additional arguments to the Model constructor beyond args,
        # you would have to add them to the save_options call below.
        model.save_options(f"{args.model}.json", args=args)
        model.save_weights(args.model)

    else:
        # Evaluating, either manually or in ReCodEx.
        model = Model(**Model.load_options(f"{args.model}.json"))
        model.load_weights(args.model)

        if args.recodex:
            return model
        else:
            score = evaluate_model(
                model, seed=args.seed, render=args.render, report_per_episode=True
            )
            print(f"The average score was {score}.")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)