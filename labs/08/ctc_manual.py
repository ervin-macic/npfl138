#!/usr/bin/env python3
import argparse

import torch

import npfl138
npfl138.require_version("2526.8")
from npfl138.datasets.morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn", default="LSTM", choices=["LSTM", "GRU"], help="RNN layer type.")
parser.add_argument("--rnn_dim", default=64, type=int, help="RNN layer dimension.")
parser.add_argument("--seed", default=41, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=128, type=int, help="Word embedding dimension.")


class Dataset(npfl138.TransformedDataset):
    def transform(self, example):
        word_ids = torch.tensor(self.dataset.words.string_vocab.indices(example["words"]), dtype=torch.long)
        tags = [tag for tag in example["tags"] if tag.startswith("B-")]
        tag_ids = torch.tensor(self.dataset.tags.string_vocab.indices(tags), dtype=torch.long)
        return word_ids, tag_ids

    def collate(self, batch):
        word_ids, tag_ids = zip(*batch)
        word_ids = torch.nn.utils.rnn.pad_sequence(word_ids, batch_first=True, padding_value=MorphoDataset.PAD)
        tag_ids = torch.nn.utils.rnn.pad_sequence(tag_ids, batch_first=True, padding_value=MorphoDataset.PAD)

        return word_ids, tag_ids


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()

        self._word_embedding = torch.nn.Embedding(
            num_embeddings=len(train.words.string_vocab),
            embedding_dim=args.we_dim,
        )

        if args.rnn == "LSTM":
            self._word_rnn = torch.nn.LSTM(
                input_size=args.we_dim,
                hidden_size=args.rnn_dim,
                bidirectional=True,
                batch_first=True,
            )
        else:
            self._word_rnn = torch.nn.GRU(
                input_size=args.we_dim,
                hidden_size=args.rnn_dim,
                bidirectional=True,
                batch_first=True,
            )

        self._output_layer = torch.nn.Linear(
            in_features=args.rnn_dim,
            out_features=len(train.tags.string_vocab),
        )

    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
        hidden = self._word_embedding(word_ids)

        lengths = (word_ids != MorphoDataset.PAD).sum(dim=1).cpu()

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            hidden,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        packed, _ = self._word_rnn(packed)

        hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed,
            batch_first=True,
        )

        # [B, T, 2 * rnn_dim] -> [B, T, rnn_dim]
        hidden = hidden[:, :, :self._word_rnn.hidden_size] + \
                 hidden[:, :, self._word_rnn.hidden_size:]

        hidden = self._output_layer(hidden)

        # [B, T, C] -> [B, C, T]
        return hidden.permute(0, 2, 1)

    def compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        word_ids: torch.Tensor,
    ) -> torch.Tensor:

        # y_pred: [B, C, T]
        batch_size, num_tags, max_time = y_pred.shape
        blank = MorphoDataset.PAD
        neg_inf = -1e9

        log_probs = y_pred.log_softmax(dim=1)

        input_lengths = (word_ids != MorphoDataset.PAD).sum(dim=1)
        target_lengths = (y_true != MorphoDataset.PAD).sum(dim=1)

        # Extended CTC targets:
        #
        # target [a, b, c]
        # becomes
        # [blank, a, blank, b, blank, c, blank]
        max_target_len = y_true.shape[1]
        ext_len = 2 * max_target_len + 1

        extended = torch.full(
            (batch_size, ext_len),
            blank,
            dtype=torch.long,
            device=y_true.device,
        )

        if max_target_len > 0:
            extended[:, 1::2] = y_true

        # Emission probability for every extended state.
        #
        # [B, C, T] -> [B, T, C]
        # then gather the probability of the label belonging to state s.
        emissions = log_probs.permute(0, 2, 1).gather(
            2,
            extended.unsqueeze(1).expand(batch_size, max_time, ext_len),
        )
        # emissions: [B, T, ext_len]

        # States beyond the actual target length are invalid.
        state_ids = torch.arange(ext_len, device=y_true.device).unsqueeze(0)
        valid_states = state_ids <= (2 * target_lengths).unsqueeze(1)

        # alpha[t, s] = log probability of reaching state s after t frames.
        alpha = torch.full(
            (batch_size, ext_len),
            neg_inf,
            device=y_true.device,
        )

        # Initial state is blank.
        alpha[:, 0] = emissions[:, 0, 0]

        # The first target can also be reached immediately.
        if max_target_len > 0:
            has_target = target_lengths > 0
            alpha[:, 1] = torch.where(
                has_target,
                emissions[:, 0, 1],
                torch.full_like(emissions[:, 0, 1], neg_inf),
            )

        alpha = torch.where(
            valid_states,
            alpha,
            torch.full_like(alpha, neg_inf),
        )

        # Forward CTC dynamic programming.
        for t in range(1, max_time):
            previous = alpha

            # Stay at the same state.
            stay = previous

            # Move from s-1 -> s.
            from_previous = torch.cat(
                [
                    torch.full(
                        (batch_size, 1),
                        neg_inf,
                        device=y_true.device,
                    ),
                    previous[:, :-1],
                ],
                dim=1,
            )

            # Move from s-2 -> s.
            from_two_back = torch.cat(
                [
                    torch.full(
                        (batch_size, 2),
                        neg_inf,
                        device=y_true.device,
                    ),
                    previous[:, :-2],
                ],
                dim=1,
            )

            # s-2 -> s is allowed only when:
            # 1. current state is not blank
            # 2. current label differs from the label two states back
            can_skip = torch.zeros(
                (batch_size, ext_len),
                dtype=torch.bool,
                device=y_true.device,
            )

            if ext_len > 2:
                can_skip[:, 2:] = (
                    (extended[:, 2:] != blank)
                    & (extended[:, 2:] != extended[:, :-2])
                )

            from_two_back = torch.where(
                can_skip,
                from_two_back,
                torch.full_like(from_two_back, neg_inf),
            )

            alpha_new = torch.logsumexp(
                torch.stack(
                    [stay, from_previous, from_two_back],
                    dim=-1,
                ),
                dim=-1,
            )

            alpha_new = alpha_new + emissions[:, t, :]

            alpha_new = torch.where(
                valid_states,
                alpha_new,
                torch.full_like(alpha_new, neg_inf),
            )

            # Once an example reaches the end of its input, freeze its alpha.
            still_active = (input_lengths > t).unsqueeze(1)

            alpha = torch.where(
                still_active,
                alpha_new,
                previous,
            )

        # The sequence can finish in either:
        #
        # ... target
        # ... blank
        #
        # i.e. states 2U-1 and 2U.
        last_target_state = 2 * target_lengths
        previous_state = (last_target_state - 1).clamp_min(0)

        final_1 = alpha.gather(
            1,
            last_target_state.unsqueeze(1),
        ).squeeze(1)

        final_2 = alpha.gather(
            1,
            previous_state.unsqueeze(1),
        ).squeeze(1)

        log_likelihood = torch.where(
            target_lengths > 0,
            torch.logsumexp(
                torch.stack([final_1, final_2], dim=-1),
                dim=-1,
            ),
            final_1,
        )

        # Same normalization as CTCLoss(reduction="mean"):
        # divide each example by its target length.
        lengths_for_loss = target_lengths.clamp_min(1)

        losses = -log_likelihood / lengths_for_loss

        # zero_infinity=True behaviour.
        impossible = log_likelihood <= neg_inf / 2
        losses = torch.where(
            impossible,
            torch.zeros_like(losses),
            losses,
        )

        return losses.mean()

    def ctc_decoding(
        self,
        logits: torch.Tensor,
        word_ids: torch.Tensor,
    ) -> list[torch.Tensor]:

        # logits: [B, C, T]
        best_labels = logits.argmax(dim=1)

        predictions = []

        for batch_idx in range(word_ids.shape[0]):
            length = int(
                (word_ids[batch_idx] != MorphoDataset.PAD).sum().item()
            )

            sequence = best_labels[batch_idx, :length]

            # Collapse consecutive repetitions.
            sequence = torch.unique_consecutive(sequence)

            # Remove CTC blanks.
            sequence = sequence[sequence != MorphoDataset.PAD]

            predictions.append(sequence)

        return predictions

    def compute_metrics(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        word_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        predictions = self.ctc_decoding(y_pred, word_ids)

        self.metrics["edit_distance"].update(predictions, y_true)

        return self.metrics

    def predict_step(self, xs):
        with torch.no_grad():
            yield from self.ctc_decoding(self.forward(*xs), *xs)


def main(args: argparse.Namespace) -> dict[str, float]:
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    morpho = MorphoDataset(
        "czech_cnec",
        max_sentences=args.max_sentences,
    )

    train = Dataset(morpho.train).dataloader(
        batch_size=args.batch_size,
        shuffle=True,
    )

    dev = Dataset(morpho.dev).dataloader(
        batch_size=args.batch_size,
    )

    model = Model(args, morpho.train)

    model.configure(
        optimizer=torch.optim.Adam(model.parameters()),
        metrics={
            "edit_distance": npfl138.metrics.EditDistance(
                ignore_index=morpho.PAD
            ),
        },
        logdir=npfl138.format_logdir(
            "logs/{file-}{timestamp}{-config}",
            **vars(args),
        ),
    )

    logs = model.fit(
        train,
        dev=dev,
        epochs=args.epochs,
    )

    return {
        metric: value
        for metric, value in logs.items()
    }


if __name__ == "__main__":
    main_args = parser.parse_args(
        [] if "__file__" not in globals() else None
    )
    main(main_args)