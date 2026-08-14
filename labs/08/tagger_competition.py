#!/usr/bin/env python3
import argparse
import os

import torch
import torchaudio.models.decoder

import npfl138
npfl138.require_version("2526.8.1")
from npfl138.datasets.common_voice_cs import CommonVoiceCs


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--cle_dim", default=32, type=int)
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--recodex", default=False, action="store_true")
parser.add_argument("--rnn", default="GRU", choices=["LSTM", "GRU"])
parser.add_argument("--rnn_dim", default=256, type=int)
parser.add_argument("--rnn_layers", default=2, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--threads", default=0, type=int)
parser.add_argument("--dropout", default=0.2, type=float)
parser.add_argument("--beam_size", default=3, type=int)
parser.add_argument("--learning_rate", default=5e-4, type=float)
parser.add_argument("--debug_samples", default=8, type=int)


class Dataset(npfl138.TransformedDataset):
    def transform(self, example):
        audio = torch.as_tensor(example["mfccs"], dtype=torch.float32)
        target_ids = torch.as_tensor(CommonVoiceCs.LETTERS_VOCAB.indices(list(example["sentence"])), dtype=torch.long)
        return audio, target_ids

    def collate(self, batch):
        audio, target_ids = zip(*batch)
        audio_lengths = torch.tensor([len(x) for x in audio], dtype=torch.long)
        target_lengths = torch.tensor([len(x) for x in target_ids], dtype=torch.long)
        target_ids = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=CommonVoiceCs.PAD)
        audio = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True, padding_value=0.0)
        return (audio, audio_lengths), (target_ids, target_lengths)


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: CommonVoiceCs.Dataset) -> None:
        super().__init__()

        self._use_amp = torch.cuda.is_available()
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")

        blank_str = CommonVoiceCs.LETTER_NAMES[CommonVoiceCs.PAD]

        self._decoder = torchaudio.models.decoder.ctc_decoder(
            lexicon=None,
            tokens=CommonVoiceCs.LETTER_NAMES,
            lm=None,
            nbest=1,
            beam_size=args.beam_size,
            blank_token=blank_str,
            sil_token=" ",
        )

        self._ctc_loss = torch.nn.CTCLoss(blank=CommonVoiceCs.PAD, zero_infinity=True)
        self._input_norm = torch.nn.LayerNorm(CommonVoiceCs.MFCC_DIM)

        rnn_cls = torch.nn.LSTM if args.rnn == "LSTM" else torch.nn.GRU
        self._rnn = rnn_cls(
            input_size=CommonVoiceCs.MFCC_DIM,
            hidden_size=args.rnn_dim,
            num_layers=args.rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=args.dropout if args.rnn_layers > 1 else 0.0,
        )

        self._dropout = torch.nn.Dropout(args.dropout)
        self._output_layer = torch.nn.Linear(2 * args.rnn_dim, len(CommonVoiceCs.LETTER_NAMES))

    def forward(self, audio: torch.Tensor, audio_lengths: torch.Tensor) -> torch.Tensor:
        audio = self._input_norm(audio)

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            audio,
            lengths=audio_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self._use_amp):
            packed, _ = self._rnn(packed)
            hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
            hidden = self._dropout(hidden)
            logits = self._output_layer(hidden)

        log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
        return log_probs.permute(1, 0, 2)

    def compute_loss(self, y_pred, y, audio, audio_lengths):
        target_ids, target_lengths = y
        return self._ctc_loss(y_pred, target_ids, audio_lengths, target_lengths)

    def compute_metrics(self, y_pred, y, audio, audio_lengths):
        if self.training:
            return {}

        target_ids, _ = y
        predictions = self.greedy_ctc_decoding(y_pred, audio_lengths)
        self.metrics["edit_distance"].update(predictions, target_ids)
        return self.metrics

    @staticmethod
    def _collapse_ctc(sequence):
        sequence = torch.unique_consecutive(sequence)
        return sequence[sequence != CommonVoiceCs.PAD]

    def greedy_ctc_decoding(self, y_pred, audio_lengths):
        best = y_pred.argmax(dim=-1)
        return [self._collapse_ctc(best[:length, i]) for i, length in enumerate(audio_lengths)]

    def beam_ctc_decoding(self, y_pred, audio_lengths):
        y_pred = y_pred.detach().cpu()
        audio_lengths = audio_lengths.detach().cpu()

        predictions = []
        for i, length in enumerate(audio_lengths):
            hypotheses = self._decoder(y_pred[:length, i, :].unsqueeze(0))
            if not hypotheses or not hypotheses[0]:
                predictions.append(torch.empty(0, dtype=torch.long))
            else:
                predictions.append(torch.as_tensor(hypotheses[0][0].tokens, dtype=torch.long))
        return predictions

    def predict_step(self, xs):
        with torch.inference_mode():
            y_pred = self.forward(xs[0], xs[1])
            for prediction in self.beam_ctc_decoding(y_pred, xs[1]):
                yield prediction

    def debug_batch(self, batch, num_samples=8):
        (audio, audio_lengths), (target_ids, target_lengths) = batch
        device = next(self.parameters()).device
        audio = audio.to(device)
        target_ids = target_ids.to(device)
        target_lengths = target_lengths.to(device)
        audio_lengths = audio_lengths.cpu()

        with torch.inference_mode():
            y_pred = self.forward(audio, audio_lengths)

        best = y_pred.argmax(dim=-1)
        blank = CommonVoiceCs.PAD

        total_frames = sum(len(best[:length, i]) for i, length in enumerate(audio_lengths))
        blank_frames = sum((best[:length, i] == blank).sum().item() for i, length in enumerate(audio_lengths))
        blank_ratio = blank_frames / max(total_frames, 1)

        predictions = self.greedy_ctc_decoding(y_pred, audio_lengths)

        print(f"device={device}, AMP={self._use_amp}")
        print(f"audio={tuple(audio.shape)}, y_pred={tuple(y_pred.shape)}")
        print(f"audio lengths: {audio_lengths.min().item()}-{audio_lengths.max().item()} (mean={audio_lengths.float().mean().item():.1f})")
        print(f"target lengths: {target_lengths.min().item()}-{target_lengths.max().item()} (mean={target_lengths.float().mean().item():.1f})")
        print(f"blank ratio={blank_ratio:.4f}")

        for i in range(min(num_samples, len(predictions))):
            reference_ids = target_ids[i, :target_lengths[i]].cpu().tolist()
            prediction_ids = predictions[i].cpu().tolist()
            reference = "".join(CommonVoiceCs.LETTERS_VOCAB.strings(reference_ids))
            prediction = "".join(CommonVoiceCs.LETTERS_VOCAB.strings(prediction_ids))
            print(f"[{i}] REF={reference!r}")
            print(f"    PRED={prediction!r}")


def write_predictions(model, dataset, output_path, debug_samples=0):
    predictions = model.predict(dataset, data_with_labels=True, as_numpy=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, sentence in enumerate(predictions):
            text = "".join(CommonVoiceCs.LETTERS_VOCAB.strings(sentence))
            print(text, file=f)
            if i < debug_samples:
                print(f"[{i}] {text!r}")

    print(f"Wrote {len(predictions)} predictions to {output_path}")


def main(args: argparse.Namespace) -> None:
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    print(f"CUDA={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU={torch.cuda.get_device_name(0)}")

    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    common_voice = CommonVoiceCs()
    train = Dataset(common_voice.train).dataloader(args.batch_size, shuffle=True)
    dev = Dataset(common_voice.dev).dataloader(args.batch_size)
    test = Dataset(common_voice.test).dataloader(args.batch_size)

    model = Model(args, train)

    print(f"RNN={args.rnn} hidden={args.rnn_dim} layers={args.rnn_layers} batch={args.batch_size}")
    print(f"epochs={args.epochs} lr={args.learning_rate} dropout={args.dropout} beam={args.beam_size}")
    print(f"parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    model.configure(
        optimizer=torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4),
        metrics={"edit_distance": CommonVoiceCs.EditDistanceMetric(ignore_index=CommonVoiceCs.PAD)},
        logdir=logdir,
    )

    print("Before training:")
    model.debug_batch(next(iter(train)), args.debug_samples)

    model.fit(train, dev=dev, epochs=args.epochs)

    print("After training:")
    model.debug_batch(next(iter(dev)), args.debug_samples)

    os.makedirs(logdir, exist_ok=True)

    print("Generating test predictions...")
    write_predictions(model, test, os.path.join(logdir, "speech_recognition.txt"), args.debug_samples)

    print("Generating dev predictions...")
    write_predictions(model, dev, os.path.join(logdir, "speech_recognition_dev.txt"), args.debug_samples)

    print(f"Predictions written to {logdir}")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)