#!/usr/bin/env python3
import argparse
import os

import torch
import torchaudio.models.decoder

import npfl138
npfl138.require_version("2526.8.1")
from npfl138.datasets.common_voice_cs import CommonVoiceCs

# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--cle_dim", default=32, type=int, help="CLE embedding dimension.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn", default="LSTM", choices=["LSTM", "GRU"], help="RNN layer type.")
parser.add_argument("--rnn_dim", default=64, type=int, help="RNN layer dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--dropout", default=0.2, type=float, help="Dropout probability.")

class Dataset(npfl138.TransformedDataset):
    def transform(self, example):
        audio = torch.as_tensor(example["mfccs"])
        target_ids = torch.as_tensor(CommonVoiceCs.LETTERS_VOCAB.indices(list(example["sentence"])))
        return audio, target_ids

    def collate(self, batch):
        audio, target_ids = zip(*batch)

        audio_lengths = torch.tensor([len(x) for x in audio])
        target_lengths = torch.tensor([len(x) for x in target_ids])

        target_ids = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=CommonVoiceCs.PAD)
        audio = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True, padding_value=CommonVoiceCs.PAD)
    
        return (audio, audio_lengths), (target_ids, target_lengths)

class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: CommonVoiceCs.Dataset) -> None:
        super().__init__()

        blank_index = CommonVoiceCs.PAD
        blank_str = CommonVoiceCs.LETTER_NAMES[blank_index]

        self._decoder = torchaudio.models.decoder.ctc_decoder(
            lexicon=None, 
            tokens=CommonVoiceCs.LETTER_NAMES,         # List of token strings or path to tokens file
            lm=None,               # Path to KenLM model file (.arpa or .bin), if available
            nbest=1,               # Return top N hypotheses per sample
            beam_size=5,          # Beam width for search
            blank_token=blank_str,       # Blank token string
            sil_token=" ",
        )
        self._ctc_loss = torch.nn.CTCLoss(blank=CommonVoiceCs.PAD)

        self._dropout = torch.nn.Dropout(args.dropout)

        if args.rnn == "LSTM":
            self._rnn = torch.nn.LSTM(
                input_size=CommonVoiceCs.MFCC_DIM,
                hidden_size=args.rnn_dim,
                bidirectional=True,
                batch_first=True,
            )
        else:
            self._rnn = torch.nn.GRU(
                input_size=CommonVoiceCs.MFCC_DIM,
                hidden_size=args.rnn_dim,
                bidirectional=True,
                batch_first=True,
            )

        self._output_layer = torch.nn.Linear(
            in_features=2 * args.rnn_dim,
            out_features=len(CommonVoiceCs.LETTER_NAMES),
        )

    def forward(self, audio: torch.Tensor, audio_lengths: torch.Tensor) -> torch.Tensor:

        packed = torch.nn.utils.rnn.pack_padded_sequence(audio, lengths=audio_lengths, batch_first=True, enforce_sorted=False)

        packed, _ = self._rnn(packed)

        hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)

        hidden = self._dropout(hidden)
        logits = self._output_layer(hidden)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs = log_probs.permute(1, 0, 2)

        return log_probs

    def compute_loss(self, y_pred, y, audio, audio_lengths):
        target_ids, target_lengths = y
        return self._ctc_loss(y_pred, target_ids, audio_lengths, target_lengths)

    def compute_metrics(self, y_pred, y, audio, audio_lengths):
        target_ids, target_lengths = y
        if not self.training:
            predictions = self.ctc_decoding(y_pred, audio_lengths)
            self.metrics["edit_distance"].update(predictions, target_ids)
            return self.metrics
        return {}

    def ctc_decoding(self, y_pred, audio_lengths):
        y_pred = y_pred.detach().cpu()

        predictions = []

        for i, length in enumerate(audio_lengths):
            # ctc decoder wants shape [1, T, C], I give [T, C]
            emission = y_pred[:length, i, :].unsqueeze(0)

            # beam search on this emission
            hypotheses = self._decoder(emission)

            predictions.append(torch.tensor(hypotheses[0][0].tokens))

        return predictions

    def predict_step(self, xs):
        with torch.no_grad():
            y_pred = self.forward(xs[0], xs[1])
            predictions = self.ctc_decoding(y_pred, xs[1])

            for prediction in predictions:
                yield prediction

def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a suitable logdir for the logs and the predictions.
    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Load the data.
    common_voice = CommonVoiceCs()

    train = Dataset(common_voice.train).dataloader(args.batch_size, shuffle=True)
    dev = Dataset(common_voice.dev).dataloader(args.batch_size)
    test = Dataset(common_voice.test).dataloader(args.batch_size)

    model = Model(args, train)

    model.configure(
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4),
            metrics={
                "edit_distance": CommonVoiceCs.EditDistanceMetric(ignore_index=CommonVoiceCs.PAD),
            },
            logdir=logdir,
        )
    
    logs = model.fit(
            train,
            dev=dev,
            epochs=args.epochs,
        )
    
    # Generate test set annotations, but in `logdir` to allow parallel execution.
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "speech_recognition.txt"), "w", encoding="utf-8") as predictions_file:
        predictions = model.predict(test, data_with_labels=True, as_numpy=True)
        for sentence in predictions:
            print("".join(CommonVoiceCs.LETTERS_VOCAB.strings(sentence)), file=predictions_file)

    # Generate dev set annotations
    with open(os.path.join(logdir, "speech_recognition_dev.txt"), "w", encoding="utf-8") as predictions_file:
        dev_predictions = model.predict(dev, data_with_labels=True, as_numpy=True)
        for sentence in dev_predictions:
            print("".join(CommonVoiceCs.LETTERS_VOCAB.strings(sentence)), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
