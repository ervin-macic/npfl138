#!/usr/bin/env python3
import argparse
import os

import fasttext
import fasttext.util
import torch
import torchmetrics

import npfl138
npfl138.require_version("2526.7")
from npfl138.datasets.morpho_dataset import MorphoDataset
from npfl138.datasets.morpho_analyzer import MorphoAnalyzer

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--cle_dim", default=32, type=int, help="CLE embedding dimension.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn", default="LSTM", choices=["LSTM", "GRU"], help="RNN layer type.")
parser.add_argument("--rnn_dim", default=128, type=int, help="RNN layer dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--word_masking", default=0.1, type=float, help="Mask words with the given probability.")
parser.add_argument("--analyzer_dim", default=32, type=int, help="Analyzer embedding dimension.")
parser.add_argument("--dropout", default=0.2, type=float, help="Dropout probability.")


def load_fasttext_embeddings(train):
    fasttext.util.download_model("cs", if_exists="ignore")
    fasttext_model = fasttext.load_model("cc.cs.300.bin")

    embedding_dim = fasttext_model.get_dimension()

    embedding_matrix = torch.empty(len(train.words.string_vocab), embedding_dim, dtype=torch.float32)

    for word_id in range(len(train.words.string_vocab)):
        word = train.words.string_vocab.string(word_id)
        embedding_matrix[word_id] = torch.from_numpy(fasttext_model.get_word_vector(word))

    return embedding_matrix


class Dataset(npfl138.TransformedDataset):
    def __init__(self, dataset, analyzer, tag_to_id):
        super().__init__(dataset)

        self.analyzer = analyzer
        self.tag_to_id = tag_to_id
        self.num_tags = len(tag_to_id)

    def transform(self, example):
        word_ids = torch.tensor(self.dataset.words.string_vocab.indices(example["words"]), dtype=torch.long)
        tag_ids = torch.tensor(self.dataset.tags.string_vocab.indices(example["tags"]), dtype=torch.long)

        analyzer_features = torch.zeros(len(example["words"]), self.num_tags, dtype=torch.float32)

        for word_index, word in enumerate(example["words"]):
            analyses = self.analyzer.get(word)

            for analysis in analyses:
                tag = analysis.tag

                if tag in self.tag_to_id:
                    analyzer_features[word_index, self.tag_to_id[tag]] = 1.0

        return word_ids, example["words"], analyzer_features, tag_ids

    def collate(self, batch):
        word_ids, words, analyzer_features, tag_ids = zip(*batch)

        word_ids = torch.nn.utils.rnn.pad_sequence(word_ids, batch_first=True)
        unique_words, words_indices = self.dataset.cle_batch(words)
        tag_ids = torch.nn.utils.rnn.pad_sequence(tag_ids, batch_first=True)
        analyzer_features = torch.nn.utils.rnn.pad_sequence(analyzer_features, batch_first=True)

        return (word_ids, unique_words, words_indices, analyzer_features), tag_ids


class Model(npfl138.TrainableModule):
    class MaskElements(torch.nn.Module):
        """A layer randomly masking elements with a given value."""
        def __init__(self, mask_probability, mask_value):
            super().__init__()
            self._mask_probability = mask_probability
            self._mask_value = mask_value

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            if self.training and self._mask_probability:
                mask = torch.rand_like(inputs, dtype=torch.float32)
                inputs = torch.where(mask < self._mask_probability, self._mask_value, inputs)
            return inputs

    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset,
                 fasttext_embeddings: torch.Tensor, num_tags: int) -> None:
        super().__init__()

        self._word_masking = self.MaskElements(args.word_masking, MorphoDataset.UNK)
        self._dropout = torch.nn.Dropout(args.dropout)

        self._char_embedding = torch.nn.Embedding(num_embeddings=len(train.words.char_vocab), embedding_dim=args.cle_dim)

        self._char_rnn = torch.nn.GRU(input_size=args.cle_dim, hidden_size=args.cle_dim, bidirectional=True)

        self._word_embedding = torch.nn.Embedding.from_pretrained(fasttext_embeddings, freeze=False)
        fasttext_dim = fasttext_embeddings.shape[1]

        self._analyzer_embedding = torch.nn.Linear(num_tags, args.analyzer_dim)

        word_rnn_input_dim = fasttext_dim + 2 * args.cle_dim + args.analyzer_dim

        if args.rnn == "LSTM":
            self._word_rnn = torch.nn.LSTM(input_size=word_rnn_input_dim, hidden_size=args.rnn_dim, bidirectional=True)
        else:
            self._word_rnn = torch.nn.GRU(input_size=word_rnn_input_dim, hidden_size=args.rnn_dim, bidirectional=True)
        self._output_layer = torch.nn.LazyLinear(out_features=len(train.tags.string_vocab))

    def forward(self, word_ids: torch.Tensor, unique_words: torch.Tensor, word_indices: torch.Tensor,
                analyzer_features: torch.Tensor) -> torch.Tensor:
        word_ids = self._word_masking(word_ids)
        word_embedding = self._word_embedding(word_ids)

        char_embeddings = self._char_embedding(unique_words)

        char_lengths = (unique_words != MorphoDataset.PAD).sum(dim=1).cpu()

        packed_char_embeddings = torch.nn.utils.rnn.pack_padded_sequence(char_embeddings, lengths=char_lengths, batch_first=True, enforce_sorted=False)

        _, char_hidden = self._char_rnn(packed_char_embeddings)

        char_hidden = torch.cat([char_hidden[-2], char_hidden[-1]], dim=1)
        char_hidden = torch.nn.functional.embedding(word_indices, char_hidden)

        analyzer_hidden = self._analyzer_embedding(analyzer_features)
        analyzer_hidden = torch.relu(analyzer_hidden)

        hidden = torch.cat([word_embedding, char_hidden, analyzer_hidden], dim=2)
        hidden = self._dropout(hidden)

        lengths = (word_ids != MorphoDataset.PAD).sum(dim=1).cpu()

        packed = torch.nn.utils.rnn.pack_padded_sequence(hidden, lengths=lengths, batch_first=True, enforce_sorted=False)

        packed, _ = self._word_rnn(packed)

        hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)

        hidden = torch.cat([hidden[:, :, :self._word_rnn.hidden_size], hidden[:, :, self._word_rnn.hidden_size:]],dim=2)

        hidden = self._dropout(hidden)
        hidden = self._output_layer(hidden)
        hidden = hidden.permute(0, 2, 1)

        return hidden


def main(args: argparse.Namespace) -> None:
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    logdir = npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    morpho = MorphoDataset("czech_pdt")
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    tag_to_id = {
        morpho.train.tags.string_vocab.string(tag_id): tag_id
        for tag_id in range(len(morpho.train.tags.string_vocab))
    }

    print(f"Number of POS tags: {len(tag_to_id)}")

    fasttext_embeddings = load_fasttext_embeddings(morpho.train)

    print(f"fastText embedding dimension: {fasttext_embeddings.shape[1]}")

    train = Dataset(morpho.train, analyses, tag_to_id).dataloader(
        batch_size=args.batch_size, shuffle=True
    )

    dev = Dataset(morpho.dev, analyses, tag_to_id).dataloader(batch_size=args.batch_size)
    test = Dataset(morpho.test, analyses, tag_to_id).dataloader(batch_size=args.batch_size)

    model = Model(args, morpho.train, fasttext_embeddings, num_tags=len(tag_to_id))

    model.configure(
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4),
        loss=torch.nn.CrossEntropyLoss(ignore_index=morpho.PAD),
        metrics={
            "accuracy": torchmetrics.Accuracy("multiclass", num_classes=len(morpho.train.tags.string_vocab), ignore_index=morpho.PAD,)
        },
        logdir=logdir,
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, "tagger_competition.txt"), "w", encoding="utf-8") as predictions_file:
        predictions = model.predict(test, data_with_labels=True, as_numpy=True)
        for predicted_tags, words in zip(predictions, morpho.test.words.strings):
            for predicted_tag in predicted_tags[:, :len(words)].argmax(axis=0):
                print(morpho.train.tags.string_vocab.string(predicted_tag), file=predictions_file)
            print(file=predictions_file)

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)