from torchtext import data, datasets
from transformers import BertTokenizer

import torch
import numpy as np


def get_raw_imdb_data():
    """
    :return: train_data, valid_data, test_data

    for sentence in train_data:
        sentence.text (['i', 'am', 'so', ...], lengths)
        sentence.label (pos, neg)
    """
    TEXT = data.Field(tokenize='spacy', include_lengths=True)
    LABEL = data.LabelField(dtype=torch.float)

    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    train_data, valid_data = train_data.split()

    return train_data, valid_data, test_data


def bert_tokenized_data(tokenizer, data, max_seq_len=128, pad_to_max_len=True):
    sentences = [' '.join(s.text) for s in data]  # I am so ... good .
    labels = [torch.tensor([1]) if l.label == 'pos' else torch.tensor([0]) for l in data]  # [1, 0, 0, ... , 1, ...]

    sentences = [torch.tensor(tokenizer.encode_plus(s, max_length=max_seq_len, pad_to_max_length=pad_to_max_len)) for s
                 in sentences]
    return sentences['input_ids'], \
           sentences['token_type_ids'], \
           sentences['attention_mask'], \
           labels


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    sequence_a = "[CLS] This is a short sequence. [SEP]"
    padded_sequence_a = tokenizer.encode_plus(sequence_a, max_length=19, pad_to_max_length=True)
    print(tokenizer.decode(padded_sequence_a['input_ids']))
    print(padded_sequence_a)

    pass
