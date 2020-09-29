import dataclasses
import json
import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from typing import Optional, Union

import pandas as pd
import torch
from transformers import AutoTokenizer
from transformers.data.processors.utils import InputFeatures

logger = logging.getLogger(__name__)

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')


def load_annotation_data(file_path, sep=";"):
    """
    Loads the annotation data from the given file_path.

    :param file_path: the path to a csv file.
    :param sep: CSV separator.
    :return: X, y: features and label entries.
    """
    df = pd.read_csv(file_path, sep=sep)
    df.drop(['used_Felix', 'Anmerkung'], axis=1, inplace=True)
    filtered_df = df[(df['used'] == "1") | (df['used'] == "0")]
    filtered_df = filtered_df.sample(frac=1, random_state=42)
    df_0 = filtered_df[filtered_df['used'] == "0"]
    df_1 = filtered_df[filtered_df['used'] == "1"]
    if len(df_0) < len(df_1):
        df_1 = df_1[0:len(df_0)]
    else:
        df_0 = df_0[0:len(df_1)]

    concat_df = pd.concat([df_0, df_1])
    result = concat_df[['ner', 'sentence', 'pre_sentence', 'post_sentence', 'used', 'section_name']]
    print("Using {} positive and {} negative labels = {} total".format(len(df_1), len(df_0), len(result)))
    return result


def encode_sentences(df, with_context, with_section_names):
    """
    Encodes a list of sentences into BERT tokens and returns a list of feature-label combinations.
    InputFeatures contains input_ids, attention_masks, token_type_ids and the label.

    :param df: a data frame of features and labels.
    :param with_context: True, if the context (pre and post sentence) should be considered.
    :param with_section_names: True, if the section name should be considered.
    :return: a list of InputFeatures where each element is a feature-label combination.
    """
    features = []
    for entry in df.iterrows():
        sentence = get_context(entry) if with_context else entry[1]['sentence']
        if with_section_names:
            sentence = entry[1]['section_name'] + " " + sentence
        inputs = encode_sentence(sentence)
        features.append(InputFeatures(**inputs, label=int(entry[1]['used'])))

    return features


def encode_sentence(sentences, return_tensors=None):
    """
    Encodes a sentence into BERT tokens.
    :param sentences: the sentence that should be encoded.
    :param return_tensors: None to return python numbers, pt for PyTorch tensors.
    :return: a dictionary containing the encoded sequence or sequence pair and additional information:
        the mask for sequence classification and the overflowing elements if a ``max_length`` is specified.
    """
    return tokenizer.encode_plus(
        sentences,
        add_special_tokens=True,
        return_tensors=return_tensors,
        pad_to_max_length=True,
        max_length=512,
        truncation=True,
        truncation_strategy="only_second",
        return_attention_mask=True
    )


def get_context(entry):
    context = ""
    context += entry[1]['sentence']
    if len(str(entry[1]['pre_sentence'])) > 5:
        if len(context) > 0:
            context += " "
        context += entry[1]['pre_sentence']
    if len(str(entry[1]['post_sentence'])) > 5:
        if len(context) > 0:
            context += " "
        context += entry[1]['post_sentence']
    return context


def get_bert_embeddings(bert_model, df, with_context, with_section_names):
    """
    Calculates the bert embeddings for all tokens in every data frame entry.
    :param bert_model:
    :param df: a data frame of features and labels.
    :param with_context: True, if the context (pre and post sentence) should be considered.
    :param with_section_names: True, if the section name should be considered.
    :return: a list of features, where each entry consists of 512x768 bert token embeddings and a single label.
    """
    features = []

    device = torch.device('cuda')
    for entry in df.iterrows():
        sentence = get_context(entry) if with_context else entry[1]['sentence']
        if with_section_names:
            sentence = entry[1]['section_name'] + " " + sentence
        inputs = encode_sentence(sentence, return_tensors="pt")
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        with torch.no_grad():
            # Word embeddings for every *token*
            result = bert_model(**inputs)
            # Sequence of hidden-states at the output of the last layer of the model.
            predictions = result[0][0]  # a 512x768 tensor
            predictions = predictions.detach().cpu().tolist()
            features.append(BertFeatures(input_tokens=predictions, label=int(entry[1]['used'])))

    return features


def get_train_test_data(*features):
    """
    Performs a train/test data split for a given set of feature entries.

    :param features: a list where each element is a feature/label combination.
    :return: train_set, test_set: a subset of the given features parameter.
    """
    return train_test_split(*features, test_size=0.33, random_state=42, shuffle=True)


@dataclass(frozen=True)
class BertFeatures:

    input_tokens: torch.Tensor = None
    label: Optional[Union[int, float]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"
