import argparse
import glob
import os

import torch
from torch.nn import Softmax
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

from annotation_data import encode_sentence
from bert_classificator import Transformer
from lightning_base import add_generic_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser)
    parser = Transformer.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    model_name = 'allenai/scibert_scivocab_uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        'output_bert/best_tfmr/pytorch_model.bin', from_tf=False, config=config)
    text = """In particular , we use classic unsupervised IR models as a weak supervision signal for training deep 
    neural ranking models . """
    encoded_text = encode_sentence(text, return_tensors="pt")
    with torch.no_grad():
        preds = model(**encoded_text)[0].detach()

    m = Softmax(dim=1)
    #print(preds)
    print(m(preds).tolist())
