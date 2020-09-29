import argparse
import glob
import os

import torch
from torch.nn import Softmax
from transformers import AutoConfig, AutoModel

from annotation_data import encode_sentence
from bert_cnn_classificator import KimCNN
from lightning_base import add_generic_args

config = AutoConfig.from_pretrained("allenai/scibert_scivocab_uncased", output_hidden_states=True)
bert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', config=config)
device = torch.device('cuda')
bert_model.to(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser)
    parser = KimCNN.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpointepoch=*.ckpt"), recursive=True)))
    checkpoint = checkpoints[-1]
    print("Using checkpoint", checkpoint)
    model = KimCNN.load_from_checkpoint(checkpoint)
    model.to(device)
    text = """In particular , we did not use classic unsupervised IR models as a weak supervision signal for training deep 
    neural ranking models . """
    inputs = encode_sentence(text, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    with torch.no_grad():
        preds = bert_model(**inputs)[0]
        preds = model(preds)

    m = Softmax(dim=1)
    #print(preds)
    print(m(preds).tolist())
