import argparse
import glob
import logging
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig, AutoModel

import annotation_data
from lightning_base import BaseTransformer, generic_train, add_generic_args


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = AutoConfig.from_pretrained("allenai/scibert_scivocab_uncased", output_hidden_states=True)
bert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', config=config)
device = torch.device('cuda')
bert_model.to(device)


class KimCNN(BaseTransformer):
    """
    From: https://romanorac.github.io/machine/learning/2019/12/02/identifying-hate-speech-with-bert-and-cnn.html

    Steps of KimCNN:
    1. Take a word embedding on the input [n x m], where n represents the maximum number of words in a sentence and
    m represents the length of the embedding.
    2. Apply convolution operations on embeddings. It uses multiple convolutions of different sizes [2 x m], [3 x m]
    and [4 x m]. The intuition behind this is to model combinations of 2 words, 3 words, etc. Note, that convolution
    width is m - the size of the embedding. This is different from CNNs for images as they use square convolutions
    like [5 x 5]. This is because [1 x m] represents a whole word and it doesn't make sense to run a convolution
    with a smaller kernel size (eg. a convolution on half of the word).
    3. Apply Rectified Linear Unit (ReLU) to add the ability to model nonlinear problems.
    4. Apply 1-max pooling to down-sample the input representation and to help to prevent overfitting. Fewer
    parameters also reduce computational cost.
    5. Concatenate vectors from previous operations to a single vector.
    6. Add a dropout layer to deal with overfitting.
    7. Apply a softmax function to distribute the probability between classes. Our network differs here because we
    are dealing with a multilabel classification problem - each comment can have multiple labels (or none). We use a
    sigmoid function, which scales logits between 0 and 1 for each class. This means that multiple classes can be
        predicted at the same time.
    """

    def __init__(self, hparams):
        """
        :param hparams: hyper parameters.
        :param embed_num: represents the maximum number of words in a sentence.
        :param embed_dim: represents the size of BERT embedding (768).
        :param class_num: is the number of labels (2, used and not-used).
        :param kernel_num: is the number of filters for each convolution operation (eg. 3 filters for [2 x m]
        convolution)
        :param kernel_sizes: of convolutions. Eg. look at combinations 2 words, 3 words, etc.
        :param dropout: is the percentage of randomly set hidden units to 0 at each update of the training phase.
        :param static: True means that we don't calculate gradients of embeddings and they stay static. If we set it to
        False, it would increase the number of parameters the model needs to learn and it could overfit.
        """
        super(KimCNN, self).__init__(hparams)

        V = self.hparams.embed_num
        D = self.hparams.embed_dim
        C = self.hparams.class_num
        Co = self.hparams.kernel_num
        Ks = self.hparams.kernel_sizes

        self.static = self.hparams.static
        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(self.hparams.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        output = self.sigmoid(logit)
        return output

    def prepare_data(self):
        """Called to initialize data. Use the call to construct features"""
        create = False
        for mode in ["train", "dev"]:
            cached_features_file = self._feature_file(mode)
            if not os.path.exists(cached_features_file):
                create = True
        if not create:
            return
        logger.info("Creating features from dataset file")
        # Load the annotation features and labels from the .csv file
        df = annotation_data.load_annotation_data(self.hparams.annotation_file)
        # Encode sentences into BERT tokens
        features = annotation_data.get_bert_embeddings(bert_model, df, self.hparams.with_context,
                                                       self.hparams.with_section_names)
        # Split into train and test data set
        train_set, dev_set = annotation_data.get_train_test_data(features)

        sets = {"train": train_set, "dev": dev_set}
        for mode in ["train", "dev"]:
            cached_features_file = self._feature_file(mode)
            if not os.path.exists(cached_features_file):
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(sets[mode], cached_features_file)

    def load_dataset(self, mode, batch_size):
        """Load datasets. Called after prepare data."""

        # We test on dev set to compare to benchmarks without having to submit to GLUE server
        mode = "dev" if mode == "test" else mode
        cached_features_file = self._feature_file(mode)
        logger.info("Loading features from cached file %s", cached_features_file)
        # Load the cached BERT tokens
        features = torch.load(cached_features_file)

        # Concatenate all input ids, attention masks, token type ids and labels
        all_input_ids = torch.tensor([f.input_tokens for f in features], dtype=torch.float32)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

        return DataLoader(
            TensorDataset(all_input_ids, all_labels),
            batch_size=batch_size,
            shuffle=mode == "train",
            num_workers=16
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        self.opt = optimizer
        return optimizer

    def training_step(self, batch, batch_idx):
        tokens = batch[0]
        labels = batch[1]

        logits = self.forward(tokens)
        # We use CrossEntropyLoss for binary classifications.
        # .view(-1) converts one-hot encoding to a single dimensional array
        loss = F.cross_entropy(logits.view(-1, 2), labels.view(-1))

        tensorboard_logs = {"loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        tokens = batch[0]
        labels = batch[1]

        logits = self(tokens)
        loss = F.cross_entropy(logits.view(-1, 2), labels.view(-1))
        out_label_ids = labels.detach().cpu().numpy()
        out_pred = logits.detach().cpu().numpy()
        return {"val_loss": loss, "pred": out_pred, "target": out_label_ids}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser)
    parser = KimCNN.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    args.embed_num = 512
    args.embed_dim = 768
    args.class_num = 2
    args.kernel_num = 3
    args.kernel_sizes = [2, 3, 4]
    args.dropout = 0.5
    args.static = True

    model = KimCNN(args)
    trainer = generic_train(model, args)
    print("Training done")
    # Optionally, predict on dev set and write to output_dir
    if args.do_predict:
        checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpointepoch=*.ckpt"), recursive=True)))
        model = model.load_from_checkpoint(checkpoints[-1], **args.__dict__)
        trainer.test(model)
