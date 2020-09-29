import argparse
import glob
import logging
import os
from typing import Dict, Any

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

import annotation_data
from lightning_base import BaseTransformer, add_generic_args, generic_train


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Transformer(BaseTransformer):
    """
    Class is responsible for training and prediction of the BertSequenceClassification Task.
    See: https://pytorch-lightning.readthedocs.io/en/stable/introduction_guide.html
    """

    # Mode will be used to select the correct BERT model that should be fine-tuned
    mode = "sequence-classification"
    model_name = "allenai/scibert_scivocab_uncased"

    def __init__(self, hparams, model_path=None):
        super().__init__(hparams)
        cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None
        self.config = AutoConfig.from_pretrained(
            self.model_name,
            cache_dir=cache_dir
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path if model_path else self.model_name,
            from_tf=False,
            config=self.config,
            cache_dir=cache_dir,
        )

    def forward(self, **inputs):
        return self.model(**inputs)

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
        features = annotation_data.encode_sentences(df, self.hparams.with_context, self.hparams.with_section_names)
        # Split into train and test data set
        train_set, dev_set = annotation_data.get_train_test_data(features)

        sets = {"train": train_set, "dev": dev_set}
        for mode in ["train", "dev"]:
            cached_features_file = self._feature_file(mode)
            if not os.path.exists(cached_features_file):
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(sets[mode], cached_features_file)

    def load_dataset(self, mode, batch_size):
        """
        Load datasets. Called after prepare data.
        Loads all annotation data from the annotations_method.csv, splits the
        data into a train and test set and converts every entry into BERT tokens.
        The tokens are then cached for later use.
        """

        # We test on dev set to compare to benchmarks without having to submit to GLUE server
        mode = "dev" if mode == "test" else mode
        cached_features_file = self._feature_file(mode)
        logger.info("Loading features from cached file %s", cached_features_file)
        # Load the cached BERT tokens
        features = torch.load(cached_features_file)

        # Concatenate all input ids, attention masks, token type ids and labels
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

        return DataLoader(
            TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels),
            batch_size=batch_size,
            shuffle=mode == "train",
            num_workers=16
        )

    def training_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3]}

        # Call forward on BertForSequenceClassification in modeling_bert.py.
        outputs = self(**inputs)
        # loss is CrossEntropyLoss for binary classification tasks such as ours
        loss = outputs[0]

        tensorboard_logs = {"loss": loss, "rate": self.lr_scheduler.get_last_lr()[-1]}
        # PyTorch-lightning performs a backward using the resulting loss.
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3]}

        # Call forward on BertForSequenceClassification with torch.no_grad
        outputs = self(**inputs)
        # outputs is (loss), logits, (hidden_states), (attentions)
        tmp_eval_loss, logits = outputs[:2]
        # preds is one-hot encoding of the predicted sentence label
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()

        return {"val_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids}

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = self.output_dir.joinpath("best_tfmr")
        save_path.mkdir(exist_ok=True)
        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tfmr_ckpts[self.step_count] = save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser)
    parser = Transformer.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    model = Transformer(args)
    trainer = generic_train(model, args)
    print("Training done")
    # Optionally, predict on dev set and write to output_dir
    if args.do_predict:
        checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpointepoch=*.ckpt"), recursive=True)))
        setattr(args, 'model_path', os.path.join(args.output_dir, "best_tfmr/pytorch_model.bin"))
        model = model.load_from_checkpoint(checkpoints[-1], **args.__dict__)
        trainer.test(model)
