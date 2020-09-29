import argparse
from abc import ABC
from pathlib import Path

import pytorch_lightning as pl


class BaseTransformer(pl.LightningModule, ABC):
    def __init__(
            self,
            hparams: argparse.Namespace
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        self.hparams = hparams  # TODO: move to self.save_hyperparameters()
        self.step_count = 0
        self.tfmr_ckpts = {}
        self.output_dir = Path(self.hparams.output_dir)
