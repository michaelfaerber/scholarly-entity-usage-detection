import argparse
import logging
import os
import random
from abc import ABC
from pathlib import Path
import datetime
from typing import Union, List, Dict

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from torch import Tensor
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)

import lightning_utils

logger = logging.getLogger(__name__)


def set_seed(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpus > 0:
        torch.cuda.manual_seed_all(args.seed)


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
        self.training_log_file = self.output_dir / f"training_log_{datetime.datetime.now()}.csv"
        self.validation_log_file = self.output_dir / f"validation_log_{datetime.datetime.now()}.csv"

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None, **kwargs):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()  # By default, PL will only step every epoch.
        lrs = {f"lr_group_{i}": lr for i, lr in enumerate(self.lr_scheduler.get_lr())}
        self.logger.log_metrics(lrs)

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def train_dataloader(self):
        train_batch_size = self.hparams.train_batch_size
        dataloader = self.load_dataset("train", train_batch_size)

        t_total = (
                (len(dataloader.dataset) // (train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        return self.load_dataset("dev", self.hparams.eval_batch_size)

    def test_dataloader(self):
        return self.load_dataset("test", self.hparams.eval_batch_size)

    def training_epoch_end(self, outputs):
        logs = {"loss": torch.stack([x["loss"] for x in outputs]).mean().detach().cpu()}
        self.log_epoch(self.training_log_file, logs)
        return logs

    def validation_epoch_end(self, outputs: list) -> dict:
        ret, preds, targets = lightning_utils.eval_end(outputs)
        self.log_epoch(self.validation_log_file, ret)
        logs = ret["log"]
        return {"val_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    def log_epoch(self, file, ret):
        if not file.exists():
            with open(file, "w") as f:
                f.write(";".join(k for k in ret) + "\n")
        with open(file, "a") as f:
            f.write(";".join(str(ret[k].item()) if torch.is_tensor(ret[k]) else str(ret[k]) for k in ret) + "\n")

    def test_epoch_end(self, outputs) -> dict:
        ret, predictions, targets = lightning_utils.eval_end(outputs)
        logs = ret["log"]
        # `val_loss` is the key returned by `eval_end()` but actually refers to `test_loss`
        return {"avg_test_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    def _feature_file(self, mode):
        return os.path.join(
            self.hparams.data_dir,
            "cached_{}_{}".format(
                mode,
                str(self.hparams.max_seq_length),
            ),
        )

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
        )
        parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--num_workers", default=4, type=int, help="kwarg passed to DataLoader")
        parser.add_argument(
            "--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform."
        )

        parser.add_argument("--train_batch_size", default=32, type=int)
        parser.add_argument("--eval_batch_size", default=32, type=int)

        parser.add_argument(
            "--max_seq_length",
            default=512,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
                 "than this will be truncated, sequences shorter will be padded.",
        )

        parser.add_argument(
            "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
        )

        parser.add_argument(
            "--data_dir",
            default=None,
            type=str,
            required=True,
            help="The input data dir.",
        )
        parser.add_argument(
            "--n_gpu",
            default=1,
            type=int
        )
        parser.add_argument(
            "--annotation_file",
            type=str,
            default="annotations_method.csv"
        )
        parser.add_argument(
            "--with_context",
            action="store_true",
            default=False
        )
        parser.add_argument(
            "--with_section_names",
            action="store_true",
            default=False
        )
        return parser


class LoggingCallback(pl.Callback):
    @rank_zero_only
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Validation results *****")
        metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                rank_zero_info("{} = {}\n".format(key, str(metrics[key])))

    @rank_zero_only
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        logger.info("***** Test results *****")
        metrics = trainer.callback_metrics
        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


def add_generic_args(parser) -> None:
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--n_tpu_cores", type=int, default=0)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--val_check_interval", default=1.0, type=float)


def generic_train(
        model: BaseTransformer,
        args: argparse.Namespace,
        early_stopping_callback=False,
        logger=True,  # can pass WandbLogger() here
        extra_callbacks=[],
        checkpoint_callback=None,
        logging_callback=None,
        **extra_train_kwargs
):
    # init model
    set_seed(args)
    odir = Path(model.hparams.output_dir)
    odir.mkdir(exist_ok=True)
    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=args.output_dir, prefix="checkpoint", monitor="f1", mode="max", save_top_k=1
        )
    if logging_callback is None:
        logging_callback = LoggingCallback()

    train_params = {}

    if args.fp16:
        train_params["use_amp"] = args.fp16
        train_params["amp_level"] = args.fp16_opt_level

    if args.n_tpu_cores > 0:
        global xm

        train_params["num_tpu_cores"] = args.n_tpu_cores
        train_params["gpus"] = 0

    if args.gpus > 1:
        train_params["distributed_backend"] = "ddp"

    trainer = pl.Trainer(
        logger=logger,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.gpus,
        max_epochs=args.num_train_epochs,
        early_stop_callback=early_stopping_callback,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback,
        callbacks=[logging_callback] + extra_callbacks,
        fast_dev_run=args.fast_dev_run,
        val_check_interval=args.val_check_interval,
        weights_summary=None,
        resume_from_checkpoint=args.resume_from_checkpoint,
        **train_params,
    )

    if args.do_train:
        trainer.fit(model)
    trainer.logger.log_hyperparams(args)
    trainer.logger.save()
    return trainer
