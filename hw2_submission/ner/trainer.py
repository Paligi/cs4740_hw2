# AUTO-GENERATED (DO NOT MODIFY)
# NET IDS: RT529,YC2838

import logging
from dataclasses import dataclass
from itertools import chain
from typing import Union, List, Optional, Dict, Tuple

import numpy as np
import torch
from datasets import Dataset
from rich.progress import track
from sklearn.utils import class_weight
from torch import nn
from torch.utils.data import DataLoader

from ner.data_processing.constants import NER_ENCODING_MAP, PAD_NER_TAG, PAD_TOKEN, UNK_TOKEN
from ner.data_processing.data_collator import DataCollator
from ner.data_processing.tokenizer import Tokenizer
from ner.models.ner_predictor import NERPredictor
from ner.nn.module import Module
from ner.utils.metrics import compute_loss, compute_metrics
from ner.utils.tracker import Tracker
from ner.utils.utils import get_named_entity_spans


class Trainer(object):
    def __init__(
        self,
        model: Module,
        optimizer: torch.optim.Optimizer,
        data_collator: DataCollator,
        train_data: Dataset,
        val_data: Optional[Dataset] = None,
        grad_clip_max_norm: Optional[float] = None,
        use_class_weights: bool = False,
        class_weights: Optional[Union[List, np.ndarray, torch.Tensor]] = None,
        tracker: Optional[Tracker] = None,
        device: torch.device = torch.device("cpu"),
        label_colname="NER",
    ) -> None:
        """
        Creates a :py:class:`~ner.trainers.trainer.Trainer` to train a neural network (here, FFNN or RNN).

        Parameters
        ----------
        model : Module
            The neural model to train: :py:class:`~ner.models.ner_predictor.NERPredictor`.
        optimizer : torch.optim.Optimizer
            The optimizer to be used in training.
        data_collator : DataCollator
            The data collator to collate data into batches for batched processing.
        train_data : Dataset
            The training data (arrow format) to train the model.
        val_data : Optional[Dataset], default: None
            The validation data (arrow format) to evaluate the model.
        grad_clip_max_norm : Optional[float], default: None
            The maximum norm to use in gradient clipping.
        use_class_weights : bool, default: True
            Whether to use class weights in the loss computation.
        class_weights : Optional[Union[List, np.ndarray, torch.Tensor]], default: None
            The class weights to use in the loss computation. If ``use_class_weights = True``, but ``class_weights``
            are not specified, class weights are automatically computed using class distribution in the training data.
        tracker : Optional[Tracker], default: None
            Tracker to use for logging.
        device : torch.device, default: torch.device("cpu")
            The device (e.g., cuda) to be used for training and validation.
        label_colname : str, default: "NER"
            The name of the column in the arrow dataset that contains the NER labels.
        """
        super().__init__()

        self.device = device

        self.model = model.to(device)
        self.optimizer = optimizer

        self.labels_ignore_idx = NER_ENCODING_MAP[PAD_NER_TAG]
        self.other_ner_tag_idx = NER_ENCODING_MAP["O"]

        self.train_data = train_data
        self.val_data = val_data
        self.data_collator = data_collator
        self.grad_clip_max_norm = grad_clip_max_norm
        if use_class_weights:
            class_weights = class_weights if class_weights else self._compute_class_weights(train_data, label_colname)
            # PyTorch FloatTensor doesn't support device: https://github.com/pytorch/pytorch/issues/20122.
            class_weights = torch.Tensor(class_weights).to(device)
            logging.info(f"class weights: {class_weights}")
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.labels_ignore_idx, weight=class_weights)

        self.tracker = tracker
        self._epoch = 0

    @staticmethod
    def _compute_class_weights(train_data, label_colname="NER"):
        """
        Takes in the training data and computes the class weights for the loss function using class distribution
        in the training data.

        Parameters
        ----------
        train_data : Dataset
            The training data (arrow format) used to train the model.
        label_colname : str
            The name of the column in the arrow dataset that contains the NER labels.

        Returns
        -------
        class_weights : np.ndarray
            The computed class weights for the loss function (as a numpy array).
        """
        train_labels = list(chain(*train_data[label_colname]))
        class_weights = class_weight.compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
        return class_weights

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Saves the training epoch, model state, and optimizer state to the given path.

        Parameters
        ----------
        checkpoint_path : str
            Path to save the checkpoint to.
        """
        torch.save(
            {
                "epoch": self._epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )

    def from_checkpoint(self, checkpoint_path: str) -> None:
        """
        Loads the epoch, model state, and optimizer state from the given path.

        Parameters
        ----------
        checkpoint_path : str
            Path to load the checkpoint from.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._epoch = checkpoint["epoch"] + 1

    def _train_epoch(self, dataloader) -> Dict[str, float]:
        """
        Trains the model for one epoch on the given dataloader.

        Parameters
        ----------
        dataloader : DataLoader
            The (train) Dataloader used in training the model.

        Returns
        -------
        Dict[str, float]
            A dictionary of metrics, includes loss, precision, recall, accuracy, F1, and weighted-average of
            entity-level F1 scores.
        """
        metrics = {"loss": [], "precision": [], "recall": [], "accuracy": [], "f1": [], "entity_f1": []}

        self.model.train()

        for batch in dataloader:
            self.optimizer.zero_grad()
            preds = self.model(batch["input_ids"].to(self.device))
            loss = compute_loss(self.loss_fn, preds, batch["labels"].to(self.device))
            loss.backward()
            if self.grad_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
            self.optimizer.step()
            batch_metrics = compute_metrics(
                preds=preds.to(device="cpu"),
                labels=batch["labels"],
                padding_mask=batch["padding_mask"],
                other_ner_tag_idx=self.other_ner_tag_idx,
                average="weighted",
            )
            for key in metrics:
                metrics[key].append(batch_metrics[key]) if key != "loss" else metrics[key].append(loss.item())

        average_metrics = {metric: np.average(score) for metric, score in metrics.items()}
        return average_metrics

    @torch.no_grad()
    def _eval_epoch(self, dataloader) -> Dict[str, float]:
        """
        Evaluates the model for one epoch on the given dataloader.

        Parameters
        ----------
        dataloader : DataLoader
            The (validation) Dataloader used in evaluating the trained model.

        Returns
        -------
        Dict[str, float]
            A dictionary of metrics, includes loss, precision, recall, accuracy, F1, and weighted-average of
            entity-level F1 scores.
        """
        metrics = {"loss": [], "precision": [], "recall": [], "accuracy": [], "f1": [], "entity_f1": []}

        self.model.eval()

        for batch in dataloader:
            preds = self.model(batch["input_ids"].to(self.device))
            loss = compute_loss(self.loss_fn, preds, batch["labels"].to(self.device))
            batch_metrics = compute_metrics(
                preds=preds.to(device="cpu"),
                labels=batch["labels"],
                padding_mask=batch["padding_mask"],
                other_ner_tag_idx=self.other_ner_tag_idx,
                average="weighted",
            )
            for key in metrics:
                metrics[key].append(batch_metrics[key]) if key != "loss" else metrics[key].append(loss.item())

        average_metrics = {metric: np.average(score) for metric, score in metrics.items()}
        return average_metrics

    def train_and_eval(
        self, batch_size: int = 128, num_epochs: int = 8, checkpoint_every: int = 1, num_workers: int = 0
    ) -> None:
        """
        Trains and evaluates the model for the given number of epochs.

        Parameters
        ----------
        batch_size : int, default: 128
            The batch size to be used in training and evaluation.
        num_epochs : int, default: 8
            The total number of epochs to train for.
        checkpoint_every : int, default: 1
            The frequency (of epochs) of saving a checkpoint.
        num_workers : int, default: 0
            The number of workers to use for data loading; enable multiprocess data loading by simply setting the
            argument ``num_workers`` to a positive value.
        """
        train_dataloader = DataLoader(
            self.train_data,
            collate_fn=self.data_collator,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )
        val_dataloader = None
        if self.val_data is not None:
            val_dataloader = DataLoader(
                self.val_data,
                collate_fn=self.data_collator,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=num_workers,
            )

        for epoch in range(num_epochs):
            train_metrics = self._train_epoch(train_dataloader)
            val_metrics = self._eval_epoch(val_dataloader) if val_dataloader is not None else None
            if self.tracker is not None:
                self.tracker.log_metrics(epoch=self._epoch, split="train", metrics=train_metrics)
                if val_metrics is not None:
                    self.tracker.log_metrics(epoch=self._epoch, split="val", metrics=val_metrics)
                if (epoch + 1) % checkpoint_every == 0:
                    self.tracker.save_checkpoint(self, epoch=self._epoch)
            self._epoch = self._epoch + 1

        if self.tracker:
            self.tracker.save_model(self.model)

    @staticmethod
    @torch.no_grad()
    def test(
        test_data: Dataset,
        data_collator: DataCollator,
        model: Module,
        batch_size: int = 128,
        num_workers: int = 0,
        index_colname: str = "index",
        device=torch.device("cpu"),
    ) -> Dict[str, List[Tuple[int]]]:
        """
        Tests the trained model on the given test (unseen) data.

        Parameters
        ----------
        test_data : Dataset
            The test data (arrow format) to test the model.
        data_collator : DataCollator
            The data collator to collate data into batches for batched processing.
        model : Module
            The neural model to test: :py:class:`~ner.models.ner_predictor.NERPredictor`.
        batch_size : int, default: 128
            The batch size to be used in training and evaluation.
        num_workers : int, default: 0
            The number of workers to use for data loading; enable multiprocess data loading by simply setting the
            argument ``num_workers`` to a positive value.
        index_colname : str, default: "index"
            The name of the column in the arrow dataset that contains the token indices.
        device : torch.device, default: torch.device("cpu")
            The device (e.g., cuda) to be used for testing.

        Returns
        -------
        Dict[str, List[Tuple[int]]]
            A dictionary of named-entity spans with keys being "LOC", "PER", "MISC", and "ORG".
        """
        test_dataloader = DataLoader(
            test_data,
            collate_fn=data_collator,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        token_idxs = list(chain.from_iterable(test_data[index_colname]))
        all_preds = []

        model = model.to(device)
        model.eval()
        for batch in track(test_dataloader, description=f"test"):
            # preds: (batch_size, max_length, output_dim)
            preds = model(input_ids=batch["input_ids"].to(device))

            padding_mask = batch["padding_mask"]
            if padding_mask.dtype == torch.long or padding_mask.dtype == torch.int:
                padding_mask = torch.BoolTensor(padding_mask == 1)
            mask = ~padding_mask.view(-1)

            preds = preds.view(-1, preds.shape[-1]).argmax(dim=-1)
            all_preds.append(preds[mask].cpu().numpy().squeeze())
        preds_dict = get_named_entity_spans(encoded_ner_ids=np.concatenate(all_preds), token_idxs=token_idxs)
        return preds_dict
